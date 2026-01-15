#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


def iter_pickle_objects(path: str) -> Iterable[Any]:
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def _is_empty_or_nan_array(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        arr = np.asarray(value)
    except Exception:
        return True
    if arr.size == 0:
        return True
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).all():
        return True
    return False


def _coerce_embedding(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _infer_binary_label(value: Any, numeric_threshold: float) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            v = float(value)
        except Exception:
            return None
        if np.isnan(v):
            return None
        return int(v >= numeric_threshold)

    if isinstance(value, str):
        v = value.strip().lower().replace("-", "_").replace(" ", "_")
        if v in {"pathogenic", "likely_pathogenic", "p", "lp", "true", "1"}:
            return 1
        if v in {"benign", "likely_benign", "b", "lb", "false", "0"}:
            return 0

        try:
            f = float(v)
            if np.isnan(f):
                return None
            return int(f >= numeric_threshold)
        except Exception:
            return None

    return None


def _normalize_label_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _infer_binary_label_with_policy(
    value: Any,
    numeric_threshold: float,
    label_policy: str,
    label_map: Optional[Mapping[str, Optional[int]]] = None,
) -> Optional[int]:
    """Infer a 0/1 label from `Pathogenicity`.

    label_policy:
      - strict: only (Benign|Likely_Benign|Pathogenic|Likely_Pathogenic); VUS-like values -> None
      - include_vus_leaning: map VUS_Benign/Benign_VUS -> 0 and VUS_Pathogenic/Pathogenic_VUS -> 1
    """
    if value is None:
        return None

    if isinstance(value, str):
        v = _normalize_label_key(value)

        if label_map is not None and v in label_map:
            return label_map[v]

        # Dylan Tan reference script (`FinTest_MLP.py`) mapping:
        # if np.isin("B", y): y = np.where(y == "B", 0, 1)
        
        # This means:
        # - reference_bp: only map B->0 and P->1; anything else -> None (safe)
        # - reference_bp_else_one: map B->0 and everything else -> 1 (exact np.where behavior)
        if label_policy in {"reference_bp", "reference_bp_else_one"}:
            if v == "b":
                return 0
            if v == "p":
                return 1
            if label_policy == "reference_bp_else_one":
                return 1
            return None

        if v in {"benign", "likely_benign"}:
            return 0
        if v in {"pathogenic", "likely_pathogenic"}:
            return 1

        if label_policy == "include_vus_leaning":
            if v in {"vus_benign", "benign_vus"}:
                return 0
            if v in {"vus_pathogenic", "pathogenic_vus"}:
                return 1

        return _infer_binary_label(value, numeric_threshold)

    return _infer_binary_label(value, numeric_threshold)


@dataclass(frozen=True)
class IngestResult:
    output_path: str
    n_rows: int
    n_cols: int
    n_dropped_empty_embedding: int
    n_dropped_bad_shape: int
    n_dropped_unlabeled: int
    label_missing: int


def ingest(
    pickle_file: str,
    out_dir: str,
    version: str,
    embed_mode: str,
    embedding_col: str,
    pathogenicity_col: str,
    id_col: str,
    numeric_threshold: float,
    label_policy: str,
    label_map: Optional[Mapping[str, Optional[int]]],
    drop_unlabeled: bool,
    out_prefix: str,
    source_dataset: str,
) -> IngestResult:
    records: List[Dict[str, Any]] = []
    for obj in iter_pickle_objects(pickle_file):
        if isinstance(obj, dict):
            records.append(obj)
        else:
            records.append({"_record": obj})

    if not records:
        raise ValueError(f"No records found in {pickle_file}")

    df = pd.DataFrame(records)

    if embedding_col not in df.columns:
        raise ValueError(f"Expected embedding column `{embedding_col}` not found. Columns={list(df.columns)}")

    before = len(df)
    df = df[~df[embedding_col].apply(_is_empty_or_nan_array)].copy()
    dropped_empty = before - len(df)

    bad_shape = 0
    if embed_mode == "expand":
        vectors: List[np.ndarray] = []
        for v in df[embedding_col].tolist():
            arr = _coerce_embedding(v)
            if arr.shape != (2560,):
                bad_shape += 1
                vectors.append(None)  # type: ignore[arg-type]
            else:
                vectors.append(arr.astype(np.float32, copy=False))

        ok_mask = [vv is not None for vv in vectors]
        df = df.loc[ok_mask].copy()
        vectors_ok = [vv for vv in vectors if vv is not None]

        emb_arrays = np.vstack(vectors_ok) if vectors_ok else np.empty((0, 2560), dtype=np.float32)
        emb_df = pd.DataFrame(emb_arrays, columns=[f"emb_{i:04d}" for i in range(2560)])
        emb_df.index = df.index
        df = pd.concat([df.drop(columns=[embedding_col]), emb_df], axis=1)

    elif embed_mode == "array":
        df[embedding_col] = df[embedding_col].apply(lambda v: _coerce_embedding(v).astype(np.float32, copy=False))
    else:
        raise ValueError("embed_mode must be one of: expand, array")

    df["source_dataset"] = source_dataset
    df["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    if pathogenicity_col in df.columns:
        labels = df[pathogenicity_col].apply(
            lambda v: _infer_binary_label_with_policy(v, numeric_threshold, label_policy, label_map=label_map)
        )
        df["label"] = labels.astype("Int64")
        label_missing = int(df["label"].isna().sum())
    else:
        df["label"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        label_missing = len(df)

    dropped_unlabeled = 0
    if drop_unlabeled and "label" in df.columns:
        before_drop = len(df)
        df = df[df["label"].notna()].copy()
        dropped_unlabeled = before_drop - len(df)
        label_missing = int(df["label"].isna().sum())

    if drop_unlabeled and len(df) == 0:
        print(
            "WARNING: 0 rows remain after dropping unlabeled rows. "
            "This usually means the chosen label mapping does not match the dataset's `Pathogenicity` values. "
            "Try a different `--label-policy`, provide `--label-map-json`, or re-run with `--no-drop-unlabeled`.",
            file=sys.stderr,
        )

    if id_col in df.columns and id_col != "variant_id":
        df["variant_id"] = df[id_col]
    elif "variant_id" not in df.columns:
        df["variant_id"] = pd.RangeIndex(start=0, stop=len(df))

    front_cols = [
        c
        for c in ["variant_id", id_col, pathogenicity_col, "label", "source_dataset", "ingested_at"]
        if c in df.columns
    ]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_prefix}_{version}.parquet")
    df.to_parquet(out_path, index=False)

    return IngestResult(
        output_path=out_path,
        n_rows=len(df),
        n_cols=len(df.columns),
        n_dropped_empty_embedding=dropped_empty,
        n_dropped_bad_shape=bad_shape,
        n_dropped_unlabeled=dropped_unlabeled,
        label_missing=label_missing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest ESM2 PrimateAI-style feature PKLs (saved record-by-record) into a Parquet table for modeling. "
            "By default, expands `Embedding` into 2560 numeric columns."
        )
    )
    parser.add_argument("pickle_file", help="Path to .pkl (e.g., data/Dylan Tan/esm2_selected_features.pkl)")
    parser.add_argument(
        "--out-dir",
        default="data/processed/esm2_primateai",
        help="Output directory for Parquet (default: data/processed/esm2_primateai)",
    )
    parser.add_argument(
        "--out-prefix",
        default="esm2_primateai",
        help="Output file prefix (default: esm2_primateai)",
    )
    parser.add_argument(
        "--source-dataset",
        default="esm2_primateai",
        help="Value to write into `source_dataset` column (default: esm2_primateai)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version suffix for the output filename (default: UTC timestamp)",
    )
    parser.add_argument(
        "--embed-mode",
        choices=["expand", "array"],
        default="expand",
        help="How to store embeddings (default: expand into 2560 columns)",
    )
    parser.add_argument(
        "--embedding-col",
        default="Embedding",
        help="Embedding column name (default: Embedding)",
    )
    parser.add_argument(
        "--pathogenicity-col",
        default="Pathogenicity",
        help="Pathogenicity column name (default: Pathogenicity)",
    )
    parser.add_argument(
        "--id-col",
        default="ID",
        help="ID column name (default: ID)",
    )
    parser.add_argument(
        "--numeric-threshold",
        type=float,
        default=0.5,
        help="If `Pathogenicity` is numeric (or numeric string), label=1 when >= threshold (default: 0.5).",
    )
    parser.add_argument(
        "--label-policy",
        choices=["strict", "include_vus_leaning", "reference_bp", "reference_bp_else_one"],
        default="strict",
        help=(
            "How to map Pathogenicity strings to 0/1 labels. "
            "strict keeps only Benign/Likely_Benign/Pathogenic/Likely_Pathogenic; "
            "include_vus_leaning also maps VUS_Benign/Benign_VUS->0 and VUS_Pathogenic/Pathogenic_VUS->1; "
            "reference_bp maps B->0 and P->1 (others unlabeled); "
            "reference_bp_else_one matches Dylan Tan's np.where(y==B,0,1) mapping."
        ),
    )
    parser.add_argument(
        "--label-map-json",
        default=None,
        help=(
            "Optional JSON file that maps Pathogenicity strings to labels {0,1,null}. "
            "Keys are normalized by lowercasing and replacing spaces/hyphens with underscores. "
            "If provided, it takes precedence over --label-policy for string values."
        ),
    )
    parser.add_argument(
        "--drop-unlabeled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows where label could not be inferred (default: true).",
    )

    args = parser.parse_args()

    label_map: Optional[Dict[str, Optional[int]]] = None
    if args.label_map_json:
        with open(args.label_map_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("--label-map-json must be a JSON object mapping strings to 0/1/null")
        normalized: Dict[str, Optional[int]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise ValueError("--label-map-json keys must be strings")
            if v is None:
                normalized[_normalize_label_key(k)] = None
            elif v in (0, 1):
                normalized[_normalize_label_key(k)] = int(v)
            else:
                raise ValueError("--label-map-json values must be 0, 1, or null")
        label_map = normalized

    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = ingest(
        pickle_file=args.pickle_file,
        out_dir=args.out_dir,
        version=version,
        embed_mode=args.embed_mode,
        embedding_col=args.embedding_col,
        pathogenicity_col=args.pathogenicity_col,
        id_col=args.id_col,
        numeric_threshold=args.numeric_threshold,
        label_policy=args.label_policy,
        label_map=label_map,
        drop_unlabeled=args.drop_unlabeled,
        out_prefix=args.out_prefix,
        source_dataset=args.source_dataset,
    )

    print(f"Wrote: {result.output_path}")
    print(
        "Summary: "
        f"rows={result.n_rows}, cols={result.n_cols}, "
        f"dropped_empty_embedding={result.n_dropped_empty_embedding}, "
        f"dropped_bad_shape={result.n_dropped_bad_shape}, "
        f"dropped_unlabeled={result.n_dropped_unlabeled}, "
        f"label_missing={result.label_missing}"
    )


if __name__ == "__main__":
    main()
