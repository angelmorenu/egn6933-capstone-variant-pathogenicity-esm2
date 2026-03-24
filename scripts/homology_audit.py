"""Week 11 homology-aware leakage audit.

This script screens for potential cross-split biological leakage by measuring
embedding-level similarity between gene groups across train/val/test splits.

Because the curated dataset contains ESM2 embeddings (but not raw protein
sequences), this script uses cosine similarity between split-level gene
centroids as a homology proxy.

Outputs:
- JSON summary (`--out-json`)
- Human-readable text report (`--out-report`)

Example:
  python scripts/homology_audit.py \
    --data data/processed/week4_curated_dataset.parquet \
    --similarity-threshold 0.90 \
    --out-json results/homology_leakage_audit.json \
    --out-report results/homology_audit_report.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms

# Build group-level centroids by averaging variant embeddings within each group
# (e.g. gene symbol) and split.
def _build_gene_centroids(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str,
    min_group_size: int,
) -> pd.DataFrame:
    work = df[[group_col, "split"]].copy()
    work["row_idx"] = np.arange(len(work))

    rows: list[dict[str, Any]] = []
    for (split_name, group_name), sub_df in work.groupby(["split", group_col], dropna=False):
        if pd.isna(group_name):
            continue
        row_indices = sub_df["row_idx"].to_numpy(dtype=int)
        n = int(row_indices.size)
        if n < int(min_group_size):
            continue

        centroid = embeddings[row_indices].mean(axis=0)
        rows.append(
            {
                "split": str(split_name),
                "group": str(group_name),
                "n_variants": n,
                "centroid": centroid,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["split", "group", "n_variants", "centroid"])

    return pd.DataFrame(rows)


def _similarity_matrix(left_df: pd.DataFrame, right_df: pd.DataFrame) -> np.ndarray:
    if left_df.empty or right_df.empty:
        return np.empty((0, 0), dtype=np.float32)

    left_vec = np.stack(left_df["centroid"].to_list(), axis=0)
    right_vec = np.stack(right_df["centroid"].to_list(), axis=0)
    left_vec = _l2_normalize(left_vec)
    right_vec = _l2_normalize(right_vec)
    return left_vec @ right_vec.T


def _background_threshold(
    train_df: pd.DataFrame,
    base_threshold: float,
    quantile: float,
) -> float:
    if train_df.empty or len(train_df) < 3:
        return float(base_threshold)

    sim = _similarity_matrix(train_df, train_df)
    if sim.size == 0:
        return float(base_threshold)

    mask = ~np.eye(sim.shape[0], dtype=bool)
    off_diag = sim[mask]
    if off_diag.size == 0:
        return float(base_threshold)

    calibrated = float(np.quantile(off_diag, float(quantile)))
    return float(max(float(base_threshold), calibrated))


def _pair_records(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    similarity: np.ndarray,
    threshold: float,
    require_mutual_nn: bool,
) -> list[dict[str, Any]]:
    if similarity.size == 0:
        return []

    candidates = np.argwhere(similarity >= float(threshold))
    if candidates.size == 0:
        return []

    row_best = np.argmax(similarity, axis=1)
    col_best = np.argmax(similarity, axis=0)

    pairs: list[dict[str, Any]] = []
    for i, j in candidates:
        i = int(i)
        j = int(j)
        if require_mutual_nn and not (row_best[i] == j and col_best[j] == i):
            continue

        sim_val = float(similarity[i, j])
        left_row = left_df.iloc[i]
        right_row = right_df.iloc[j]
        pairs.append(
            {
                "left_group": str(left_row["group"]),
                "left_split": str(left_row["split"]),
                "left_n": int(left_row["n_variants"]),
                "right_group": str(right_row["group"]),
                "right_split": str(right_row["split"]),
                "right_n": int(right_row["n_variants"]),
                "cosine_similarity": sim_val,
            }
        )

    pairs.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    return pairs


def _pair_summary(similarity: np.ndarray, base_threshold: float, effective_threshold: float) -> dict[str, Any]:
    if similarity.size == 0:
        return {
            "total_pairs": 0,
            "n_above_base_threshold": 0,
            "rate_above_base_threshold": 0.0,
            "n_above_effective_threshold": 0,
            "rate_above_effective_threshold": 0.0,
            "similarity_q95": None,
            "similarity_q99": None,
            "similarity_max": None,
        }

    total = int(similarity.size)
    n_base = int((similarity >= float(base_threshold)).sum())
    n_eff = int((similarity >= float(effective_threshold)).sum())

    return {
        "total_pairs": total,
        "n_above_base_threshold": n_base,
        "rate_above_base_threshold": float(n_base / total),
        "n_above_effective_threshold": n_eff,
        "rate_above_effective_threshold": float(n_eff / total),
        "similarity_q95": float(np.quantile(similarity, 0.95)),
        "similarity_q99": float(np.quantile(similarity, 0.99)),
        "similarity_max": float(np.max(similarity)),
    }

# Identify groups that appear in multiple splits 
# (e.g. same gene symbol in train and test) and summarize their split membership and variant counts.
def _direct_group_overlap(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = df.groupby(group_col, dropna=False)["split"].nunique()
    leaked_groups = counts[counts > 1].index.tolist()

    for group in leaked_groups:
        if pd.isna(group):
            continue
        sub_df = df[df[group_col] == group]
        split_counts = sub_df["split"].value_counts().to_dict()
        rows.append(
            {
                "group": str(group),
                "splits": sorted(sub_df["split"].astype(str).unique().tolist()),
                "split_counts": {str(k): int(v) for k, v in split_counts.items()},
                "n_total": int(len(sub_df)),
            }
        )

    rows.sort(key=lambda row: row["n_total"], reverse=True)
    return rows

# Estimate the number and rate of potentially flagged variants in each 
# split based on the identified cross-split similar groups.
def _estimate_flagged_variants(
    df: pd.DataFrame,
    group_col: str,
    cross_split_pairs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    flagged_groups_by_split = {"train": set(), "val": set(), "test": set()}

    for _, pairs in cross_split_pairs.items():
        for pair in pairs:
            left_split = pair["left_split"]
            right_split = pair["right_split"]
            left_group = pair["left_group"]
            right_group = pair["right_group"]

            if left_split in flagged_groups_by_split:
                flagged_groups_by_split[left_split].add(left_group)
            if right_split in flagged_groups_by_split:
                flagged_groups_by_split[right_split].add(right_group)

    split_summary: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        total_n = int(len(split_df))
        group_set = flagged_groups_by_split[split_name]
        if group_set:
            flagged_df = split_df[split_df[group_col].isin(group_set)]
            flagged_n = int(len(flagged_df))
        else:
            flagged_n = 0

        split_summary[split_name] = {
            "total_variants": total_n,
            "flagged_variants": flagged_n,
            "flagged_rate": float(flagged_n / total_n) if total_n > 0 else 0.0,
            "flagged_groups": sorted(group_set),
        }

    return split_summary

# The main function orchestrates the loading of data, computation of centroids, similarity screening, and reporting.
def main() -> int:
    parser = argparse.ArgumentParser(description="Week 11 homology-aware leakage audit")
    parser.add_argument("--data", default="data/processed/week4_curated_dataset.parquet")
    parser.add_argument("--group-col", default="GeneSymbol")
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument(
        "--background-quantile",
        type=float,
        default=0.999,
        help="Calibrate effective threshold as max(similarity-threshold, train-train quantile)",
    )
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument("--top-k-pairs", type=int, default=50)
    parser.add_argument(
        "--allow-one-way",
        action="store_true",
        help="Allow one-way nearest-neighbor hits (default uses stricter mutual NN)",
    )
    parser.add_argument("--out-json", default="results/homology_leakage_audit.json")
    parser.add_argument("--out-report", default="results/homology_audit_report.txt")
    args = parser.parse_args()

    data_path = _resolve_path(args.data)
    out_json = _resolve_path(args.out_json)
    out_report = _resolve_path(args.out_report)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_parquet(data_path)
    required = {"embedding", "split", args.group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    x = np.asarray(df["embedding"].tolist(), dtype=np.float32)

    centroids = _build_gene_centroids(
        df=df,
        embeddings=x,
        group_col=args.group_col,
        min_group_size=int(args.min_group_size),
    )

    by_split = {
        split_name: centroids[centroids["split"] == split_name].reset_index(drop=True)
        for split_name in ["train", "val", "test"]
    }

    effective_threshold = _background_threshold(
        train_df=by_split["train"],
        base_threshold=float(args.similarity_threshold),
        quantile=float(args.background_quantile),
    )
    require_mutual_nn = not bool(args.allow_one_way)

    pair_defs = [
        ("train_vs_val", "train", "val"),
        ("train_vs_test", "train", "test"),
        ("val_vs_test", "val", "test"),
    ]

    cross_split_pairs_all: dict[str, list[dict[str, Any]]] = {}
    cross_split_pairs_top: dict[str, list[dict[str, Any]]] = {}
    pair_counts: dict[str, int] = {}
    pair_stats: dict[str, dict[str, Any]] = {}
    for pair_name, left_split, right_split in pair_defs:
        similarity = _similarity_matrix(by_split[left_split], by_split[right_split])
        pairs = _pair_records(
            left_df=by_split[left_split],
            right_df=by_split[right_split],
            similarity=similarity,
            threshold=effective_threshold,
            require_mutual_nn=require_mutual_nn,
        )
        cross_split_pairs_all[pair_name] = pairs
        cross_split_pairs_top[pair_name] = pairs[: int(args.top_k_pairs)]
        pair_counts[pair_name] = len(pairs)
        pair_stats[pair_name] = _pair_summary(
            similarity=similarity,
            base_threshold=float(args.similarity_threshold),
            effective_threshold=effective_threshold,
        )

    direct_overlap = _direct_group_overlap(df=df, group_col=args.group_col)
    flagged_by_split = _estimate_flagged_variants(
        df=df,
        group_col=args.group_col,
        cross_split_pairs=cross_split_pairs_all,
    )

    leak_score_test = float(flagged_by_split["test"]["flagged_rate"])

    result = {
        "data": str(data_path),
        "method": "embedding_centroid_cosine_proxy",
        "group_col": args.group_col,
        "similarity_threshold_base": float(args.similarity_threshold),
        "background_quantile": float(args.background_quantile),
        "similarity_threshold_effective": float(effective_threshold),
        "mutual_nearest_neighbor_required": bool(require_mutual_nn),
        "min_group_size": int(args.min_group_size),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "n_groups_per_split": {
            split_name: int(len(by_split[split_name])) for split_name in ["train", "val", "test"]
        },
        "direct_group_overlap": {
            "n_groups_in_multiple_splits": int(len(direct_overlap)),
            "groups": direct_overlap[: int(args.top_k_pairs)],
        },
        "cross_split_similarity": {
            "n_pairs_passing_filter": pair_counts,
            "pair_stats": pair_stats,
            "top_pairs": cross_split_pairs_top,
        },
        "flagged_variants_by_split": flagged_by_split,
        "test_leakage_rate_proxy": leak_score_test,
        "interpretation": {
            "note": (
                "This is an embedding-based homology proxy. Confirmatory sequence-level "
                "alignment (BLAST/Smith-Waterman) can be added in a follow-up pass."
            ),
            "suggested_action": (
                "If test_leakage_rate_proxy > 0.05, run strict sequence-level audit and "
                "consider re-evaluating with filtered test genes."
            ),
            "calibration_detail": (
                "Effective threshold is calibrated from train-vs-train background similarity "
                "and filtered with mutual nearest-neighbor matches by default."
            ),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    lines = [
        "Week 11 Homology-Aware Leakage Audit",
        "=" * 38,
        f"Dataset: {data_path}",
        f"Method: embedding centroid cosine proxy",
        f"Base threshold: {args.similarity_threshold}",
        f"Background quantile: {args.background_quantile}",
        f"Effective threshold: {effective_threshold:.6f}",
        f"Mutual nearest-neighbor filter: {require_mutual_nn}",
        f"Group column: {args.group_col}",
        "",
        "Split summary:",
    ]

    for split_name in ["train", "val", "test"]:
        split_row = flagged_by_split[split_name]
        lines.append(
            f"- {split_name}: flagged {split_row['flagged_variants']} / "
            f"{split_row['total_variants']} (rate={split_row['flagged_rate']:.4f})"
        )

    lines.extend(
        [
            "",
            "Pairs passing calibrated filter:",
            f"- train_vs_val: {pair_counts['train_vs_val']}",
            f"- train_vs_test: {pair_counts['train_vs_test']}",
            f"- val_vs_test: {pair_counts['val_vs_test']}",
            "",
            f"Direct group overlap across splits: {len(direct_overlap)} groups",
            "",
            "Interpretation:",
            f"- Test leakage rate proxy: {leak_score_test:.4f}",
            "- This is a proxy using embedding similarity; sequence-level audit is recommended",
        ]
    )

    out_report.write_text("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_report}")
    print(f"Test leakage rate proxy: {leak_score_test:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
