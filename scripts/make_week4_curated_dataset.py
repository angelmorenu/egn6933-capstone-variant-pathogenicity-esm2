#!/usr/bin/env python3
"""Week 4: Build a single curated dataset artifact for training.

Reads:
- Week2 training table TSV.GZ (chr_pos_ref_alt + label + metadata)
- Week2 embeddings NPY (row-aligned to the TSV)
- Week3 splits Parquet (chr_pos_ref_alt + split + grouping columns)

Writes:
- One Parquet with columns: chr_pos_ref_alt, label, split, GeneSymbol (if present), embedding
- A small meta JSON capturing input paths and basic checks
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Data class to hold metadata about the curated dataset
@dataclass(frozen=True)
class CuratedMeta:
    created_at: str
    inputs: dict[str, str]
    rows: int
    embedding_dim: int
    missing_split_rows: int
    duplicate_key_rows: int

# Argument parsing function
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 4: build curated Parquet (label+split+embedding).")
    p.add_argument(
        "--week2-table",
        default="data/processed/week2_training_table_strict.tsv.gz",
        help="Week2 TSV.GZ (must include chr_pos_ref_alt and label).",
    )
    p.add_argument(
        "--week2-embeddings",
        default="data/processed/week2_training_table_strict_embeddings.npy",
        help="Week2 embeddings NPY aligned to the Week2 table row order.",
    )
    p.add_argument(
        "--week3-splits",
        default="data/processed/week2_training_table_strict_splits.parquet",
        help="Week3 splits Parquet (must include chr_pos_ref_alt and split).",
    )
    p.add_argument(
        "--out-parquet",
        default="data/processed/week4_curated_dataset.parquet",
        help="Output Parquet path.",
    )
    p.add_argument(
        "--out-meta",
        default="data/processed/week4_curated_dataset_meta.json",
        help="Output meta JSON path.",
    )
    return p.parse_args()

# Helper to check for file existence and iCloud placeholders 
def _resolve_existing_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    icloud_stub = p.with_name("." + p.name + ".icloud")
    if icloud_stub.exists():
        raise FileNotFoundError(
            f"Missing file: {p}. Found iCloud placeholder: {icloud_stub}. "
            "Download the file locally (Finder) or re-run the generating script to recreate it."
        )

    raise FileNotFoundError(f"Missing file: {p}")

# Main function to run the script 
def main() -> None:
    args = parse_args()

    week2_table = _resolve_existing_path(args.week2_table)
    week2_embeddings = _resolve_existing_path(args.week2_embeddings)
    week3_splits = _resolve_existing_path(args.week3_splits)

    out_parquet = Path(args.out_parquet)
    out_meta = Path(args.out_meta)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(week2_table, sep="\t", compression="gzip")
    for required in ["chr_pos_ref_alt", "label"]:
        if required not in df.columns:
            raise ValueError(f"Week2 table missing required column: {required}")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    emb = np.load(week2_embeddings)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array; got shape={emb.shape}")
    if emb.shape[0] != len(df):
        raise ValueError(
            "Embeddings row count does not match Week2 table. "
            f"embeddings_rows={emb.shape[0]} table_rows={len(df)}"
        )

    df_splits = pd.read_parquet(week3_splits)
    for required in ["chr_pos_ref_alt", "split"]:
        if required not in df_splits.columns:
            raise ValueError(f"Week3 splits missing required column: {required}")

    keep_cols = [c for c in ["chr_pos_ref_alt", "split", "GeneSymbol", "GeneID"] if c in df_splits.columns]
    df = df.merge(df_splits[keep_cols], on="chr_pos_ref_alt", how="left")

    missing_split_rows = int(df["split"].isna().sum())
    if missing_split_rows:
        raise ValueError(
            f"Found {missing_split_rows} rows missing a split assignment after join. "
            "Check that Week2 and Week3 artifacts were generated from the same keys."
        )

    duplicate_key_rows = int(df["chr_pos_ref_alt"].duplicated().sum())
    if duplicate_key_rows:
        raise ValueError(
            f"Found {duplicate_key_rows} duplicate chr_pos_ref_alt keys in the merged dataset. "
            "Deduplicate upstream before building the curated dataset."
        )

    df["embedding"] = [row.astype(np.float32, copy=False).tolist() for row in emb]

    out_cols = [
        c
        for c in ["chr_pos_ref_alt", "label", "split", "GeneSymbol", "GeneID", "embedding"]
        if c in df.columns
    ]
    df[out_cols].to_parquet(out_parquet, index=False, engine="pyarrow", compression="zstd")

    meta = CuratedMeta(
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "week2_table": str(week2_table),
            "week2_embeddings": str(week2_embeddings),
            "week3_splits": str(week3_splits),
        },
        rows=int(len(df)),
        embedding_dim=int(emb.shape[1]),
        missing_split_rows=missing_split_rows,
        duplicate_key_rows=duplicate_key_rows,
    )
    out_meta.write_text(json.dumps(asdict(meta), indent=2) + "\n")

    print(f"Wrote: {out_parquet} ({len(df)} rows)")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
