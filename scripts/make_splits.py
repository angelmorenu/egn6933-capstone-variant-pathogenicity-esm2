from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from variant_embeddings.splits.chromosome_split import (
    make_chromosome_holdout_plan,
    search_balanced_chromosome_holdout_plan,
)


def _find_chrom_col(df: pd.DataFrame) -> str:
    for candidate in ["Chromosome", "chrom", "CHROM", "chr"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a chromosome column (expected one of Chromosome/chrom/CHROM/chr)")


def _find_label_col(df: pd.DataFrame) -> str:
    for candidate in ["label", "Label", "y", "target", "ClinSigSimple"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a label column (expected one of label/Label/y/target/ClinSigSimple)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chromosome-holdout splits.")
    parser.add_argument("--input", required=True, help="Path to a curated labeled CSV/Parquet")
    parser.add_argument("--output", required=True, help="Output path for CSV/Parquet with split column")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-search", action="store_true", help="Search for a more balanced split")
    parser.add_argument("--max-iters", type=int, default=2000)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    chrom_col = _find_chrom_col(df)
    label_col = _find_label_col(df)

    # Normalize labels if ClinSigSimple present
    if label_col == "ClinSigSimple":
        # ClinSigSimple in ClinVar variant_summary is numeric; 1 often indicates P/LP, 0 benign, but
        # project-specific label mapping should happen during curation.
        raise ValueError("Please curate ClinSigSimple into a binary `label` column before splitting")

    chroms = df[chrom_col].to_numpy()
    labels = df[label_col].to_numpy()

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"Label column `{label_col}` must be binary 0/1")

    if args.balanced_search:
        plan = search_balanced_chromosome_holdout_plan(
            chroms=chroms,
            labels=labels,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            max_iters=args.max_iters,
        )
    else:
        plan = make_chromosome_holdout_plan(
            chromosomes=chroms,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    def assign_split(chrom: object) -> str:
        c = str(chrom).replace("chr", "").strip()
        if c in plan.test_chroms:
            return "test"
        if c in plan.val_chroms:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df[chrom_col].map(assign_split)

    # Report
    summary = (
        df.groupby("split")[label_col]
        .agg(n="size", pos="sum", pos_rate="mean")
        .reset_index()
        .sort_values("split")
    )

    print("Chromosome plan:")
    print("  train:", ",".join(plan.train_chroms))
    print("  val:", ",".join(plan.val_chroms))
    print("  test:", ",".join(plan.test_chroms))
    print("\nSplit summary:")
    print(summary.to_string(index=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
