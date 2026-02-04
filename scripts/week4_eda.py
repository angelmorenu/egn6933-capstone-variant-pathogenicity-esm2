#!/usr/bin/env python3
"""Week 4: EDA + go/no-go checks on curated dataset.

Reads:
- Week4 curated Parquet (chr_pos_ref_alt, label, split, embedding, [GeneSymbol])
Writes:
- counts_by_split.tsv
- unique_genes_by_split.tsv (if GeneSymbol present)
- positive_rate_by_split.png
- go_no_go.json (with failures/warnings and summary stats)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Data class to hold go/no-go report information 
@dataclass(frozen=True)
class GoNoGo:
    created_at: str
    ok: bool
    failures: list[str]
    warnings: list[str]
    summary: dict[str, object]

# Argument parsing function 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 4: EDA + go/no-go checks for curated dataset.")
    p.add_argument(
        "--curated-parquet",
        default="data/processed/week4_curated_dataset.parquet",
        help="Curated Parquet produced by scripts/make_week4_curated_dataset.py",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed/week4_eda",
        help="Directory to write EDA outputs.",
    )
    p.add_argument("--min-positives-train", type=int, default=50)
    p.add_argument("--min-positives-val", type=int, default=10)
    p.add_argument("--min-positives-test", type=int, default=10)
    p.add_argument("--min-rows-train", type=int, default=500)
    p.add_argument("--min-rows-val", type=int, default=100)
    p.add_argument("--min-rows-test", type=int, default=100)
    p.add_argument(
        "--max-split-posrate-gap",
        type=float,
        default=0.25,
        help="Warn if max(split positive rate) - min(split positive rate) exceeds this threshold.",
    )
    return p.parse_args()

# Helper to summarize counts by split
def _split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split", dropna=False)["label"]
    out = g.agg(rows="count", positives="sum").reset_index()
    out["negatives"] = out["rows"] - out["positives"]
    out["positive_rate"] = (out["positives"] / out["rows"]).astype(float)
    return out.sort_values("split")

# Main function to run the script 
def main() -> None:
    args = parse_args()

    curated_path = Path(args.curated_parquet)
    if not curated_path.is_absolute():
        curated_path = _REPO_ROOT / curated_path
    if not curated_path.exists():
        raise FileNotFoundError(f"Missing curated parquet: {curated_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(curated_path)
    for required in ["chr_pos_ref_alt", "label", "split", "embedding"]:
        if required not in df.columns:
            raise ValueError(f"Curated dataset missing required column: {required}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    failures: list[str] = []
    warnings: list[str] = []

    # Basic integrity
    dup_keys = int(df["chr_pos_ref_alt"].duplicated().sum())
    if dup_keys:
        failures.append(f"duplicate chr_pos_ref_alt keys: {dup_keys}")

    missing_split = int(df["split"].isna().sum())
    if missing_split:
        failures.append(f"missing split values: {missing_split}")

    allowed_splits = {"train", "val", "test"}
    bad_splits = sorted(set(df["split"].astype(str).unique()) - allowed_splits)
    if bad_splits:
        failures.append(f"unexpected split values: {bad_splits}")

    # Embedding checks
    emb_lens = df["embedding"].map(lambda x: len(x) if isinstance(x, (list, tuple, np.ndarray)) else -1)
    embedding_dim = int(emb_lens.max())
    bad_len = int((emb_lens != embedding_dim).sum())
    if bad_len:
        failures.append(f"embeddings with wrong length: {bad_len} (expected {embedding_dim})")

    # Split/class summaries
    split_summary = _split_summary(df)
    split_summary.to_csv(out_dir / "counts_by_split.tsv", sep="\t", index=False)

    # Warn if class balance differs substantially across splits.
    if split_summary.shape[0] >= 2:
        rates = split_summary["positive_rate"].astype(float)
        gap = float(rates.max() - rates.min())
        if gap > float(args.max_split_posrate_gap):
            warnings.append(
                f"split positive-rate gap is large: {gap:.3f} > {args.max_split_posrate_gap:.3f}"
            )

    overall_rows = int(len(df))
    overall_pos = int(df["label"].sum())
    overall_rate = float(overall_pos / overall_rows) if overall_rows else float("nan")

    # Go/no-go thresholds
    thresholds = {
        "train": {"min_rows": args.min_rows_train, "min_positives": args.min_positives_train},
        "val": {"min_rows": args.min_rows_val, "min_positives": args.min_positives_val},
        "test": {"min_rows": args.min_rows_test, "min_positives": args.min_positives_test},
    }

    split_rows = {r["split"]: int(r["rows"]) for r in split_summary.to_dict(orient="records")}
    split_pos = {r["split"]: int(r["positives"]) for r in split_summary.to_dict(orient="records")}

    for split_name, th in thresholds.items():
        r = split_rows.get(split_name, 0)
        p = split_pos.get(split_name, 0)
        if r < th["min_rows"]:
            warnings.append(f"{split_name} rows below threshold: {r} < {th['min_rows']}")
        if p < th["min_positives"]:
            warnings.append(f"{split_name} positives below threshold: {p} < {th['min_positives']}")

    # Leakage check (if gene identifier exists)
    gene_col = "GeneSymbol" if "GeneSymbol" in df.columns else ("GeneID" if "GeneID" in df.columns else "")
    if gene_col:
        gene_split_n = df.groupby(gene_col, dropna=False)["split"].nunique()
        leaking_genes = int((gene_split_n > 1).sum())
        if leaking_genes:
            failures.append(f"{gene_col} appears in multiple splits: {leaking_genes}")

        gene_counts = (
            df.groupby("split")[gene_col]
            .nunique(dropna=True)
            .reset_index()
            .rename(columns={gene_col: f"unique_{gene_col}"})
        )
        gene_counts.to_csv(out_dir / "unique_genes_by_split.tsv", sep="\t", index=False)

    # Simple plot: positive rate by split
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(split_summary["split"], split_summary["positive_rate"], color="#4C72B0")
    ax.set_ylabel("Positive rate")
    ax.set_title("Class balance by split")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "positive_rate_by_split.png", dpi=160)
    plt.close(fig)

    ok = len(failures) == 0
    report = GoNoGo(
        created_at=datetime.now(timezone.utc).isoformat(),
        ok=ok,
        failures=failures,
        warnings=warnings,
        summary={
            "rows": overall_rows,
            "positives": overall_pos,
            "positive_rate": overall_rate,
            "embedding_dim": embedding_dim,
        },
    )

    (out_dir / "go_no_go.json").write_text(json.dumps(asdict(report), indent=2) + "\n")

    print("EDA outputs:")
    print(f"- {out_dir / 'counts_by_split.tsv'}")
    if gene_col:
        print(f"- {out_dir / 'unique_genes_by_split.tsv'}")
    print(f"- {out_dir / 'positive_rate_by_split.png'}")
    print(f"- {out_dir / 'go_no_go.json'}")
    print("")
    print(f"GO/NO-GO: {'GO' if ok else 'NO-GO'}")
    if failures:
        for x in failures:
            print(f"FAIL: {x}")
    if warnings:
        for x in warnings:
            print(f"WARN: {x}")


if __name__ == "__main__":
    main()
