#!/usr/bin/env python3
# Week 3: Build leakage-aware train/val/test splits grouped by gene/protein identifier
import argparse
import csv
import gzip
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

"""
This script reads the Week-2 TSV (canonical chr_pos_ref_alt + labels), maps keys to a
ClinVar gene identifier (GeneSymbol or GeneID) via ClinVar variant_summary.txt.gz, then
writes split artifacts to disk.

Expected inputs
- Week-2 TSV.GZ produced by scripts/build_week2_training_table.py
- ClinVar variant_summary.txt.gz (used to map chr_pos_ref_alt -> GeneSymbol/GeneID)

Output
- <prefix>_splits.parquet
- <prefix>_{train,val,test}_idx.npy
- <prefix>_splits_meta.json

Notes
- Variants with missing gene mapping can be handled via --missing-gene-policy.
    'key' groups each missing row by its own key (no leakage, minimal coupling).
    'unknown' groups all missing rows together.
    'error' fails if any are missing.
    
- Split assignment is group-aware and targets split sizes (default).
"""

# Check if allele is a single-nucleotide variant (A, C, G, T) 
def is_snv_allele(allele: str) -> bool:
    a = (allele or "").strip().upper()
    return len(a) == 1 and a in {"A", "C", "G", "T"}

# Parse command-line arguments 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Week 3: Build leakage-aware train/val/test splits. "
            "All variants from the same gene/protein identifier are assigned to the same split. "
            "This script reads the Week-2 TSV (canonical chr_pos_ref_alt + labels), maps keys to a ClinVar gene identifier "
            "(GeneSymbol or GeneID) via ClinVar variant_summary.txt.gz, then writes split artifacts to disk."
        )
    )
    p.add_argument(
        "--input-tsv",
        default="data/processed/week2_training_table_strict.tsv.gz",
        help="Input Week-2 TSV.GZ produced by scripts/build_week2_training_table.py",
    )
    p.add_argument(
        "--clinvar-variant-summary",
        default="data/clinvar/variant_summary.txt.gz",
        help="ClinVar variant_summary.txt.gz (used to map chr_pos_ref_alt -> GeneSymbol/GeneID)",
    )
    p.add_argument(
        "--clinvar-group-field",
        choices=["GeneSymbol", "GeneID"],
        default="GeneSymbol",
        help="ClinVar field to group by when building leakage-aware splits (GeneSymbol or GeneID).",
    )
    p.add_argument(
        "--assembly",
        default="GRCh38",
        help="Assembly to keep when scanning variant_summary (GRCh38 or GRCh37). Use '' to disable filtering.",
    )
    p.add_argument(
        "--group-column",
        default="",
        help=(
            "Optional override: group on an existing input TSV column instead of ClinVar GeneSymbol/GeneID. "
            "If set, no ClinVar scan is performed."
        ),
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of rows targeted for train split (group-aware; approximate)",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of rows targeted for validation split (group-aware; approximate)",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of rows targeted for test split (group-aware; approximate)",
    )
    p.add_argument("--seed", type=int, default=13, help="Random seed for group shuffling")
    p.add_argument(
        "--method",
        choices=["size", "size+label"],
        default="size",
        help=(
            "Split assignment method. 'size' targets split sizes only (recommended default). "
            "'size+label' also tries to match overall positive rate, but may be constrained by large groups."
        ),
    )
    p.add_argument(
        "--missing-gene-policy",
        choices=["key", "unknown", "error"],
        default="error",
        help=(
            "What to do if a chr_pos_ref_alt key can't be mapped to the chosen ClinVar group field. "
            "'key' groups each missing row by its own key (no leakage, minimal coupling). "
            "'unknown' groups all missing rows together. 'error' fails if any are missing."
        ),
    )
    p.add_argument(
        "--missing-report-path",
        default="auto",
        help=(
            "Where to write a TSV report of rows that are missing a ClinVar group mapping. "
            "Default 'auto' writes <out_prefix>_missing_group_mapping.tsv. "
            "Use 'none' to disable writing the report."
        ),
    )
    p.add_argument(
        "--out-prefix",
        default="data/processed/week2_training_table_strict",
        help=(
            "Output prefix (writes <prefix>_splits.parquet, <prefix>_{train,val,test}_idx.npy, <prefix>_splits_meta.json)"
        ),
    )
    return p.parse_args()

# Resolve missing report path based on user input 
def resolve_missing_report_path(repo_root: Path, out_prefix: Path, raw: str) -> Optional[Path]:
    v = (raw or "").strip()
    if not v or v.lower() == "auto":
        return Path(str(out_prefix) + "_missing_group_mapping.tsv")
    if v.lower() in {"none", "false", "0"}:
        return None
    p = Path(v)
    return p if p.is_absolute() else (repo_root / p)

# Write a report of rows missing group mapping 
def write_missing_mapping_report(
    df: pd.DataFrame,
    missing_mask: pd.Series,
    out_path: Path,
    group_field: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols: list[str] = []
    for c in [
        "pickle_ID",
        "chr_pos_ref_alt",
        "Chromosome",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Pathogenicity",
        "label",
    ]:
        if c in df.columns:
            cols.append(c)

    report = df.loc[missing_mask, cols].copy()
    report.insert(0, "missing_group_field", group_field)
    report.to_csv(out_path, sep="\t", index=False)

# Load Week-2 TSV into DataFrame
def load_week2_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip")
    if "chr_pos_ref_alt" not in df.columns:
        raise ValueError("Input TSV is missing required column 'chr_pos_ref_alt'.")
    if "label" not in df.columns:
        raise ValueError("Input TSV is missing required column 'label'.")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df

# Map chr_pos_ref_alt keys to a ClinVar grouping field using ClinVar variant_summary.txt.gz
def map_key_to_group_value(
    variant_summary_gz: Path,
    needed_keys: set[str],
    assembly: str,
    group_field: str,
) -> dict[str, str]:
    if group_field not in {"GeneSymbol", "GeneID"}:
        raise ValueError(f"Unsupported group_field: {group_field}")

    key_to_group: dict[str, str] = {}
    with gzip.open(variant_summary_gz, "rt", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return key_to_group

        for row in reader:
            if assembly:
                if (row.get("Assembly") or "").strip() != assembly:
                    continue

            chrom = (row.get("Chromosome") or "").strip()
            pos = (row.get("PositionVCF") or "").strip()
            ref = (row.get("ReferenceAlleleVCF") or "").strip().upper()
            alt = (row.get("AlternateAlleleVCF") or "").strip().upper()

            if not chrom or not pos or not ref or not alt:
                continue
            if not (is_snv_allele(ref) and is_snv_allele(alt)):
                continue

            key = f"{chrom}_{pos}_{ref}_{alt}"
            if key not in needed_keys:
                continue

            group_value = (row.get(group_field) or "").strip()
            if not group_value:
                continue

            # If duplicates exist, keep first non-empty value.
            if key not in key_to_group:
                key_to_group[key] = group_value

            if len(key_to_group) >= len(needed_keys):
                break

    return key_to_group

# Assign groups to splits while balancing label distribution 
def assign_groups_to_splits(
    groups: pd.Series,
    labels: pd.Series,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    method: str,
) -> dict[str, set[str]]:
    fracs = np.array([train_frac, val_frac, test_frac], dtype=float)
    if np.any(fracs < 0):
        raise ValueError("Split fractions must be non-negative.")
    total = fracs.sum()
    if total <= 0:
        raise ValueError("At least one split fraction must be > 0.")
    fracs = fracs / total

    df_stats = pd.DataFrame({"group": groups.astype(str), "label": labels.astype(int)})
    by_group = df_stats.groupby("group", dropna=False)["label"].agg(["count", "sum"]).reset_index()
    by_group = by_group.rename(columns={"count": "rows", "sum": "positives"})
    by_group["rows"] = by_group["rows"].astype(int)
    by_group["positives"] = by_group["positives"].astype(int)
    by_group["negatives"] = (by_group["rows"] - by_group["positives"]).astype(int)

    group_items = by_group.to_dict(orient="records")
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda r: int(r["rows"]), reverse=True)

    split_names = ["train", "val", "test"]
    split_targets = {"train": fracs[0], "val": fracs[1], "test": fracs[2]}

    split_groups: dict[str, set[str]] = {k: set() for k in split_names}
    split_rows: dict[str, int] = {k: 0 for k in split_names}
    split_pos: dict[str, int] = {k: 0 for k in split_names}

    total_rows = int(groups.shape[0])
    split_target_rows = {k: split_targets[k] * total_rows for k in split_names}

    total_pos = int((labels.astype(int) == 1).sum())
    overall_pos_rate = (total_pos / total_rows) if total_rows else 0.0
    split_target_pos = {k: split_target_rows[k] * overall_pos_rate for k in split_names}

    if method == "size":
        for rec in group_items:
            group_key = str(rec["group"])
            group_rows = int(rec["rows"])
            group_pos = int(rec["positives"])

            deficits = {
                k: split_target_rows[k] - split_rows[k]
                for k in split_names
                if split_target_rows[k] > 0
            }
            chosen = max(deficits.items(), key=lambda kv: kv[1])[0]
            split_groups[chosen].add(group_key)
            split_rows[chosen] += group_rows
            split_pos[chosen] += group_pos

        return split_groups

    # method == "size+label": also try to match positive-rate targets.
    w_rows = 1.0
    w_pos = 0.5
    for rec in group_items:
        group_key = str(rec["group"])
        group_rows = int(rec["rows"])
        group_pos = int(rec["positives"])

        best_split: Optional[str] = None
        best_score: Optional[float] = None
        for k in split_names:
            target_rows = split_target_rows[k]
            target_pos = split_target_pos[k]
            if target_rows <= 0:
                continue

            new_rows = split_rows[k] + group_rows
            new_pos = split_pos[k] + group_pos

            row_err = abs(1.0 - (new_rows / target_rows))
            if target_pos > 0:
                pos_err = abs(1.0 - (new_pos / target_pos))
            else:
                pos_err = 0.0

            score = (w_rows * row_err) + (w_pos * pos_err)
            if best_score is None or score < best_score:
                best_score = score
                best_split = k

        assert best_split is not None
        split_groups[best_split].add(group_key)
        split_rows[best_split] += group_rows
        split_pos[best_split] += group_pos

    return split_groups

# Compute summary statistics for each split 
def compute_split_summary(df: pd.DataFrame, group_col: str) -> dict:
    summary: dict[str, dict] = {}
    for split_name, sub in df.groupby("split", dropna=False):
        split_name = str(split_name)
        n = int(sub.shape[0])
        pos = int((sub["label"] == 1).sum())
        neg = int((sub["label"] == 0).sum())
        pr = float(pos / n) if n else 0.0
        n_groups = int(sub[group_col].nunique(dropna=False))
        summary[split_name] = {
            "rows": n,
            "positives": pos,
            "negatives": neg,
            "positive_rate": pr,
            "groups": n_groups,
        }
    return summary

# Ensure no group leakage across splits 
def assert_no_group_leakage(df: pd.DataFrame, group_col: str) -> None:
    g = df[[group_col, "split"]].dropna(subset=[group_col])
    leakage = (
        g.groupby(group_col)["split"].nunique().sort_values(ascending=False)
    )
    if (leakage > 1).any():
        bad = leakage[leakage > 1]
        raise RuntimeError(
            f"Group leakage detected for {bad.shape[0]} groups (e.g., {bad.index[0]} appears in {bad.iloc[0]} splits)."
        )

# Dataclass to hold label row information for output 
@dataclass(frozen=True)
class SplitMeta:
    input_tsv: str
    clinvar_variant_summary: str
    assembly: str
    group_column: str
    train_frac: float
    val_frac: float
    test_frac: float
    seed: int
    missing_gene_policy: str
    missing_gene_rows: int
    created_at: str
    split_summary: dict

# Main function to run the script 
def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    input_tsv = repo_root / args.input_tsv
    clinvar_variant_summary = repo_root / args.clinvar_variant_summary
    out_prefix = repo_root / args.out_prefix
    missing_report_path = resolve_missing_report_path(repo_root, out_prefix, args.missing_report_path)

    df = load_week2_table(input_tsv)

    group_col = args.group_column.strip()
    missing_gene_rows = 0

    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"--group-column '{group_col}' not found in input TSV columns: {list(df.columns)}")
    else:
        group_col = args.clinvar_group_field
        needed_keys = set(df["chr_pos_ref_alt"].astype(str).tolist())
        key_to_group = map_key_to_group_value(
            clinvar_variant_summary,
            needed_keys=needed_keys,
            assembly=args.assembly,
            group_field=group_col,
        )
        group_value = df["chr_pos_ref_alt"].map(key_to_group)

        missing = group_value.isna()
        missing_gene_rows = int(missing.sum())
        if missing_gene_rows:
            if missing_report_path is not None:
                write_missing_mapping_report(
                    df=df,
                    missing_mask=missing,
                    out_path=missing_report_path,
                    group_field=group_col,
                )
                print(f"Wrote missing-mapping report: {missing_report_path}")

            if args.missing_gene_policy == "error":
                raise RuntimeError(
                    f"Missing {group_col} mapping for {missing_gene_rows} rows. "
                    "Re-run with --missing-gene-policy key|unknown or verify ClinVar inputs."
                )
            elif args.missing_gene_policy == "unknown":
                group_value = group_value.fillna("UNKNOWN")
            else:  # key
                group_value = group_value.fillna(df.loc[missing, "chr_pos_ref_alt"])

        df[group_col] = group_value.astype(str)

    split_groups = assign_groups_to_splits(
        df[group_col],
        df["label"],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        method=args.method,
    )

    def _group_to_split(g: str) -> str:
        if g in split_groups["train"]:
            return "train"
        if g in split_groups["val"]:
            return "val"
        return "test"

    df["split"] = df[group_col].astype(str).map(_group_to_split)

    # Sanity checks
    assert set(df["split"].unique().tolist()) <= {"train", "val", "test"}
    assert_no_group_leakage(df, group_col=group_col)
    split_summary = compute_split_summary(df, group_col=group_col)

    # Write artifacts
    parquet_path = Path(str(out_prefix) + "_splits.parquet")
    df.to_parquet(parquet_path, index=False)

    idx = np.arange(df.shape[0], dtype=np.int64)
    for split_name in ["train", "val", "test"]:
        out_idx = idx[df["split"].values == split_name]
        np.save(Path(str(out_prefix) + f"_{split_name}_idx.npy"), out_idx)

    meta = SplitMeta(
        input_tsv=str(args.input_tsv),
        clinvar_variant_summary=str(args.clinvar_variant_summary),
        assembly=str(args.assembly),
        group_column=str(group_col),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        missing_gene_policy=str(args.missing_gene_policy),
        missing_gene_rows=int(missing_gene_rows),
        created_at=datetime.now(timezone.utc).isoformat(),
        split_summary=split_summary,
    )
    meta_path = Path(str(out_prefix) + "_splits_meta.json")
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n")

    print(json.dumps(split_summary, indent=2, sort_keys=True))
    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {Path(str(out_prefix) + '_train_idx.npy')}")
    print(f"Wrote: {Path(str(out_prefix) + '_val_idx.npy')}")
    print(f"Wrote: {Path(str(out_prefix) + '_test_idx.npy')}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
