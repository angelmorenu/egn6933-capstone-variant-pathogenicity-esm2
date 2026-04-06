#!/usr/bin/env python3
"""Clean and normalize spreadsheet-style variant inputs.

This script normalizes variant IDs to `chromosome_position_ref_alt`, supports
common spreadsheet column names, and can optionally annotate rows using the
curated label override file used by `app/app.py`.

Examples:
    python scripts/clean_variant_spreadsheet.py \
        --input ~/Downloads/2026-04-05T17-33_export.csv \
        --output /tmp/cleaned_variants.csv

    python scripts/clean_variant_spreadsheet.py \
        --input ~/Downloads/Angel_Morenu.csv \
        --output /tmp/normalized_genotypes.csv \
        --lookup data/processed/curated_variant_label_overrides.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


VALID_BASES = {"A", "C", "G", "T"}
VALID_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
DEFAULT_LOOKUP = Path(__file__).resolve().parent.parent / "data" / "processed" / "curated_variant_label_overrides.tsv"


def normalize_variant_id(variant_id: str) -> str:
    """Normalize `chr17_...`, `17_...`, and `17:...` into `17_...`."""
    value = str(variant_id).strip()
    if not value:
        raise ValueError("Variant ID is empty.")

    parts = [token for token in re.split(r"[:_]", value) if token]
    if len(parts) != 4:
        raise ValueError(
            "Invalid variant format. Use `chr_pos_ref_alt` or `chr:pos:ref:alt` (chr optional)."
        )

    chromosome, position, ref, alt = parts
    chromosome = chromosome.strip().lower().removeprefix("chr").upper()
    if chromosome == "M":
        chromosome = "MT"

    if chromosome not in VALID_CHROMOSOMES:
        raise ValueError(f"Invalid chromosome `{chromosome}`.")

    position = position.strip()
    if not position.isdigit() or int(position) <= 0:
        raise ValueError("Position must be a positive integer.")

    ref = ref.strip().upper()
    alt = alt.strip().upper()
    if ref not in VALID_BASES or alt not in VALID_BASES:
        raise ValueError("Reference and alternate alleles must be one of A/C/G/T.")

    return f"{chromosome}_{int(position)}_{ref}_{alt}"


def load_label_lookup(path: Optional[Path]) -> Dict[str, str]:
    """Load variant_id -> label from a TSV lookup file."""
    if not path or not path.exists():
        return {}

    lookup: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            variant_id = (row.get("variant_id") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if variant_id and label in {"PATHOGENIC", "BENIGN"}:
                lookup[variant_id] = label
    return lookup


def extract_normalized_variants(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """Extract normalized variants from common spreadsheet schemas."""
    columns_lower = {col.lower(): col for col in df.columns}
    candidate_id_columns = [
        "variant_id",
        "variant",
        "canonical_id",
        "canonical_variant_id",
        "id",
    ]

    for candidate in candidate_id_columns:
        if candidate in columns_lower:
            source_col = columns_lower[candidate]
            normalized: list[str] = []
            invalid_rows: list[int] = []
            for row_index, raw in enumerate(df[source_col].astype(str).tolist()):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    normalized.append(normalize_variant_id(raw))
                except ValueError:
                    invalid_rows.append(row_index)
            return normalized, invalid_rows

    required = ["chromosome", "position", "ref", "alt"]
    if all(col in columns_lower for col in required):
        chrom_col = columns_lower["chromosome"]
        pos_col = columns_lower["position"]
        ref_col = columns_lower["ref"]
        alt_col = columns_lower["alt"]
        normalized: list[str] = []
        invalid_rows: list[int] = []
        for row_index, row in df.iterrows():
            try:
                normalized.append(
                    normalize_variant_id(
                        f"{row[chrom_col]}_{row[pos_col]}_{row[ref_col]}_{row[alt_col]}"
                    )
                )
            except ValueError:
                invalid_rows.append(int(row_index))
        return normalized, invalid_rows

    raise ValueError(
        "Unsupported input schema. Use a variant column or `chromosome`, `position`, `ref`, `alt`."
    )


def build_output_table(df: pd.DataFrame, lookup: Dict[str, str]) -> pd.DataFrame:
    """Return a cleaned table with normalized IDs and optional labels."""
    normalized_variants, invalid_rows = extract_normalized_variants(df)

    output = df.copy().reset_index(drop=True)
    output = output.loc[[i for i in range(len(output)) if i not in set(invalid_rows)]].reset_index(drop=True)
    output = output.copy()

    variant_columns = {"variant_id", "variant", "canonical_id", "canonical_variant_id", "id"}
    if any(col.lower() in variant_columns for col in output.columns):
        source_col = next(col for col in output.columns if col.lower() in variant_columns)
        cleaned_rows = []
        for raw in output[source_col].astype(str).tolist():
            cleaned_rows.append(normalize_variant_id(raw))
        output["normalized_variant_id"] = cleaned_rows
    else:
        output["normalized_variant_id"] = normalized_variants

    if lookup:
        output["label"] = output["normalized_variant_id"].map(lookup)

    return output


def wide_genotype_to_long(df: pd.DataFrame, lookup: Dict[str, str]) -> pd.DataFrame:
    """Convert a wide genotype matrix into a long table of observed variants."""
    columns_lower = {col.lower(): col for col in df.columns}
    sample_col = columns_lower.get("sampleid") or columns_lower.get("sample_id")

    variant_columns = [col for col in df.columns if col != sample_col]
    records = []

    for _, row in df.iterrows():
        sample_id = row[sample_col] if sample_col else None
        for column in variant_columns:
            value = row[column]
            if pd.isna(value):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = None

            if numeric_value is not None and numeric_value <= 0:
                continue
            if str(value).strip() in {"", "0", "0.0", "NA", "NaN", "nan"}:
                continue

            try:
                normalized_variant = normalize_variant_id(column)
            except ValueError:
                continue

            records.append(
                {
                    "sampleid": sample_id,
                    "normalized_variant_id": normalized_variant,
                    "observed_value": value,
                    "label": lookup.get(normalized_variant),
                }
            )

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV/TSV spreadsheet.")
    parser.add_argument("--output", required=True, type=Path, help="Output cleaned CSV path.")
    parser.add_argument(
        "--lookup",
        type=Path,
        default=DEFAULT_LOOKUP,
        help="Optional TSV lookup file with columns variant_id,label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)
    lookup = load_label_lookup(args.lookup)
    try:
        cleaned = build_output_table(df, lookup)
    except ValueError:
        cleaned = wide_genotype_to_long(df, lookup)
        if cleaned.empty:
            raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"Wrote {len(cleaned)} normalized rows to {args.output}")
    if lookup:
        print(f"Loaded {len(lookup)} curated labels from {args.lookup}")


if __name__ == "__main__":
    main()
