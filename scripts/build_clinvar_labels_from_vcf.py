#!/usr/bin/env python3
"""Build per-VariationID labels from a ClinVar VCF (combined SCV submissions).

Purpose
- Reproduce Dr. Fan's label logic for conflicting/VUS using counts from ClinVar's VCF.

Expected inputs
- A ClinVar VCF (optionally .gz). For many ClinVar VCFs, INFO includes:
  - CLNSIG: clinical significance terms
  - CLNSIGCONF: for conflicting variants, a per-significance count summary
  - CLNVC: variant type
  - (sometimes) CLNREVSTAT, CLNSIGSCV, etc.

Output
- TSV (optionally gz) keyed by VariationID with:
  - label (0/1/blank), label_reason
  - counts: n_plp, n_vus (when available)

Notes
- VCF field names vary slightly by ClinVar release. This script is defensive:
  it will still emit basic labels for clean B/LB and P/LP sites even when
  conflict counts are unavailable.
"""

from __future__ import annotations

import argparse
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PLP_TERMS = {
    "pathogenic",
    "likely_pathogenic",
}
BLB_TERMS = {
    "benign",
    "likely_benign",
}
VUS_TERMS = {
    "uncertain_significance",
    "vus",
    "uncertain",
}


def _normalize_term(term: str) -> str:
    return term.strip().lower().replace(" ", "_").replace("-", "_")


def _open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt")


def iter_vcf_records(path: Path) -> Iterable[tuple[str, dict[str, str]]]:
    """Yield (chrom, info_dict) for each non-header VCF record."""
    with _open_text_auto(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom = parts[0]
            info = parts[7]
            info_dict: dict[str, str] = {}
            for item in info.split(";"):
                if not item:
                    continue
                if "=" in item:
                    k, v = item.split("=", 1)
                    info_dict[k] = v
                else:
                    info_dict[item] = ""
            yield chrom, info_dict

# Function to parse INFO fields from VCF and decide labels
def parse_info_terms(info_dict: dict[str, str], key: str) -> list[str]:
    v = info_dict.get(key, "")
    if not v:
        return []
    return [_normalize_term(x) for x in re.split(r"[|,]", v) if x]

# Function to extract VariationID from INFO fields
def parse_variation_id(info_dict: dict[str, str]) -> Optional[int]:
    """Try common ClinVar VCF keys for VariationID."""
    for k in ("CLNVI", "VARID", "VariationID", "CLNVARID"):
        v = info_dict.get(k)
        if not v:
            continue
        # Some formats: 'ClinVar:12345' or '12345'
        m = re.search(r"(\d+)", v)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            continue
    return None

# Function to parse CLNSIGCONF counts
def parse_clnsigconf_counts(info_dict: dict[str, str]) -> tuple[Optional[int], Optional[int]]:
    """Parse CLNSIGCONF to get (n_plp, n_vus) when available.

    CLNSIGCONF formatting varies. Common patterns include pairs like:
      Pathogenic(3)|Likely_pathogenic(2)|Uncertain_significance(4)
    This function extracts integer counts next to recognized terms.
    """
    raw = info_dict.get("CLNSIGCONF", "")
    if not raw:
        return None, None

    n_plp = 0
    n_vus = 0
    found_any = False

    # Split on | or , and parse TERM(NUM)
    for chunk in re.split(r"[|,]", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"([^()]+)\((\d+)\)$", chunk)
        if not m:
            continue
        term = _normalize_term(m.group(1))
        cnt = int(m.group(2))
        found_any = True
        if term in PLP_TERMS:
            n_plp += cnt
        elif term in VUS_TERMS or term == "uncertain_significance":
            n_vus += cnt

    if not found_any:
        return None, None

    return n_plp, n_vus

# Dataclass to hold label row information for output
@dataclass(frozen=True)
class LabelRow:
    variation_id: int
    label: str
    label_reason: str
    n_plp: str
    n_vus: str

# Function to decide label based on INFO fields 
def decide_label(info_dict: dict[str, str]) -> Optional[LabelRow]:
    variation_id = parse_variation_id(info_dict)
    if variation_id is None:
        return None

    clnsig_terms = set(parse_info_terms(info_dict, "CLNSIG"))

    # Clean cases: only B/LB or only P/LP
    if clnsig_terms and clnsig_terms.issubset(BLB_TERMS):
        return LabelRow(variation_id, "0", "CLNSIG_only_B/LB", "", "")
    if clnsig_terms and clnsig_terms.issubset(PLP_TERMS):
        return LabelRow(variation_id, "1", "CLNSIG_only_P/LP", "", "")

    # Conflicts / VUS-like: attempt rescue using CLNSIGCONF counts
    n_plp, n_vus = parse_clnsigconf_counts(info_dict)
    if n_plp is not None and n_vus is not None:
        if n_vus > 0 and n_plp >= 2 * n_vus:
            return LabelRow(variation_id, "1", "rescued_conflict_plp_ge_2x_vus", str(n_plp), str(n_vus))
        return LabelRow(variation_id, "", "excluded_conflict_or_vus", str(n_plp), str(n_vus))

    # If we can't count, exclude ambiguous (keeps behavior conservative)
    if clnsig_terms and (clnsig_terms & VUS_TERMS):
        return LabelRow(variation_id, "", "excluded_vus_no_counts", "", "")

    return LabelRow(variation_id, "", "excluded_ambiguous_no_counts", "", "")

# Command-line argument parsing function
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labels from a ClinVar VCF (combined SCV submissions).")
    p.add_argument("--vcf", required=True, help="Path to ClinVar VCF (.vcf or .vcf.gz)")
    p.add_argument(
        "--out",
        default="data/processed/clinvar_labels_from_vcf.tsv.gz",
        help="Output TSV (optionally .gz) keyed by VariationID",
    )
    p.add_argument("--max-rows", type=int, default=0, help="If >0, stop after this many VCF records")
    return p.parse_args()

# Main function to run the script
def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    vcf_path = Path(args.vcf)
    if not vcf_path.is_absolute():
        vcf_path = repo_root / vcf_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if str(out_path).endswith(".gz"):
        out_f = gzip.open(out_path, "wt")
    else:
        out_f = out_path.open("wt")

    written = 0
    seen: set[int] = set()

    with out_f:
        out_f.write("VariationID\tlabel\tlabel_reason\tn_plp\tn_vus\n")
        for i, (_chrom, info_dict) in enumerate(iter_vcf_records(vcf_path), start=1):
            row = decide_label(info_dict)
            if row is None:
                continue
            if row.variation_id in seen:
                continue
            seen.add(row.variation_id)
            out_f.write(
                f"{row.variation_id}\t{row.label}\t{row.label_reason}\t{row.n_plp}\t{row.n_vus}\n"
            )
            written += 1
            if args.max_rows and i >= args.max_rows:
                break

    print(f"Wrote: {out_path} ({written} unique VariationIDs)")


if __name__ == "__main__":
    main()
