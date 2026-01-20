#!/usr/bin/env python3
# Build Week 2 training table with strict labels and canonical variant keys
import argparse
import csv
import gzip
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

"""
Build Week 2 training table with strict labels and canonical variant keys.

This script processes a line-by-line pickle file containing variant data with embeddings,
maps VariationIDs to canonical chr/pos/ref/alt keys using ClinVar data, filters to strict
binary pathogenicity labels (Benign/Likely_Benign -> 0, Pathogenic/Likely_Pathogenic -> 1),
and outputs a TSV table with embeddings saved as a NumPy array. It supports limiting rows
for pilot builds and handles deduplication on canonical keys.
"""

# Iterate over objects in a line-by-line pickle file
# (Yield each pickled object one by one)
def iter_pickle_objects(path: Path) -> Iterable[Any]:
    with path.open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

# Normalize label keys for consistent comparison
def _normalize_label_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")

# Convert pathogenicity labels to strict binary labels
def strict_binary_label(pathogenicity: Any) -> Optional[int]:
    if pathogenicity is None:
        return None

    if isinstance(pathogenicity, str):
        v = _normalize_label_key(pathogenicity)
        if v in {"benign", "likely_benign"}:
            return 0
        elif v in {"pathogenic", "likely_pathogenic"}:
            return 1
        else:
            return None

    return None

# Check if a value is missing (empty, NA, NaN, None)
def is_missing(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"", "na", "n/a", "nan", "none"}

# Check if allele is a single-nucleotide variant (A, C, G, T)
def is_snv_allele(allele: str) -> bool:
    a = (allele or "").strip().upper()
    return len(a) == 1 and a in {"A", "C", "G", "T"}

# Coerce embedding-like value to a 1D NumPy float32 array, or None if invalid
def coerce_embedding_1d(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.size == 0:
        return None

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).all():
        return None

    return arr.astype(np.float32, copy=False)

# Data class to hold build statistics
@dataclass(frozen=True)
class Week2BuildStats:
    pickle_rows_seen: int
    candidates_seen: int
    clinvar_rows_scanned: int
    clinvar_rows_kept: int
    mapped_ids: int
    rows_written: int
    dropped_unlabeled: int
    dropped_bad_embedding: int
    dropped_missing_id: int
    dropped_missing_mapping: int
    dropped_duplicate_key: int
    embedding_dim: int
    assembly: str
    created_at: str

# Parse command-line arguments for the script
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Week 2: Build a trainable table with canonical variant keys (chr/pos/ref/alt) and strict labels. "
            "Reads Dylan's line-by-line pickle, maps numeric ID (ClinVar VariationID) -> chr_pos_ref_alt via ClinVar, "
            "filters to strict labels, and writes TSV + NumPy embeddings."
        )
    )
    p.add_argument(
        "--pickle",
        default="data/Dylan Tan/esm2_selected_features.pkl",
        help="Path to Dylan .pkl (line-by-line pickle)",
    )
    p.add_argument(
        "--clinvar",
        default="data/clinvar/variant_summary.txt.gz",
        help="Path to ClinVar variant_summary.txt.gz",
    )
    p.add_argument(
        "--assembly",
        default="GRCh38",
        help="ClinVar Assembly to keep (GRCh38 or GRCh37). Use '' to disable filtering.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Max number of rows to write (use a small number for the Week 2 pilot; set very large for full build)",
    )
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=2560,
        help="Expected embedding dimension (default: 2560 for ESM2)",
    )
    p.add_argument(
        "--out-prefix",
        default="data/processed/week2_training_table_strict",
        help="Output prefix (writes <prefix>.tsv.gz, <prefix>_embeddings.npy, <prefix>_meta.json)",
    )
    return p.parse_args()

# Build mapping from VariationID to (chrom, pos, ref, alt, chr_pos_ref_alt)
def build_variationid_mapping(
    ids: set[int],
    clinvar_path: Path,
    assembly: str,
) -> tuple[dict[int, tuple[str, str, str, str, str]], int, int]:
    """Return mapping VariationID -> (chrom, pos, ref, alt, chr_pos_ref_alt)."""
    candidates: dict[int, dict[str, tuple[str, str, str, str]]] = {}
    scanned = 0
    kept = 0

    with gzip.open(clinvar_path, "rt", newline="") as gz:
        reader = csv.DictReader(gz, delimiter="\t")
        for row in reader:
            scanned += 1
            try:
                variation_id = int(row["VariationID"])
            except Exception:
                continue
            if variation_id not in ids:
                continue

            row_assembly = (row.get("Assembly") or "").strip()
            if assembly and row_assembly and row_assembly != assembly:
                continue

            chrom = (row.get("Chromosome") or "").strip()
            pos = (row.get("PositionVCF") or "").strip()
            ref = (row.get("ReferenceAlleleVCF") or "").strip()
            alt = (row.get("AlternateAlleleVCF") or "").strip()

            if is_missing(chrom) or is_missing(pos) or is_missing(ref) or is_missing(alt):
                continue
            if not pos.isdigit():
                continue
            if not (is_snv_allele(ref) and is_snv_allele(alt)):
                continue

            key = f"{chrom}_{pos}_{ref.upper()}_{alt.upper()}"
            candidates.setdefault(variation_id, {})[key] = (chrom, pos, ref.upper(), alt.upper())
            kept += 1

    mapping: dict[int, tuple[str, str, str, str, str]] = {}
    for variation_id, key_map in candidates.items():
        if len(key_map) == 1:
            only_key, (chrom, pos, ref, alt) = next(iter(key_map.items()))
            mapping[variation_id] = (chrom, pos, ref, alt, only_key)

    return mapping, scanned, kept

# Main function to execute the build process with argument parsing and file handling
def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(p: str | Path) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo_root / pp)

    pkl_path = _resolve(args.pickle)
    clinvar_path = _resolve(args.clinvar)

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Pickle file not found: {pkl_path} (repo_root={repo_root}). "
            "Either run from the repo root or pass --pickle with an absolute/relative path."
        )
    if not clinvar_path.exists():
        raise FileNotFoundError(
            f"ClinVar file not found: {clinvar_path} (repo_root={repo_root}). "
            "Either run from the repo root or pass --clinvar with an absolute/relative path."
        )

    out_prefix = _resolve(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    tsv_gz_path = out_prefix.with_suffix(".tsv.gz")
    emb_npy_path = out_prefix.parent / f"{out_prefix.name}_embeddings.npy"
    meta_json_path = out_prefix.parent / f"{out_prefix.name}_meta.json"

    # First pass: identify candidate IDs from the pickle file that pass strict label + embedding checks
    candidate_ids: list[int] = []
    candidate_set: set[int] = set()

    pickle_rows_seen = 0
    candidates_seen = 0
    dropped_unlabeled = 0
    dropped_bad_embedding = 0
    dropped_missing_id = 0

    for obj in iter_pickle_objects(pkl_path):
        pickle_rows_seen += 1
        if not isinstance(obj, dict):
            continue

        raw_id = obj.get("ID")
        if raw_id is None:
            dropped_missing_id += 1
            continue

        try:
            variation_id = int(raw_id)
        except Exception:
            dropped_missing_id += 1
            continue

        label = strict_binary_label(obj.get("Pathogenicity"))
        if label is None:
            dropped_unlabeled += 1
            continue

        emb = coerce_embedding_1d(obj.get("Embedding"))
        if emb is None or emb.shape != (args.embedding_dim,):
            dropped_bad_embedding += 1
            continue

        candidates_seen += 1
        if variation_id not in candidate_set:
            candidate_set.add(variation_id)
            candidate_ids.append(variation_id)

    # Build mapping only for candidate IDs.
    mapping, clinvar_rows_scanned, clinvar_rows_kept = build_variationid_mapping(
        ids=candidate_set,
        clinvar_path=clinvar_path,
        assembly=args.assembly,
    )

    # Pass 2: write rows (and embeddings) in a stable order
    # Iterate pickle again and write in first-seen order, deduping on chr_pos_ref_alt
    # Preallocate NumPy memmap for embeddings
    rows_written = 0
    dropped_missing_mapping = 0
    dropped_duplicate_key = 0

    seen_key: set[str] = set()

    # Prepare list of kept IDs in order for writing embeddings array and TSV
    kept_ids: list[int] = []
    for vid in candidate_ids:
        if vid in mapping:
            kept_ids.append(vid)

    # Enforce max rows if specified
    if args.max_rows is not None and args.max_rows > 0:
        kept_ids = kept_ids[: args.max_rows]

    # Preallocate embeddings array
    emb_mm = np.lib.format.open_memmap(
        emb_npy_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(kept_ids), args.embedding_dim),
    )

    id_to_row_index = {vid: i for i, vid in enumerate(kept_ids)}

    with gzip.open(tsv_gz_path, "wt", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "pickle_ID",
                "Chromosome",
                "PositionVCF",
                "ReferenceAlleleVCF",
                "AlternateAlleleVCF",
                "chr_pos_ref_alt",
                "Pathogenicity",
                "label",
            ]
        )

        for obj in iter_pickle_objects(pkl_path):
            if rows_written >= len(kept_ids):
                break
            if not isinstance(obj, dict):
                continue

            raw_id = obj.get("ID")
            if raw_id is None:
                continue
            try:
                variation_id = int(raw_id)
            except Exception:
                continue

            row_idx = id_to_row_index.get(variation_id)
            if row_idx is None:
                continue

            mapped = mapping.get(variation_id)
            if mapped is None:
                dropped_missing_mapping += 1
                continue

            chrom, pos, ref, alt, key = mapped
            if key in seen_key:
                dropped_duplicate_key += 1
                continue

            label = strict_binary_label(obj.get("Pathogenicity"))
            if label is None:
                dropped_unlabeled += 1
                continue

            emb = coerce_embedding_1d(obj.get("Embedding"))
            if emb is None or emb.shape != (args.embedding_dim,):
                dropped_bad_embedding += 1
                continue

            emb_mm[row_idx, :] = emb
            seen_key.add(key)

            writer.writerow(
                [
                    variation_id,
                    chrom,
                    pos,
                    ref,
                    alt,
                    key,
                    obj.get("Pathogenicity"),
                    label,
                ]
            )
            rows_written += 1

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    stats = Week2BuildStats(
        pickle_rows_seen=pickle_rows_seen,
        candidates_seen=candidates_seen,
        clinvar_rows_scanned=clinvar_rows_scanned,
        clinvar_rows_kept=clinvar_rows_kept,
        mapped_ids=len(mapping),
        rows_written=rows_written,
        dropped_unlabeled=dropped_unlabeled,
        dropped_bad_embedding=dropped_bad_embedding,
        dropped_missing_id=dropped_missing_id,
        dropped_missing_mapping=dropped_missing_mapping,
        dropped_duplicate_key=dropped_duplicate_key,
        embedding_dim=args.embedding_dim,
        assembly=args.assembly,
        created_at=created_at,
    )

    meta = {
        "script": "scripts/build_week2_training_table.py",
        "pickle": str(pkl_path),
        "clinvar": str(clinvar_path),
        "outputs": {
            "table_tsv_gz": str(tsv_gz_path),
            "embeddings_npy": str(emb_npy_path),
            "meta_json": str(meta_json_path),
        },
        "stats": asdict(stats),
        "notes": [
            "Strict label policy: {Pathogenic, Likely_Pathogenic}->1 and {Benign, Likely_Benign}->0; other labels excluded.",
            "Canonical key derived from ClinVar VCF fields (Chromosome, PositionVCF, ReferenceAlleleVCF, AlternateAlleleVCF).",
            "Embeddings saved as float32 NumPy array aligned to row order in the TSV.",
        ],
    }

    meta_json_path.write_text(json.dumps(meta, indent=2) + "\n")

    print("Wrote:")
    print(f"  {tsv_gz_path}")
    print(f"  {emb_npy_path}")
    print(f"  {meta_json_path}")
    print("\nSummary:")
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
