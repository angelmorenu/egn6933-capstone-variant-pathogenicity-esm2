#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

"""
Sanity-check Week 2 training table artifacts (TSV.gz + embeddings .npy + meta .json).
Expects files:
- <prefix>.tsv.gz
- <prefix>_embeddings.npy
- <prefix>_meta.json

Prints summary stats and checks for common problems:
- Missing files
- Row count mismatches
- Embedding dimension mismatches
- NaNs in embeddings (sampled) 
"""

# Sanity-check Week 2 training table artifacts:
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check Week 2 training table artifacts (TSV.gz + embeddings .npy + meta .json).")
    p.add_argument(
        "--prefix",
        default="data/processed/week2_training_table_strict",
        help="Artifact prefix (expects <prefix>.tsv.gz, <prefix>_embeddings.npy, <prefix>_meta.json)",
    )
    return p.parse_args()

# Main function to run the script 
def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    prefix = Path(args.prefix)
    if not prefix.is_absolute():
        prefix = repo_root / prefix

    tsv_gz = prefix.with_suffix(".tsv.gz")
    emb_npy = prefix.parent / f"{prefix.name}_embeddings.npy"
    meta_json = prefix.parent / f"{prefix.name}_meta.json"

    if not tsv_gz.exists():
        raise FileNotFoundError(tsv_gz)
    if not emb_npy.exists():
        raise FileNotFoundError(emb_npy)
    if not meta_json.exists():
        raise FileNotFoundError(meta_json)

    meta = json.loads(meta_json.read_text())

    rows = 0
    label_counts = Counter()
    dup_count = 0
    seen_keys = set()

    with gzip.open(tsv_gz, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        required = ["chr_pos_ref_alt", "label"]
        missing = [c for c in required if c not in idx]
        if missing:
            raise ValueError(f"Missing columns in TSV: {missing}")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            key = parts[idx["chr_pos_ref_alt"]]
            label = parts[idx["label"]]
            rows += 1
            label_counts[label] += 1
            if key in seen_keys:
                dup_count += 1
            else:
                seen_keys.add(key)

    emb = np.load(emb_npy, mmap_mode="r")

    problems = []
    if emb.ndim != 2:
        problems.append(f"embeddings ndim={emb.ndim} (expected 2)")
    else:
        if emb.shape[0] != rows:
            problems.append(f"row count mismatch: TSV rows={rows} vs embeddings rows={emb.shape[0]}")
        expected_dim = meta.get("stats", {}).get("embedding_dim")
        if expected_dim is not None and emb.shape[1] != int(expected_dim):
            problems.append(f"embedding dim mismatch: meta={expected_dim} vs embeddings={emb.shape[1]}")

    # Sample embeddings for NaNs (to avoid loading all into memory)
    try:
        has_nan = bool(np.isnan(emb[: min(rows, 5000)]).any())
    except Exception:
        has_nan = None

    print("=== Week2 Sanity Check ===")
    print(f"prefix: {prefix}")
    print(f"label_source: {meta.get('label_source')}")
    print(f"rows: {rows}")
    print(f"duplicates(chr_pos_ref_alt): {dup_count}")
    print(f"label_counts: {dict(label_counts)}")
    print(f"embeddings_shape: {tuple(getattr(emb, 'shape', ())) }")
    print(f"embeddings_has_nan(sample): {has_nan}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"- {p}")
        raise SystemExit(2)

    print("\nOK")


if __name__ == "__main__":
    main()
