#!/usr/bin/env python3

import argparse
import pickle
from collections import Counter
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def iter_pickle_objects(path: str, max_records: Optional[int] = None) -> Iterable[Any]:
    count = 0
    with open(path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            yield obj
            count += 1
            if max_records is not None and count >= max_records:
                break


def _shape_of_embedding(value: Any) -> Optional[Tuple[int, ...]]:
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
    return tuple(arr.shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ESM2 PrimateAI-style pickle files saved record-by-record. "
            "Prints basic schema and embedding shape stats."
        )
    )
    parser.add_argument(
        "pickle_file",
        help="Path to a .pkl file (e.g., data/Dylan Tan/esm2_selected_features.pkl)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help="Max records to read (default: 5000). Use 0 for all records.",
    )
    parser.add_argument(
        "--embedding-col",
        default="Embedding",
        help="Name of embedding column (default: Embedding)",
    )
    args = parser.parse_args()

    max_records = None if args.max_records == 0 else args.max_records
    records: List[Any] = list(iter_pickle_objects(args.pickle_file, max_records=max_records))

    if not records:
        raise SystemExit("No records found (empty file or unreadable pickle format).")

    print(f"Loaded records: {len(records)}")
    print(f"First record type: {type(records[0]).__name__}")

    if isinstance(records[0], dict):
        key_counts = Counter()
        for r in records:
            if isinstance(r, dict):
                key_counts.update(r.keys())
        print("\nTop keys (count across records):")
        for k, c in key_counts.most_common(25):
            print(f"  {k}: {c}")

    try:
        df = pd.DataFrame(records)
    except Exception as e:
        raise SystemExit(f"Could not coerce records into a DataFrame: {e}")

    print(f"\nDataFrame shape: {df.shape}")
    print("Columns:")
    for col in df.columns:
        print(f"  - {col} ({df[col].dtype})")

    emb_col = args.embedding_col
    if emb_col in df.columns:
        shapes = df[emb_col].map(_shape_of_embedding).dropna().tolist()
        print(f"\nNon-empty `{emb_col}` count: {len(shapes)}")
        if shapes:
            shape_counts = Counter(shapes)
            print("Most common embedding shapes:")
            for s, c in shape_counts.most_common(10):
                print(f"  {s}: {c}")

            first_idx = df[emb_col].map(_shape_of_embedding).first_valid_index()
            if first_idx is not None:
                example = np.asarray(df.loc[first_idx, emb_col])
                print(f"Example embedding dtype: {example.dtype}")
                print(f"Example embedding min/max: {np.nanmin(example):.6g} / {np.nanmax(example):.6g}")
    else:
        print(f"\nNote: embedding column `{emb_col}` not found.")


if __name__ == "__main__":
    main()
