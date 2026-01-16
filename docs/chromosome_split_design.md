# Gene/protein-aware split design (recommended)

## Goal
Prevent train/test leakage by ensuring variants from the same biological unit (gene/protein/transcript) do not appear in multiple splits.

## Why gene/protein holdout?
In coding-variant prediction with protein-language-model features (e.g., ESM2 embeddings), leakage can occur when the same gene/protein (or highly related sequence context) is present in both training and test splits. A naive random row-level split can overestimate generalization.

## Recommended approach
1. Curate a labeled table with at least these columns:
   - `label` (0/1)
   - one grouping column such as `gene`, `gene_symbol`, `transcript_id`, or `protein_id`
2. Create a group-aware split plan:
   - Pick a single grouping column for v1 (prefer `protein_id` if available; otherwise `gene`)
   - Assign whole groups to train/val/test with a fixed seed
   - Verify class balance per split (may require multiple seeds or stratified heuristics)

## Trade-offs
- Strong leakage control, but can reduce split stratification control.
- If some groups dominate the label distribution, you may need:
  - repeated seeds / heuristic search for balance
  - multiple folds for more stable estimates

## Minimal implementation (sklearn)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_parquet("data/processed/dataset_strict.parquet")
groups = df["protein_id"]  # or "gene"

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=groups))

temp = df.iloc[temp_idx]
temp_groups = temp["protein_id"]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_rel_idx, test_rel_idx = next(gss2.split(temp, groups=temp_groups))

df.loc[df.index[train_idx], "split"] = "train"
df.loc[temp.index[val_rel_idx], "split"] = "val"
df.loc[temp.index[test_rel_idx], "split"] = "test"

df.to_parquet("data/processed/dataset_strict_with_split.parquet", index=False)
```

## Note on chromosome-based holdout
Chromosome holdout can still be useful for certain genomics pipelines, and the repo may contain legacy helpers (e.g., `src/variant_embeddings/splits/chromosome_split.py`). For the current coding-variant ESM2 scope, prefer gene/protein-aware splits.
