# Chromosome-based split design

## Goal
Prevent local-sequence leakage by assigning entire chromosomes to train/validation/test.

## Why chromosome holdout?
Variants near each other share sequence context and annotation characteristics. A random row-level split can leak highly similar examples across splits, inflating performance.

## Recommended approach
1. Curate a labeled table with at least these columns:
   - `Chromosome` (or `chrom`)
   - `label` (0/1)
2. Create a chromosome holdout plan:
   - **Default:** random assignment of chromosomes to train/val/test with a fixed seed.
   - **Preferred:** balance-aware search that tries many chromosome partitions and picks one with:
     - approximate target split sizes (e.g., 80/10/10)
     - similar positive-class rates in each split

## Trade-offs
- Chromosome holdout improves leakage control but reduces stratification control.
- If label distribution across chromosomes is highly uneven, you may need:
  - more iterations of the balance-aware search
  - a different split strategy (e.g., chromosome groups or multiple folds)

## Implementation
- Library: `src/variant_embeddings/splits/chromosome_split.py`
- Script: `scripts/make_splits.py`

Example (after you have a curated dataset):

```bash
python scripts/make_splits.py \
  --input data/processed/clinvar_non_coding_labeled.parquet \
  --output data/processed/clinvar_non_coding_labeled_with_split.parquet \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 42 \
  --balanced-search
```
