# Dataset orientation (Dr. Fan lab / Dylan Tan)

This note captures the key dataset logistics and conventions shared by Dr. Xiao Fan and Dylan Tan (Spring 2026) so the rest of the repo can stay reproducible and consistent.

## What Dylan shared

On HiPerGator, Dylan indicated the primary dataset directory contains:

- `Baseline/`
  - `esm2_selected_features.pkl` (main file used in this repo)
  - gene-specific MaveDB-derived datasets (e.g., `esm2_BRCA1_embed.pkl`)
- `CoordsData/`
  - `esm2_selected_coord_features.pkl`

**Difference between Baseline vs CoordsData:**
- `esm2_selected_coord_features.pkl` appends coordinate columns to the feature matrix.
- The “test pickles” for specific genes (BRCA1, etc.) contain embeddings and other pathogenic scores from other methods, and **do not** contain coordinates.

## File format: line-by-line pickle

The main PKL is a *stream* of pickled Python objects written sequentially (“line by line”). Loading requires a loop until `EOFError`.

This repo already implements that pattern in `scripts/build_week2_training_table.py` via `iter_pickle_objects()`.

## Canonical variant key

Dr. Fan confirmed all variants are missense SNVs, and the canonical key is:

- `chrom`, `pos`, `ref`, `alt`
- often aggregated as a single string: `chr_pos_ref_alt` (e.g., `17_43045705_T_C`)

This repo treats `chr_pos_ref_alt` as the join key between:
- embeddings (from Dylan PKL)
- strict labels / reference tables
- ClinVar `variant_summary.txt.gz` mapping (e.g., to gene identifiers)

## Reference code Dylan mentioned

Dylan’s best reference script for loading/cleaning was:
- `FinTest_MLP.py` (HiPerGator path provided in Teams)

Key takeaways:
- It includes helper logic to extract one-hot columns and other cleaning steps.
- It demonstrates the correct loop-based PKL loading pattern.

Dylan also later pointed to a notebook as a reference for how he joined variants and performed cleaning steps:
- `ESM2_Post_OMIM.ipynb` (HiPerGator path provided in Teams)

## MaveDB gene sheets (for chr/pos/ref/alt)

Dylan shared an Excel file with 4 sheets (BRCA1, MSH2, PTEN, TP53) that contains:
- `chrom`, `pos`, `ref`, `alt`
- an `ID` column already formatted as `chr_pos_ref_alt`

This is useful when you need to validate joins on `chr_pos_ref_alt` for specific gene subsets.

## Repo alignment

This repository’s “Week 2 → Week 4” data path is designed to match the dataset conventions above:

- Week 2: `scripts/build_week2_training_table.py`
  - reads Dylan’s `esm2_selected_features.pkl`
  - produces a row-aligned table + embedding matrix
- Week 3: `scripts/make_week3_splits.py`
  - uses `chr_pos_ref_alt` and maps to a gene grouping identifier via ClinVar `variant_summary.txt.gz`
  - produces gene-disjoint train/val/test splits
- Week 4: `scripts/make_week4_curated_dataset.py`
  - merges Week 2 table + Week 3 splits and writes a single Parquet used by baselines

## Documentation note (formal writing)

Per Dr. Fan’s guidance, **avoid including cluster paths** in formal reports.
In formal writing, describe *what the data are* (ClinVar-derived missense SNVs with strict labels and aligned ESM2 embeddings), not where they live.
