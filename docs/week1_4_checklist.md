# Weeks 1–4 Execution Checklist

This is a concrete Week 1–4 plan aligned to the updated proposal: curated coding-variant dataset (labels + ESM2 embeddings) → ingestion/QC → leakage-aware splits → baseline training readiness.

## Week 1 — Dataset access + ingestion skeleton
- [ ] Obtain dataset location from Dylan (HiPerGator path or share link)
- [ ] Confirm raw dataset format(s) and contents
  - Expected: coding variants + pathogenicity labels + ESM2 embeddings (+ optional gene/protein identifiers)
- [ ] Create/validate ingestion output artifact(s)
  - Target: `data/processed/<dataset_name>_strict.parquet`
  - Columns to confirm: variant ID fields, label, embedding vector, and any gene/protein fields
- [ ] Define label mapping rules (high-confidence only)
  - Keep: Pathogenic/Likely Pathogenic vs Benign/Likely Benign
  - Exclude: VUS, conflicting interpretations
- [ ] Run basic QC
  - No NaNs in embeddings, consistent embedding dimensionality, no duplicate variant IDs

## Week 2 — Metadata standardization + (optional) validation
- [ ] Standardize variant identifiers/metadata
  - Minimal fields: `chrom`, `pos`, `ref`, `alt`, assembly (if present)
  - Preferred: transcript/gene/protein identifiers if provided
- [ ] Decide whether to use ClinVar/VEP as a secondary validation pathway
  - Goal: confirm coding consequences / reconcile labels (optional)
- [ ] Create a small pilot split (e.g., 1k–5k variants) and validate end-to-end training table creation

## Week 3 — Leakage-aware split design
- [ ] Implement gene/protein-aware train/val/test split plan
  - Constraint: all variants from the same gene/protein stay in one split
- [ ] Sanity-check split sizes and positive rates (train/val/test)
- [ ] Write split artifacts to disk
  - Store a `split` column in the processed Parquet and/or save split index files

## Week 4 — EDA deliverables + baseline readiness
- [ ] Finalize the curated dataset artifact
  - Parquet with `label` (0/1), `split` (train/val/test), and embedding vectors
- [ ] Produce core EDA plots/tables
  - Class balance overall + by split
  - Distribution by gene/protein (if available)
  - Embedding dimensionality checks and summary statistics
  - Variant-type breakdown (e.g., missense, nonsense, frameshift) if available
- [ ] Write down “go/no-go” checks before model training
  - Minimum positive class size
  - No leakage across gene/protein splits
  - No duplicate variants across splits
  - Embeddings present and consistent for all retained samples
