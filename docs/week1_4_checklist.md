# Weeks 1–4 Execution Checklist

This is a concrete Week 1–4 plan aligned to the updated proposal: ClinVar (labels) + VEP missense filtering (+ embeddings) → ingestion/QC → leakage-aware splits → baseline training readiness.

## Week 1 — Dataset access + ingestion skeleton
- ✅ Download ClinVar release data (VCF / variant summary, as needed)
- ✅ Confirm raw data format(s) and fields needed for a joinable table
  - Expected: variant identifiers (e.g., VariationID and/or VCF fields), clinical significance, review status (if used), and assembly
- [ ] Create/validate ingestion output artifact(s)
  - Target: `data/processed/clinvar_missense_strict.parquet` (or equivalent)
    - Columns to confirm: canonical variant fields, label, and any gene/protein/transcript identifiers (if available)
  - ✅ Define label mapping rules (high-confidence only)
  - Keep: Pathogenic/Likely Pathogenic vs Benign/Likely Benign
  - Exclude: VUS, conflicting interpretations
- ✅ Run basic QC
  - No duplicate variant IDs, consistent canonical variant representation

## Week 2 — Metadata standardization + (optional) validation
- ✅ Standardize variant identifiers/metadata
  - Minimal fields: `chrom`, `pos`, `ref`, `alt`, assembly (if present)
  - Preferred: transcript/gene/protein identifiers if provided
- ✅ Annotate consequences with Ensembl VEP and filter to missense-only
  - Keep: `missense_variant` (or equivalent missense annotation)
- ✅ Create a small pilot table (e.g., 1k–5k variants) and validate end-to-end training table creation
  - Acceptable artifact for Week 2: `TSV + NumPy embeddings` (Parquet also works in the `egn6933-variant-embeddings` env)

## Week 3 — Leakage-aware split design
- ✅ Implement gene/protein-aware train/val/test split plan
  - Constraint: all variants from the same gene/protein stay in one split
- ✅ Sanity-check split sizes and positive rates (train/val/test)
- ✅ Write split artifacts to disk
  - Store a `split` column in the processed Parquet and/or save split index files

## Week 4 — EDA deliverables + baseline readiness
- ✅ Finalize the curated dataset artifact
  - Parquet with `label` (0/1), `split` (train/val/test), and embedding vectors
- ✅ Produce core EDA plots/tables
  - Class balance overall + by split
  - Distribution by gene/protein (if available)
  - Embedding dimensionality checks and summary statistics
  - Missense consequence QC summary (e.g., retained fraction after filtering)
- ✅ Write down “go/no-go” checks before model training
  - Minimum positive class size
  - No leakage across gene/protein splits
  - No duplicate variants across splits
  - Embeddings present and consistent for all retained samples
