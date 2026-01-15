# Weeks 1–4 Execution Checklist

This is a concrete Week 1–4 plan aligned to the proposal: ClinVar → non-coding subset → EDA → split/window decisions.

## Week 1 — Data in hand + basic profiling
- [ ] Confirm ClinVar source file exists locally
  - Target: `data/clinvar/variant_summary.txt.gz` (symlink/copy from `../Project/data/clinvar/variant_summary.txt.gz`)
- [ ] Load file in notebook and validate schema (columns, dtypes, missingness)
- [ ] Define the label mapping rules (high-confidence only)
  - Keep: Pathogenic/Likely pathogenic vs Benign/Likely benign
  - Exclude: VUS, conflicting interpretations
- [ ] Decide which assembly to use for the first pass (default: GRCh38)
- [ ] Create a first-pass filtered table (CSV/Parquet) with:
  - `Chromosome`, `Start/Stop` or `PositionVCF`, `ReferenceAlleleVCF`, `AlternateAlleleVCF`, `Assembly`, label fields

## Week 2 — Non-coding definition + VEP plan
- [ ] Finalize what counts as “non-coding” consequences (list of consequence terms)
- [ ] Decide VEP execution strategy
  - Local VEP vs online vs Docker
  - Inputs/outputs: variant coordinates + assembly
- [ ] Create a small pilot set (e.g., 1k–5k variants) and run VEP end-to-end
- [ ] Validate that consequences look correct (spot-check)

## Week 3 — Split and window design decisions
- [ ] Implement chromosome holdout split plan on the curated labeled table
  - Use `docs/chromosome_split_design.md` + `scripts/make_splits.py`
- [ ] Sanity-check split sizes and positive rates (train/val/test)
- [ ] Decide the initial sequence window sizes to test
  - Planned sweep: 101 bp, 201 bp, 501 bp
- [ ] Define the variant representation needed for sequence extraction
  - `chrom`, `pos`, `ref`, `alt`, assembly

## Week 4 — EDA deliverables + readiness for embeddings
- [ ] Finalize the curated dataset artifact
  - Parquet with `label` (0/1) and `split` (train/val/test)
- [ ] Produce core EDA plots/tables:
  - Class balance overall + by split
  - Distribution by chromosome
  - Assembly breakdown (confirm single-assembly focus)
  - Variant types (SNV vs indel) and decision on whether to keep indels in v1
- [ ] Write down “go/no-go” checks before embedding generation
  - Minimum positive class size
  - No obvious leakage or duplicated variants across splits
  - Non-coding filter behaving as intended
