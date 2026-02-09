# Project Milestones & Progress Tracker

**Project Title:** Machine Learning Classification of Pathogenic vs. Benign Coding Genetic Variants Using Protein Language Model Embeddings  
**Student Name:** Angel Morenu  
**Faculty Advisor:** Dr. Fan  
**Last Updated:** February 8, 2026

---

## Pre-Course Checklist

- [x] Develop preliminary project idea (coding-variant pathogenicity classification)
- [x] Complete project abstract (half-page)
- [x] Research potential faculty advisors (Bioinformatics, Computational Biology, Genomics)
- [x] Contact faculty advisors (aim for 3-5 professors)
- [x] Secure faculty advisor agreement
- [x] Prepare presentation for first class

---

## Course Milestones

### Milestone 1: Project Planning & Setup
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [x] Present preliminary project idea in class
- [x] Draft formal project proposal (repo artifact under `project-proposal/`)
- [x] Confirm faculty advisor
- [x] Establish project scope (individual vs. group justification if applicable)

**Notes:**


---

### Milestone 2: Literature Review & Related Work
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Comprehensive literature review
- [ ] Related work analysis
- [ ] Identification of gaps and opportunities
- [ ] Refined problem statement

**Notes:**


---

### Milestone 3: Data Collection & Exploration
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [x] Dataset acquired and documented
- [x] Exploratory data analysis (EDA)
- [x] Data quality assessment
- [x] Preprocessing pipeline

**Notes:**


---

### Milestone 4: Methodology & Implementation Plan
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Detailed methodology document
- [ ] Technical architecture design
- [ ] Implementation timeline
- [ ] Risk assessment and mitigation plan

**Notes:**


---

### Milestone 5: Initial Development
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [x] Baseline model/system implementation
- [x] Initial results and metrics
- [x] Code repository with documentation
- [ ] Progress presentation

**Notes:**


---

### Milestone 6: Refinement & Optimization
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Improved model/system
- [ ] Comparative analysis
- [ ] Performance optimization
- [ ] Validation results

**Notes:**


---

### Milestone 7: Evaluation & Testing
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Comprehensive evaluation framework
- [ ] Testing results and analysis
- [ ] Error analysis and limitations
- [ ] Documentation updates

**Notes:**


---

### Milestone 8: Final Deliverables
**Target Date:** [TBD - check syllabus]

**Deliverables:**
- [ ] Final project report
- [ ] Complete code repository with README
- [ ] Final presentation slides
- [ ] Project demonstration/demo
- [ ] Poster (if required)

**Notes:**


---

## Weekly Progress Log

### Week 1 (January 2026)
**Date:** January 7–15, 2026

**Accomplished:**
- Received course welcome and syllabus
- Set up project workspace and documentation structure
- ✅ Finalized project scope: coding-variant pathogenicity classification using ESM2 embeddings
- ✅ Updated proposal/README/architecture docs to match the coding-variant scope
- ✅ Renamed repository to match the updated scope and updated local remote

**Next Steps:**
- Obtain the curated coding-variant dataset details (location/format/schema)
- Implement ingestion + QC checks and create a versioned processed dataset artifact (Parquet)
- Decide and implement a gene/protein-aware split strategy to prevent leakage
- Begin baseline model training on embeddings (logistic regression / random forest)

**Blockers/Questions:**
- Confirm how gene/protein identifiers are represented in the curated dataset (gene symbol vs transcript vs protein ID)
- Confirm expected embedding format (vector dim; storage as columns vs arrays)

---

### Week 2

**Date:** January 16–23, 2026

**Accomplished:**
- Ingested/standardized ClinVar-derived missense variant dataset inputs and identifiers (canonical key: `chr_pos_ref_alt`).
- Confirmed ESM2 embedding representation assumptions (fixed-length vectors; expected dim=2560).
- Began consolidating dataset orientation notes and pipeline checkpoints under `docs/` and `scripts/`.

**Next Steps:**
- Implement/confirm leakage-aware splitting (gene-disjoint) and generate split artifacts.
- Build a Week 4-style curated modeling dataset artifact (Parquet) that joins labels, splits, and embeddings.

**Blockers/Questions:**
- None blocking; continue verifying gene/protein identifier mapping used for grouping.


### Week 3

**Date:** January 24–31, 2026

**Accomplished:**
- Implemented/confirmed gene-disjoint split strategy for evaluation (no gene overlap across train/val/test).
- Built/validated intermediate processed artifacts needed for Week 4 curated dataset construction.

**Next Steps:**
- Generate final curated dataset artifact (Parquet) for modeling and run Week 4 EDA/GO-NO-GO checks.

**Blockers/Questions:**
- Ensure split generation is reproducible and will be re-run under multiple seeds in Week 6.

---

### Week 4

**Date:** February 1–7, 2026

**Accomplished:**
- Created the Week 4 curated modeling dataset: `data/processed/week4_curated_dataset.parquet`.
- Validated dataset integrity (no duplicate `chr_pos_ref_alt`, expected columns, consistent embedding dimension).
- Ran Week 4 EDA + go/no-go checks (`scripts/week4_eda.py`) → GO.

**Next Steps:**
- Train baseline models using the curated dataset (LogReg, RF) and save reports/plots.

**Blockers/Questions:**
- None blocking.

---

### Week 5

**Date:** February 8, 2026

**Accomplished:**
- Trained Week 5 baselines on ESM2 embeddings using the gene-disjoint split and saved outputs under `results/`.
	- Logistic Regression: test AUROC=0.7663, test AUPRC=0.7365
	- Random Forest (RF): test AUROC=0.9306, test AUPRC=0.9063
- Added Platt calibration for RF to improve probability quality (Brier/log loss improvements) and generated reliability/PR plots.
- Wrote Week 5 baseline conclusion: `docs/week5_baseline_conclusion.md`.
- Updated Weeks 5–8 execution plan: `docs/week5_8_checklist.md`.

**Next Steps:**
- Week 6: robustness across ≥2 split seeds + bootstrap confidence intervals (CIs) on held-out test performance.
- Keep **shallow RF** as the reference baseline for all subsequent comparisons.

**Blockers/Questions:**
- None; proceed with split-seed robustness and CIs.

---

## Advisor Meeting Notes

### Meeting 1
**Date:** [TBD]

**Attendees:** Angel Morenu, Dr. Fan (advisor), Dylan Tan (dataset/orientation)

**Topics Discussed:**
- Curated missense dataset location/format/schema and how to load large PKLs safely (line-by-line).
- Identifier conventions and join keys (notably `chr_pos_ref_alt`).
- Embedding variants (baseline ESM2 vs coordinates-augmented variant) and keeping modeling scope controlled.
- Evaluation design emphasizing leakage-aware gene-disjoint generalization.

**Action Items:**
- Keep dataset orientation documented (`docs/dataset_orientation_dylan_tan.md`).
- Use `data/processed/week4_curated_dataset.parquet` as the modeling “source of truth.”
- Establish/retain a strong reference baseline (Week 5 shallow RF + optional Platt calibration).

**Next Meeting:** [TBD]
### Key Papers
1. ESM-2 (protein language model embeddings) — primary representation approach
2. Gene/protein-aware evaluation and leakage control for variant prediction
3. Baseline classifiers for tabular embeddings (logistic regression, random forest)

### Datasets
1. Public curated coding-variant dataset (provided by Dylan)
2. ClinVar (optional validation/standardization): https://www.ncbi.nlm.nih.gov/clinvar/
3. Ensembl VEP (optional consequence annotation): https://www.ensembl.org/info/docs/tools/vep/

### Tools & Frameworks

1. PyTorch / TensorFlow (deep learning)
2. scikit-learn (baseline models, evaluation)
3. pandas, numpy, pyarrow (data processing)
4. MLflow / Weights & Biases (experiment tracking)
5. Streamlit / Gradio (demo interface)
6. HiPerGator (UF Research Computing) or Google Colab Pro

### Useful Links

1. ESM / ESM-2 project resources (overview + model cards)
2. ClinVar documentation and release notes
3. Ensembl VEP docs (if used for optional annotation)

---

## Notes & Ideas

[Use this section for quick notes, ideas, or things to remember]

