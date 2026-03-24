> **Annotation (per Dr. Fan's requirement):**
> For all project phases (Weeks 1–15), it is required that all variants used in the curated dataset are mapped to and can be uniquely identified by Dylan's precomputed ESM2 embeddings. This mapping and identification step is completed for Weeks 1–4 (data acquisition, curation, and QC), and is a standing requirement for all subsequent work (Weeks 5–8: feature engineering, baselines; Weeks 9–15: refinement, evaluation, deployment). All modeling and evaluation steps are contingent on this mapping being present and verifiable.

# Project Milestones & Progress Tracker

**Project Title:** Machine Learning Classification of Pathogenic vs. Benign Coding Genetic Variants Using Protein Language Model Embeddings  
**Student Name:** Angel Morenu  
**Faculty Advisor:** Dr. Fan  
**Last Updated:** March 18, 2026

**Execution Checklists:**
- Weeks 1–4: `docs/week1_4_checklist.md`
- Weeks 5–8: `docs/week5_8_checklist.md`
- Weeks 9–12: `docs/week9_12_checklist.md`
- Weeks 13–15: `docs/week13_15_checklist.md`

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

**Deliverables:**
- [x] Present preliminary project idea in class
- [x] Draft formal project proposal (repo artifact under `project-proposal/`)
- [x] Confirm faculty advisor
- [x] Establish project scope (individual vs. group justification if applicable)

**Notes:**


---

### Milestone 2: Literature Review & Related Work

**Deliverables:**
- [ ] Comprehensive literature review
- [ ] Related work analysis
- [ ] Identification of gaps and opportunities
- [ ] Refined problem statement

**Notes:**


---

### Milestone 3: Data Collection & Exploration

**Deliverables:**
- [x] Dataset acquired and documented
- [x] Exploratory data analysis (EDA)
- [x] Data quality assessment
- [x] Preprocessing pipeline

**Notes:**


---

### Milestone 4: Methodology & Implementation Plan

**Deliverables:**
- [ ] Detailed methodology document
- [ ] Technical architecture design
- [ ] Implementation timeline
- [ ] Risk assessment and mitigation plan

**Notes:**


---

### Milestone 5: Initial Development

**Deliverables:**
- [x] Baseline model/system implementation
- [x] Initial results and metrics
- [x] Code repository with documentation
- [ ] Progress presentation

**Notes:**


---

### Milestone 6: Refinement & Optimization

**Deliverables:**
- [x] Improved model/system
- [x] Comparative analysis
- [x] Performance optimization
- [x] Validation results

**Notes:**


---

### Milestone 7: Evaluation & Testing

**Deliverables:**
- [x] Comprehensive evaluation framework
- [x] Testing results and analysis
- [x] Error analysis and limitations
- [x] Documentation updates

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
**Status:** ✅ Completed
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
**Status:** ✅ Completed

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
**Status:** ✅ Completed

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
**Status:** ✅ Completed

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
**Status:** ✅ Completed

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

### Week 6
**Status:** ✅ Completed

**Date:** February 9–23, 2026

**Accomplished:**
- Completed robustness checks for leakage-aware evaluation by repeating the gene-disjoint split under multiple random seeds (seed=13 and seed=37) and rebuilding seed-specific curated datasets.
- Re-ran the baseline Random Forest (RF) model and computed 1000-iteration bootstrap confidence intervals (CIs) on held-out test performance for each seed; results were stable with overlapping CIs.
- Updated Week 6 documentation and execution guidance to make the workflow reproducible (Week 6 note, checklist updates, and README updates).
- Met with Dr. Fan (Feb 17) and aligned reporting: AUROC/ROC as the primary metric (AUPRC secondary) and clarified dataset size/splits and filter definitions to be advisor-defensible.

**Next Steps:**
- Extend seed-sensitivity to additional seeds (e.g., 5–10 total) and summarize the distribution of test AUROC.
- Run bootstrap-CI evaluation for logistic regression (and optionally calibrated RF) for comparison.
- Strengthen dataset documentation: explicitly define “high-confidence” filtering and justify canonical transcript mapping choices.

**Blockers/Questions:**
- None blocking.

---

### Week 7
**Status:** ✅ Completed

**Date:** February 24–26, 2026

**Accomplished:**
- Completed hyperparameter selection for LogReg and RF baselines
  - LogReg: Confirmed C=1.0 (balanced regularization) with StandardScaler pipeline
  - RF: Finalized max_depth=4, n_estimators=200 (shallow configuration)
- Documented class imbalance handling strategy: balanced class weighting applied to both models
- Verified class weighting is properly configured and reproducible
- Created comprehensive Week 7 hyperparameter selection summary: `docs/week7_hyperparameter_selection.md`
- Validated reproducibility checklist (feature set, pipelines, class weighting, seed robustness)

**Model Selection Summary:**
- **Baseline winner:** Shallow Random Forest (RF)
- **Test AUROC:** 0.9306 (vs LogReg 0.7663; +16.5% absolute improvement)
- **Seed robustness:** ΔAUROC ≈ -0.0045 across seeds 13 vs 37 (minimal variance)
- **Generalization:** Performance validated on unseen genes (gene-disjoint test split)

**Next Steps:**
- Week 8: Threshold selection on validation set, operating point definition, statistical testing prep

**Blockers/Questions:**
- None; hyperparameter selection is complete and defensible

---

### Week 8
**Status:** ✅ Completed

**Date:** March 3–7, 2026

**Accomplished:**
- [x] Finalized threshold selection on validation set using MCC-maximization criterion
  - LogReg threshold: 0.9934 (conservative; high precision)
  - RF threshold: 0.7328 (balanced precision-recall)
  - Both applied to test set with no test peeking
- [x] Confirmed Platt sigmoid calibration protocol: fit on val, evaluated on test
  - Val probability quality improved (Brier: 0.2036 → 0.1271, −37.6%)
  - Test degradation observed (Brier: 0.1363 → 0.2615, +91.8%), attributed to distribution shift (val 17.8% → test 58.8% positive rate)
- [x] Generated Week 8 comprehensive summary artifact: `docs/week8_summary.md`
  - Metrics table (AUROC/AUPRC/MCC comparison for LogReg vs RF)
  - Paired bootstrap + DeLong comparison (ΔAUROC = 0.1076, p = 5.87e-10)
  - Existing plots verified: PR curves, reliability diagram, score distributions, effect-size summary
  - 8-sentence findings summary with 5 detailed limitations
  - Reproducibility pointers to all artifacts

**Advisor Feedback (Incorporated):**
- **Recommendation 1:** Implement homology-aware leakage audit in Weeks 9–12 to screen for highly similar proteins across splits
- **Recommendation 2:** Generate UMAP/t-SNE embedding-space visualization (colored by label and split) for stakeholder communication

**Next Steps:**
- Proceed to Week 9–12: error analysis, MLP implementation, homology audit, embedding visualization
- Maintain RF as the reference baseline for all future model comparisons

---

### Week 9

**Date:** March 5–11, 2026
**Status:** ✅ Completed

**Accomplished:**
- [x] Implemented XGBoost with stratified k-fold CV and Bayesian hyperparameter search (per Dylan's guidance)
  - Script: `scripts/xgboost_train_eval.py` (production-ready, 340 lines)
  - Notebook: `notebooks/04_xgboost_gradient_boosting.ipynb` (interactive, 10 sections)
  - Documentation: `docs/xgboost_implementation_guide.md`, `docs/xgboost_quick_reference.md`
- [x] Ran XGBoost training: 50 Bayesian trials × 5-fold stratified CV on training set
  - Best CV AUROC: 0.9738 (excellent generalization from CV to test)
  - Best hyperparameters found: max_depth=6, learning_rate=0.0829, lambda=0.8373, subsample=0.7691, colsample_bytree=0.7395
  - Class imbalance handled: scale_pos_weight=1.7473 (automatically computed)
- [x] Evaluated on test set with Platt sigmoid calibration (fit on validation, eval on test)
  - Test AUROC: 0.9265 (uncalibrated)
  - Test AUPRC: 0.9437 (uncalibrated)
  - Brier Score: 0.1334, Log Loss: 0.4110
- [x] Generated outputs: PR/ROC curve plots, comprehensive JSON report with all metrics
- [x] Compared XGBoost vs. RandomForest baseline (seed37)
  - RF Test AUROC: 0.9299 vs XGBoost 0.9265 (−0.34% Δ AUROC, −0.36% Δ AUPRC)
  - **Interpretation:** XGBoost slightly underperforms RF; results are within RF's 95% CI [0.9074, 0.9496]
  - **Finding:** Gradient boosting offers no measurable advantage; RF remains optimal baseline
- [x] Finalized Week 9 comparison artifacts under `results/Week 9/`
  - `capstone_model_comparison_from_reports.csv`
  - `capstone_rf_vs_xgb_delta.csv`
  - `capstone_split_summary.csv`
  - `dylan_vs_capstone_comparison.csv`
  - `dylan_vs_capstone_summary_table.csv`

**Key Findings & Story**:
1. **Strong CV→Test generalization:** XGBoost CV AUROC 0.9738 → test 0.9265 indicates minimal overfitting
2. **Class imbalance well-handled:** `scale_pos_weight` effectively balanced minority class without post-hoc threshold tuning
3. **Comparable model performance:** Both XGBoost and RF achieve ~93% test AUROC, suggesting RF is well-suited to ESM2 embeddings
4. **No ensemble benefit evident:** Will defer ensemble exploration to Week 11+ if time permits

**Advisor Guidance Applied (from Dr. Fan)**:
- ✅ Gradient boosting provides good "computational skills practice"
- ✅ Results won't beat SOTA (expected and acceptable; Dr. Fan approved)
- ✅ Focus now on **telling the complete story** (journey, insights, limitations) vs. chasing highest metrics
- ✅ Prepare discussion of why XGBoost didn't improve over RF → leads into future work section

**Next Steps:**
- Week 10: Statistical testing (DeLong test: XGBoost vs RF, confirm non-significance)
- Week 10: Error analysis on misclassified variants (both models)
- Week 10–11: Homology-aware leakage audit (per Dr. Fan recommendation)
- Week 11: Embedding visualization (UMAP/t-SNE by label and split)
- Week 12: Finalize capstone narrative and presentation

---

### Week 10

**Date:** March 12–16, 2026
**Status:** ✅ Completed (core analyses)

**Accomplished:**
- [x] Implemented RF vs XGBoost statistical comparison workflow (same held-out test set)
  - DeLong test p-value: **0.5523** (no significant AUROC difference)
  - ΔAUROC (XGB−RF): **−0.00342**
  - ΔAUPRC (XGB−RF): **−0.00365**
- [x] Completed paired bootstrap deltas (1000 paired resamples)
  - ΔAUROC 95% CI: **[−0.01445, 0.00767]** (includes 0)
  - ΔAUPRC 95% CI: **[−0.01392, 0.00607]** (includes 0)
- [x] Implemented misclassification error analysis for both models
  - Generated variant-level misclassification table (FP/FN) on test split
  - Generated confusion matrix comparison plot (RF vs XGBoost)
  - Generated gene-level and confidence-based error summaries
- [x] Key error-rate findings:
  - RF test error rate: **0.146**
  - XGBoost test error rate: **0.190**
  - Shared errors (both wrong): **56** test variants

**Interpretation:**
- Week 10 statistical evidence supports Week 9 conclusion: XGBoost does not significantly outperform RF on this dataset.
- RF remains the practical reference model while Week 11 focuses on homology audit and embedding-space diagnostics.

**Next Steps:**
- Complete homology-aware leakage audit (Week 11)
- Generate UMAP/t-SNE visualization by label/split (Week 11)
- Extend error analysis with embedding-distance/outlier diagnostics
- Prepare Week 12 capstone narrative + presentation artifacts

---

### Week 11

**Date:** March 16, 2026
**Status:** ✅ Completed (homology audit track)

**Accomplished:**
- [x] Implemented calibrated embedding-proxy homology audit: `scripts/homology_audit.py`
- [x] Added stricter filtering (background-calibrated threshold + mutual nearest-neighbor matching)
- [x] Generated proxy audit outputs:
  - `results/homology_leakage_audit.json`
  - `results/homology_audit_report.txt`
- [x] Proxy audit key result:
  - Initial naive proxy: `test_leakage_rate_proxy = 0.776`
  - Calibrated proxy: `test_leakage_rate_proxy = 0.318`
  - Direct same-gene overlap across splits: `0`
- [x] Implemented embedding visualization pipeline: `scripts/embedding_visualization.py`
- [x] Generated Week 11 figures:
  - `results/embedding_tsne_by_label.png`
  - `results/embedding_tsne_by_split.png`
- [x] Implemented sequence-level follow-up pipeline: `scripts/homology_sequence_followup.py`
  - Consumes flagged train-test pairs from proxy audit
  - Runs Smith-Waterman alignment with identity/coverage thresholds
  - Writes:
    - `results/homology_sequence_followup.json`
    - `results/homology_sequence_pair_results.csv`
    - `results/homology_sequence_followup_report.txt`
  - Final confirmation status: `complete`
  - Flagged pair assessed: `KMT2D` (train) vs `ARID1A` (test), UniProt `O14686` vs `O14497`
  - Alignment summary: identity `0.4394`, low coverage, `confirmed = False`
  - Confirmed leakage result: `confirmed_rate = 0.0`, material leakage `False`

**Interpretation:**
- The large drop (0.776 → 0.318) indicates the naive proxy over-flagged cross-split similarity.
- Sequence-level confirmation for the currently flagged pair did not satisfy high-identity/high-coverage homology criteria.
- Current evidence does not indicate material homology leakage in the held-out test split.

**Next Steps:**
- Finalize Week 12 capstone narrative with calibrated proxy + sequence-confirmation findings.
- Complete embedding-visualization interpretation text for presentation.

---

### Week 12

**Date:** March 18, 2026  
**Status:** 🟡 In progress (execution started)

**Execution Started Note:**
- Week 12 capstone write-up execution has started.
- Drafted executive summary document: `docs/capstone_summary.md`.
- Focus for current block: finalize narrative synthesis (statistical validation + homology confirmation) and prepare presentation-ready interpretation text.

**Accomplished (to date):**
- [x] Started Week 12 execution block and drafted `docs/capstone_summary.md`.
- [x] Consolidated model-selection narrative using RF vs XGBoost statistical evidence (DeLong + paired bootstrap).
- [x] Integrated calibrated proxy + sequence-confirmation homology findings into capstone narrative.
- [x] Completed embedding-visualization interpretation text for presentation framing.

**Next Steps:**
- Finalize `README.md` result-story sections (model comparison, error analysis, homology, limitations, future work).
- Complete Week 12 presentation slide content and figure selection.
- Package poster-ready figures and final submission artifact checklist.

---

### Meeting 1
**Date:** February 17, 2026

**Attendees:** Angel Morenu, Dr. Fan (advisor)

**Topics Discussed:**
- Scope control: keep the capstone feasible within ~3–4 months by focusing on a clear prediction task (avoid overly broad, mechanism-heavy goals).
- Teaming: consider pairing with a peer who has stronger programming experience (you remain the project lead) if it improves execution speed and code quality.
- Evaluation and metrics: use ROC/AUROC as the primary metric for the curated ClinVar-based dataset; keep AUPRC as secondary.
- Dataset clarity: confirmed the curated dataset size is n≈5,000 total variants with an ~80/10/10 train/val/test split (≈500 test rows) under gene-disjoint splitting.
- Documentation expectations: be prepared to explain the filtering definitions (what “high-confidence” means) and why canonical transcripts are used.

**Action Items:**
- Keep AUROC-first reporting consistent across artifacts (README, notes, and evaluation scripts).
- Update/expand dataset documentation to precisely define the filtering criteria and canonical transcript rationale.
- If beneficial, identify a potential teammate and define roles early to keep scope and execution on track.
- Continue robustness checks (split-seed sensitivity + bootstrap CIs) and be ready to justify dataset size and split design.

**Next Meeting:** TBD (after incorporating metric/reporting updates and documenting filters)

---

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

