# Capstone Summary (Weeks 1–12)

## Executive Summary
This capstone built a reproducible machine learning pipeline to classify ClinVar missense variants as pathogenic vs. benign using precomputed ESM2 protein language model embeddings. The strongest baseline model was shallow Random Forest (RF), and a Week 9–10 XGBoost expansion was evaluated with formal statistical testing rather than point-metric comparison alone. Final evidence shows RF and XGBoost are statistically comparable on the held-out gene-disjoint test split, with RF retained as the practical reference model due to stable performance and simpler deployment profile. A Week 11 homology audit added biological leakage controls beyond gene-disjoint splitting and found no confirmed material cross-split sequence leakage.

## Problem Statement
Clinical interpretation of missense variants is slower than variant discovery. This project addresses that gap with a robust prediction pipeline that prioritizes evaluation rigor, leakage prevention, and reproducibility over model novelty.

## Data, Features, and Split Design
- **Label source:** ClinVar 20240805 missense-only subset with strict pathogenic/benign filtering.
- **Canonical ID:** `chr_pos_ref_alt` for traceability across pipeline stages.
- **Features:** Dylan Tan lab-provided precomputed ESM2 embedding vectors.
- **Split strategy:** gene-disjoint train/validation/test splits to measure generalization to unseen genes and reduce biological leakage risk.

## Modeling Workflow
1. Baseline models: Logistic Regression and shallow Random Forest.
2. Advanced model exploration: XGBoost with Bayesian hyperparameter search (50 trials, stratified 5-fold CV).
3. Calibration and held-out evaluation on test split.
4. Statistical model comparison using DeLong and paired bootstrap deltas.
5. Error analysis, embedding visualization, and homology-aware leakage audit.

## Key Findings
- **Random Forest (reference model):** AUROC = `0.9299`, AUPRC = `0.9473`.
- **XGBoost:** AUROC = `0.9265`, AUPRC = `0.9437`.
- **Observed deltas (XGB − RF):** AUROC = `-0.00342`, AUPRC = `-0.00365`.
- **Interpretation:** XGBoost did not provide measurable improvement over RF for this curated ESM2 feature setting.

## Statistical Validation (Week 10)
- **DeLong AUROC comparison:** `p = 0.5523` (no statistically significant difference).
- **Paired bootstrap (1000 resamples):**
  - ΔAUROC 95% CI: `[-0.01445, 0.00767]`
  - ΔAUPRC 95% CI: `[-0.01392, 0.00607]`
- Both confidence intervals include zero, supporting the conclusion that performance differences are not significant.

## Homology-Aware Leakage Audit (Week 11)
A two-stage audit was used:
1. **Embedding-proxy screening** for potentially similar train/test proteins.
2. **Sequence-level confirmation** with Smith-Waterman on flagged pairs.

Findings:
- Proxy leakage estimate was reduced after calibration and filtering.
- Sequence-level follow-up for the currently flagged pair (`KMT2D` train vs `ARID1A` test) did not meet high-homology confirmation thresholds.
- **Confirmed leakage rate:** `0.0` (non-material).

## Error Analysis and Embedding Interpretation
- RF and XGBoost showed substantial overlap in misclassified variants, suggesting stable, shared hard cases rather than model-specific failure patterns.
- Embedding-space projections show partial separation between pathogenic and benign classes, with overlap near decision-boundary regions.
- Dense local clusters indicate heterogeneous subpopulations in ESM2 feature space rather than a single simple manifold.
- Split-colored projections indicate train/validation/test points occupy comparable regions, with no obvious large-scale distribution gap in the curated 5k set.
- Presentation takeaway: the model is strong overall, and most remaining errors are concentrated near ambiguous boundary zones.

## Limitations
- Curated strict-label subset size limits rare-pattern coverage.
- ESM2 dimensions are not inherently human-interpretable without additional attribution layers.
- Results are evaluated on a specific curated setting and should be externally validated before clinical use.

## Evidence-Based Extensions (Next Steps)
1. Validate the final RF model on an independent external set.
2. Add post-hoc explainability (e.g., SHAP) for case-level interpretation.
3. Explore calibrated ensemble combinations with established variant predictors.
4. Expand data coverage to improve performance on rare or boundary variants.

## Reproducibility Artifacts
Primary artifacts supporting this summary:
- `results/baseline_rf_seed37_bootstrap.json`
- `results/xgboost_train_eval_report.json`
- `results/xgboost_vs_rf_statistical_comparison.json`
- `results/error_analysis_report.json`
- `results/homology_leakage_audit.json`
- `results/homology_sequence_followup.json`
- `results/embedding_tsne_by_label.png`
- `results/embedding_tsne_by_split.png`
- `results/embedding_umap_by_split.png`
- `results/embedding_umap_by_model_agreement.png`

## Stakeholder Summary (Non-Technical)
The project successfully built a dependable classifier for missense variant pathogenicity using modern protein embeddings and careful scientific validation. We tested an advanced model (XGBoost), but it did not beat the simpler Random Forest in a statistically meaningful way. We also audited potential biological leakage and found no confirmed material leakage in the final held-out test set. The final recommendation is to use Random Forest as the reference model for deployment and reporting in this capstone.
