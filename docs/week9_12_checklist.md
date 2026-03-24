# Weeks 9–12 Execution Checklist

This is a concrete Week 9–12 plan aligned to Phase 3 (Model Refinement & Error Analysis): **advanced model exploration, error analysis, homology audit, visualization, and capstone narrative** using the Week 4 curated dataset and baselines from Weeks 5–8.

**Primary artifacts from Weeks 5–8:**
- `data/processed/week4_curated_dataset.parquet` (train/val/test splits, ESM2 embeddings, labels)
- `results/baseline_rf_seed37_bootstrap.json` (best baseline: RF, AUROC=0.9299)
- `docs/week8_summary.md` (comprehensive baseline findings and limitations)

**Primary new entrypoints for Weeks 9–12:**
- `scripts/xgboost_train_eval.py` (gradient boosting exploration)
- `notebooks/04_xgboost_gradient_boosting.ipynb` (interactive analysis)
- `scripts/error_analysis.py` (misclassified variant analysis; TBD)
- `scripts/embedding_visualization.py` (UMAP/t-SNE by label/split; TBD)
- `scripts/homology_audit.py` (leakage screening; TBD)

---

## Phase 3 (Weeks 9–12) — High-level deliverables

- [x] Train gradient boosting model (XGBoost) with stratified k-fold CV + Bayesian search
- [x] Statistically compare XGBoost vs. RandomForest baseline (DeLong test)
- [x] Error analysis on misclassified variants (both RF and XGBoost)
- [x] Homology-aware leakage audit (protein similarity screening across splits)
- [x] Embedding-space visualization (UMAP/t-SNE by label and split)
- [x] Capstone narrative finalization (methodology, findings, limitations, future work)
- [ ] Presentation and poster preparation

---

## Week 9 — Gradient boosting exploration (XGBoost/LightGBM)

**Status:** ✅ Completed

- [x] Implement XGBoost with stratified k-fold CV and Bayesian hyperparameter search
  - [x] Production script: `scripts/xgboost_train_eval.py` (340 lines, fully documented)
  - [x] Interactive notebook: `notebooks/04_xgboost_gradient_boosting.ipynb` (10 sections)
  - [x] Documentation: `docs/xgboost_implementation_guide.md`, `docs/xgboost_quick_reference.md`

- [x] Run Bayesian hyperparameter search (Optuna TPE sampler)
  - [x] 50 trials × 5-fold stratified CV on training set
  - [x] Best CV AUROC: 0.9738 (excellent generalization)
  - [x] Best hyperparameters found and saved: max_depth=6, learning_rate=0.0829, lambda=0.8373, subsample=0.7691, colsample_bytree=0.7395

- [x] Evaluate on held-out test set with Platt sigmoid calibration
  - [x] Test AUROC: 0.9265 (uncalibrated)
  - [x] Test AUPRC: 0.9437 (uncalibrated)
  - [x] Brier Score: 0.1334, Log Loss: 0.4110
  - [x] Saved comprehensive JSON report: `results/xgboost_train_eval_report.json`
  - [x] Generated visualizations: PR curves, ROC curves (saved as PNG)

- [x] Compare XGBoost vs. RandomForest baseline
  - [x] RF Test AUROC: 0.9299 vs XGBoost 0.9265 (−0.34% Δ AUROC)
  - [x] Interpretation: XGBoost underperforms RF slightly; results within RF's 95% CI [0.9074, 0.9496]
  - [x] Finding: Gradient boosting offers no measurable advantage; RF is optimal for this dataset

- [x] Key learnings documented for capstone narrative
  - [x] Strong CV→Test generalization indicates minimal overfitting
  - [x] Class imbalance handling via `scale_pos_weight` is effective
  - [x] Both XGBoost and RF achieve ~93% test AUROC, suggesting RF is well-suited to ESM2 embeddings
  - [x] This demonstrates the importance of model selection: not all advanced methods improve performance

**Recommended commands (Week 9):**
```bash
# Run XGBoost training (50 trials)
python scripts/xgboost_train_eval.py \
  --n-trials 50 \
  --plot-pr results/xgboost_pr_curves.png \
  --plot-roc results/xgboost_roc_curves.png

# Check results
cat results/xgboost_train_eval_report.json | python -m json.tool

# Run interactive notebook
jupyter notebook notebooks/04_xgboost_gradient_boosting.ipynb
```

---

## Week 10 — Statistical validation and error analysis

**Status:** ✅ Completed

### 10.1: Statistical Comparison (XGBoost vs. RandomForest)

- [x] Implement DeLong test for AUROC comparison
  - [x] Tested whether XGBoost AUROC (0.9265) differs from RF AUROC (0.9299)
  - [x] Result: DeLong p-value = 0.5523 (no statistically significant difference)
  - [x] Generated comparison outputs: `results/xgboost_vs_rf_statistical_comparison.json`, `results/xgboost_vs_rf_bootstrap_deltas.png`

- [x] Paired bootstrap confidence intervals
  - [x] 1000 paired bootstrap iterations on the same held-out test set
  - [x] Reported 95% CIs for ΔAUROC (XGB−RF) and ΔAUPRC (XGB−RF)
  - [x] ΔAUROC CI = [-0.01445, 0.00767], ΔAUPRC CI = [-0.01392, 0.00607] (both include zero)

**Recommended commands (Week 10.1):**
```bash
python scripts/compare_xgboost_vs_rf.py \
  --data data/processed/week4_curated_dataset.parquet \
  --rf-report results/baseline_rf_seed37_bootstrap.json \
  --xgb-report results/xgboost_train_eval_report.json \
  --bootstrap-iters 1000 \
  --out-json results/xgboost_vs_rf_statistical_comparison.json \
  --plot-deltas results/xgboost_vs_rf_bootstrap_deltas.png
```

### 10.2: Error Analysis on Misclassified Variants

- [x] Identify misclassified variants
  - [x] Extracted false positives and false negatives for both RF and XGBoost
  - [x] Captured variant-level identifiers and gene annotations on test split

- [x] Analyze error patterns
  - [x] Computed gene-level error rates and top error-prone genes
  - [x] Compared overlap/disagreement between RF and XGBoost errors
  - [x] Summarized confidence behavior for correct vs incorrect predictions
  - [→ Week 11.2] Embedding-space distance / outlier linkage (deferred to embedding visualization work)

- [x] Generate error analysis report
  - [x] Saved to: `results/error_analysis_report.json`
  - [x] Saved misclassified variants table: `results/error_analysis_misclassified_variants.csv`
  - [x] Saved confusion matrix visualization: `results/confusion_matrix_rf_vs_xgb.png`

**Recommended commands (Week 10.2):**
```bash
python scripts/error_analysis.py \
  --data data/processed/week4_curated_dataset.parquet \
  --rf-report results/baseline_rf_seed37_bootstrap.json \
  --xgb-report results/xgboost_train_eval_report.json \
  --out-json results/error_analysis_report.json \
  --out-csv results/error_analysis_misclassified_variants.csv \
  --plot-confusion results/confusion_matrix_rf_vs_xgb.png
```

---

## Week 11 — Homology audit and embedding visualization

**Status:** ✅ Completed

### 11.1: Homology-Aware Leakage Audit

- [x] Run embedding-proxy homology screen across train/val/test splits
  - [x] Implemented calibrated centroid-cosine audit in `scripts/homology_audit.py`
  - [x] Added background-calibrated thresholding + mutual-nearest-neighbor filtering
  - [x] Initial proxy result: `test_leakage_rate_proxy = 0.318` (down from 0.776 after calibration)
  - [x] Direct same-gene overlap across splits: 0 groups

- [x] Compute sequence-level protein similarity follow-up (required for final conclusion)
  - [x] Implemented `scripts/homology_sequence_followup.py` (Smith-Waterman alignment pipeline)
  - [x] Consumes flagged pairs from `results/homology_leakage_audit.json` (`train_vs_test` scope)
  - [x] Computes identity and coverage thresholds (default: identity >= 0.90, coverage >= 0.50)
  - [x] Resolved sequence source for flagged genes via UniProt retrieval
  - [x] Re-ran and confirmed leakage rate with completed alignments
  - [x] Final sequence result: flagged `KMT2D` (train) vs `ARID1A` (test), UniProt `O14686` vs `O14497`, identity `0.4394` with low coverage; pair not confirmed as high-homology leakage
  - [x] Confirmed test leakage rate: `0.0000`; material leakage: `False`

- [x] Generate proxy leakage audit report
  - [x] Saved to: `results/homology_leakage_audit.json`
  - [x] Saved narrative report: `results/homology_audit_report.txt`
  - [x] Reported potentially flagged variants in each split (proxy-based)
  - [x] Sequence-level confirmation completed for current flagged pair set

- [x] Generate sequence-level follow-up report scaffold
  - [x] Saved to: `results/homology_sequence_followup.json`
  - [x] Saved pair-level table: `results/homology_sequence_pair_results.csv`
  - [x] Saved narrative report: `results/homology_sequence_followup_report.txt`
  - [x] Final result status: `complete`

**Recommended commands (Week 11.1):**
```bash
# Calibrated embedding-proxy homology audit
python scripts/homology_audit.py \
  --data data/processed/week4_curated_dataset.parquet \
  --similarity-threshold 0.9 \
  --background-quantile 0.999 \
  --out-json results/homology_leakage_audit.json \
  --out-report results/homology_audit_report.txt

# Sequence-level confirmation on flagged train-test pairs
python scripts/homology_sequence_followup.py \
  --data data/processed/week4_curated_dataset.parquet \
  --audit-json results/homology_leakage_audit.json \
  --pair-scope train_vs_test \
  --fetch-uniprot \
  --identity-threshold 0.90 \
  --min-coverage 0.50 \
  --out-json results/homology_sequence_followup.json \
  --out-csv results/homology_sequence_pair_results.csv \
  --out-report results/homology_sequence_followup_report.txt

# Optional sensitivity run (less strict; allows one-way nearest neighbors)
python scripts/homology_audit.py \
  --data data/processed/week4_curated_dataset.parquet \
  --similarity-threshold 0.9 \
  --background-quantile 0.999 \
  --allow-one-way \
  --out-json results/homology_leakage_audit_oneway.json \
  --out-report results/homology_audit_report_oneway.txt
```

### 11.2: Embedding-Space Visualization

- [x] Dimensionality reduction pipeline implemented and executed
  - [x] `scripts/embedding_visualization.py` supports UMAP/t-SNE projections
  - [x] Generated label-colored and split-colored t-SNE plots
  - [x] Includes split-aware markers and JSON summaries
  - [x] Final high-resolution figure selection for presentation completed (`results/embedding_umap_by_split.png`)

- [x] Analysis
  - [x] Are classes well-separated in embedding space?
  - [x] Are there visible clusters or outliers?
  - [x] Do val/test samples overlap well with train (or are there distribution gaps)?
  - [x] Generate accompanying summary: 3–5 sentences describing embedding-space characteristics

**Week 11.2 analysis summary (embedding space):**
The embedding-space projection shows meaningful structure, with pathogenic and benign variants forming partially separated regions while still overlapping near boundary zones. Several localized dense regions are present, suggesting subpopulation clusters in ESM2 feature space rather than a single homogeneous manifold. Split-colored views indicate that train, validation, and test points cover comparable geometric regions, with no obvious large-scale distribution gap for the 5k curated set. This pattern is consistent with the observed model behavior: strong overall AUROC but concentrated errors near ambiguous boundary regions.

- [x] Save visualizations
  - [x] `results/embedding_tsne_by_label.png` (colored by label)
  - [x] `results/embedding_tsne_by_split.png` (colored by split)
  - [x] `results/embedding_umap_by_label.png` (requested UMAP; auto-fallback used in current environment)
  - [x] `results/embedding_umap_by_split.png` (colored by split)
  - [x] `results/embedding_umap_by_model_agreement.png` (colored by RF/XGBoost agreement; optional)

**Recommended commands (Week 11.2):**
```bash
# Embedding visualization (implement in new script: scripts/embedding_visualization.py)
python scripts/embedding_visualization.py \
  --data data/processed/week4_curated_dataset.parquet \
  --method umap \
  --color-by label \
  --out-png results/embedding_umap_by_label.png

python scripts/embedding_visualization.py \
  --data data/processed/week4_curated_dataset.parquet \
  --method umap \
  --color-by split \
  --out-png results/embedding_umap_by_split.png
```

---

## Week 12 — Capstone narrative and presentation

**Status:** 🟡 In Progress (execution started)

### Week 12 execution start (2026-03-18)

- [x] Start the actual Week 12 execution block by drafting `docs/capstone_summary.md`
- [x] Finalize Week 12 capstone narrative with calibrated proxy + sequence-confirmation findings
- [x] Complete embedding-visualization interpretation text for presentation

### 12.1: Capstone Write-Up

- [x] Finalize `README.md` and project documentation
  - [x] Update "Results Summary" section with Week 9–11 findings
  - [x] Add "Model Comparison" subsection: RF vs XGBoost, why RF won
  - [x] Add "Error Analysis Insights" subsection: top error patterns, gene-level heterogeneity
  - [x] Add "Homology Audit" subsection: leakage risk assessment
  - [x] Add "Limitations" section: acknowledge class imbalance, embedding dimensionality, limited dataset size
  - [x] Add "Future Work" section: larger datasets, deep learning (MLP/CNN), ensemble methods, clinical validation

- [x] Create capstone summary document
  - [x] File: `docs/capstone_summary.md` (2–3 pages, executive summary)
  - [x] Include: problem statement, methodology, key findings, limitations, recommendations
  - [x] Target audience: technical + non-technical stakeholders
  - [x] Note: repository support document only; final turn-in artifact is `Final Report/Morenu_EGN6933_FinalReport.tex`

- [x] Update `docs/week8_summary.md` to include Weeks 9–12 findings
  - [x] Add XGBoost results, error analysis, homology audit findings

### 12.2: Presentation & Poster

- [ ] Create final presentation slides (PowerPoint or Google Slides)
  - [ ] Slide 1: Title, student name, advisor, date
  - [ ] Slide 2: Problem statement and motivation
  - [ ] Slide 3: Dataset overview (size, splits, class distribution, gene-disjoint design)
  - [ ] Slide 4: Methodology (baseline models, XGBoost, evaluation protocol)
  - [ ] Slide 5: Results table (RF vs XGBoost test AUROC/AUPRC with CIs)
  - [ ] Slide 6: PR/ROC curves (best model)
  - [ ] Slide 7: Embedding visualization (UMAP colored by label/split)
  - [ ] Slide 8: Error analysis (top error genes, error patterns)
  - [ ] Slide 9: Homology audit findings
  - [ ] Slide 10: Limitations (class imbalance, dataset size, computational cost)
  - [ ] Slide 11: Future work (larger datasets, deep learning, clinical validation)
  - [ ] Slide 12: Conclusion and key takeaways

- [ ] Create poster (if required by course)
  - [ ] Standard dimensions: 36"×48" (or institution-specific)
  - [ ] Layout: title → problem → method → results → limitations → future work
  - [ ] Include key figures (results table, PR/ROC curves, embedding visualization, error analysis)
  - [ ] Target: readable from 3 feet away

**Recommended commands (Week 12):**
```bash
# Generate final summary artifacts
python scripts/generate_capstone_summary.py \
  --rf-report results/baseline_rf_seed37_bootstrap.json \
  --xgb-report results/xgboost_train_eval_report.json \
  --error-analysis results/error_analysis_report.json \
  --homology-audit results/homology_leakage_audit.json \
  --out-md docs/capstone_summary.md
```

---

## Definition of Done for Weeks 9–12

- [x] Gradient boosting model (XGBoost) trained with Bayesian search and stratified k-fold CV
- [x] XGBoost compared statistically vs. RandomForest baseline (interpretation: comparable performance)
- [x] Statistical significance test completed (DeLong test)
- [x] Error analysis report generated (misclassified variants, error patterns)
- [x] Homology-aware leakage audit completed (protein similarity screening)
- [x] Embedding-space visualization generated (UMAP/t-SNE by label and split)
- [x] Capstone narrative finalized (methods, findings, limitations, future work)
- [ ] Presentation slides and poster prepared
- [ ] All code documented and reproducible from command line
- [x] Final README updated with complete project story and recommendations

---

## Key Learnings & Story Elements for Capstone

**What we learned (Weeks 5–12):**
1. **Baseline selection is dataset-dependent**: RF significantly outperforms LogReg on ESM2 embeddings (AUROC 0.9299 vs 0.8223), but XGBoost offers no additional benefit (0.9265).
2. **Class imbalance handling matters**: `scale_pos_weight` in both RF and XGBoost effectively prevents threshold bias.
3. **Strong generalization indicates good train/val/test separation**: XGBoost CV AUROC (0.9738) → test AUROC (0.9265) shows minimal overfitting; gene-disjoint splits prevent biological leakage.
4. **Not all advanced methods improve performance**: Gradient boosting is well-suited for tabular data, but ESM2 embeddings already contain rich signal well-captured by shallow RF.
5. **Error analysis reveals systematic patterns**: Certain genes/proteins may have higher error rates (future target for focused data collection).
6. **Embedding visualization contextualizes model decisions**: UMAP/t-SNE clarifies whether errors occur at decision boundaries or in well-separated regions.

**Story for capstone:**
> We implemented a rigorous ML pipeline to classify pathogenic vs. benign missense variants using pre-trained protein language model (ESM2) embeddings. Starting with strong baselines (LogReg, RandomForest), we explored gradient boosting (XGBoost) as a natural progression. Surprisingly, RandomForest achieved comparable performance to XGBoost (AUROC 0.9299 vs 0.9265), suggesting that shallow trees are sufficient to capture variant pathogenicity signals in ESM2 embeddings. Our error analysis revealed systematic patterns in gene-level performance, providing concrete targets for future data collection and model refinement. This work demonstrates the importance of rigorous evaluation, careful baseline selection, and honest interpretation of results—even when advanced methods don't outperform simpler alternatives.

---

## Recommended Timeline

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 9 | XGBoost training + comparison | `xgboost_train_eval_report.json`, PR/ROC curves |
| 10 | Statistical testing + error analysis | DeLong test result, `error_analysis_report.json` |
| 11 | Homology audit + visualization | `homology_leakage_audit.json`, UMAP/t-SNE figures |
| 12 | Capstone narrative + presentation | Final README, slides, poster, capstone summary |

---

## Resources

**Statistical Testing:**
- DeLong et al. (1988): "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach"
- Implementation: `sklearn.metrics.roc_auc_score` or dedicated library (e.g., `roc_comparison` package)

**Dimensionality Reduction:**
- UMAP: https://umap-learn.readthedocs.io/ (faster, preserves global structure)
- t-SNE: https://scikit-learn.org/stable/modules/manifold.html#t-sne (slower, good for clusters)

**Error Analysis:**
- Focus on: False Positives (false alarms) and False Negatives (missed pathogenic variants)
- Visualize: confusion matrices, error-rate heatmaps by gene/consequence type

**Homology Audit:**
- BLAST: https://blast.ncbi.nlm.nih.gov/ (local or via NCBI API)
- AlphaFold2 PAE (Predicted Aligned Error): alternative for protein structure similarity
- Threshold guidance: >90% sequence identity typically indicates potential data leakage
