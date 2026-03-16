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
- [ ] Homology-aware leakage audit (protein similarity screening across splits)
- [ ] Embedding-space visualization (UMAP/t-SNE by label and split)
- [ ] Capstone narrative finalization (methodology, findings, limitations, future work)
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

**Status:** 🟡 Planned (per Dr. Fan recommendation)

### 11.1: Homology-Aware Leakage Audit

- [ ] Compute protein sequence similarity across train/val/test splits
  - [ ] Use BLAST or Smith-Waterman alignment to screen for highly similar proteins
  - [ ] Flag variants on proteins with >90% sequence identity across split boundaries
  - [ ] Estimate false leakage rate (# variants in "leaked" proteins / total test variants)

- [ ] Generate leakage audit report
  - [ ] Saved to: `results/homology_leakage_audit.json`
  - [ ] Report number of potentially leaked variants in each split
  - [ ] If significant leakage detected (>5% of test set), consider re-evaluating with filtered test set
  - [ ] Document findings and mitigation strategy

**Recommended commands (Week 11.1):**
```bash
# Homology audit (implement in new script: scripts/homology_audit.py)
python scripts/homology_audit.py \
  --data data/processed/week4_curated_dataset.parquet \
  --alignment-method blast \
  --similarity-threshold 0.9 \
  --out-json results/homology_leakage_audit.json \
  --out-report results/homology_audit_report.txt
```

### 11.2: Embedding-Space Visualization

- [ ] Dimensionality reduction (UMAP or t-SNE)
  - [ ] Reduce 2560-dimensional ESM2 embeddings to 2D using UMAP (preferred; faster) or t-SNE (preferred; interpretable)
  - [ ] Color by label (benign=blue, pathogenic=red)
  - [ ] Overlay split information (markers: train=circle, val=square, test=triangle)
  - [ ] Generate high-resolution figure for capstone presentation

- [ ] Analysis
  - [ ] Are classes well-separated in embedding space?
  - [ ] Are there visible clusters or outliers?
  - [ ] Do val/test samples overlap well with train (or are there distribution gaps)?
  - [ ] Generate accompanying summary: 3–5 sentences describing embedding-space characteristics

- [ ] Save visualizations
  - [ ] `results/embedding_umap_by_label.png` (colored by label)
  - [ ] `results/embedding_umap_by_split.png` (colored by split)
  - [ ] `results/embedding_umap_by_model_agreement.png` (colored by RF/XGBoost agreement; optional)

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

**Status:** 🟡 Planned

### 12.1: Capstone Write-Up

- [ ] Finalize `README.md` and project documentation
  - [ ] Update "Results Summary" section with Week 9–11 findings
  - [ ] Add "Model Comparison" subsection: RF vs XGBoost, why RF won
  - [ ] Add "Error Analysis Insights" subsection: top error patterns, gene-level heterogeneity
  - [ ] Add "Homology Audit" subsection: leakage risk assessment
  - [ ] Add "Limitations" section: acknowledge class imbalance, embedding dimensionality, limited dataset size
  - [ ] Add "Future Work" section: larger datasets, deep learning (MLP/CNN), ensemble methods, clinical validation

- [ ] Create capstone summary document
  - [ ] File: `docs/capstone_summary.md` (2–3 pages, executive summary)
  - [ ] Include: problem statement, methodology, key findings, limitations, recommendations
  - [ ] Target audience: technical + non-technical stakeholders

- [ ] Update `docs/week8_summary.md` to include Weeks 9–12 findings
  - [ ] Extend to `docs/week8_12_comprehensive_summary.md`
  - [ ] Add XGBoost results, error analysis, homology audit findings

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
- [ ] Homology-aware leakage audit completed (protein similarity screening)
- [ ] Embedding-space visualization generated (UMAP/t-SNE by label and split)
- [ ] Capstone narrative finalized (methods, findings, limitations, future work)
- [ ] Presentation slides and poster prepared
- [ ] All code documented and reproducible from command line
- [ ] Final README updated with complete project story and recommendations

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
