# Week 5 Baseline Conclusion (ESM2 missense variant classification)

**Primary metric:** test AUPRC (robust under class imbalance).

- **Evaluation protocol (what “generalization” means):** Models are trained/evaluated on the Week 4 curated dataset (`n=5000`) using a **gene-disjoint** train/val/test split, where all variants from a given gene are confined to a single split; the **test set therefore measures generalization to unseen genes/proteins**, not an IID random holdout.

- **Baseline winner (primary metric = test AUPRC):** A shallow **Random Forest (RF)** trained on fixed-length ESM2 embeddings is the clear baseline winner by **test AUPRC**.
  - LogReg: test AUROC=0.7663, test AUPRC=0.7365
  - RF: test AUROC=0.9306, test AUPRC=0.9063

The baseline comparison indicates that, using test AUPRC as the primary metric, the best-performing approach is a shallow Random Forest (RF) trained on fixed-length ESM2 embeddings. In other words, among the baseline models evaluated on the held-out test set, the RF achieves the strongest precision–recall performance, which is typically the most informative metric when class imbalance is a concern.

The numbers support this conclusion. Logistic Regression reaches a test AUROC of 0.7663 and a test AUPRC of 0.7365, while the Random Forest attains a much higher test AUROC of 0.9306 and test AUPRC of 0.9063. The RF therefore outperforms Logistic Regression on both ranking ability (AUROC) and precision–recall tradeoff (AUPRC), with the largest gap on the primary metric (AUPRC), making it the clear baseline winner.

- **Interpretation of the baseline gap:** The RF’s large improvement over Logistic Regression suggests the embedding→label relationship is **nonlinear** enough that a controlled nonlinear model adds meaningful signal, even under leakage-aware gene holdout.

- **Why keep Platt calibration (even when AUROC/AUPRC don’t change):** Platt (sigmoid) calibration fitted on validation preserves ranking metrics (AUROC/AUPRC unchanged) but improves **probability reliability** on test, which matters for downstream decision-making and threshold selection.
  - Test Brier: 0.1378 → 0.1181 (lower is better)
  - Test log loss: 0.4513 → 0.3804 (lower is better)

- **Week 5 takeaway and Week 6 setup:** Week 5 establishes a strong, reproducible baseline: **shallow RF + (optional) Platt calibration**. Week 6 will focus on robustness (≥2 split seeds) and uncertainty quantification (bootstrap CIs on the held-out test set) before any additional model complexity (e.g., MLP).

- **Limitations (important context):** Performance may vary across gene families with limited representation, and labels may contain clinical annotation noise.

**Reference baseline going forward:** All subsequent extensions will be compared against the **shallow RF baseline** selected in Week 5.

## Reproducibility pointers

- Reports: `results/week5_logreg_report.json`, `results/week5_rf_report.json`, `results/week5_rf_calibrated_report.json`
- Plots: `results/week5_rf_pr_cal_vs_uncal.png`, `results/week5_rf_reliability.png`, `results/week5_test_score_distributions.png`
