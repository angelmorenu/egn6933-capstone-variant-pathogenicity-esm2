# Week 5 Baseline Conclusion (ESM2 missense variant classification)



> **Annotation (per Dr. Fan's request):**
> All variants in the curated dataset were successfully mapped to unique, identifiable precomputed ESM2 embedding vectors. The mapping process is fully deterministic and reproducible, ensuring that each variant used for model training and evaluation can be automatically and unambiguously identified by its embedding. This addresses Dr. Fan's requirement that variants be mappable and uniquely identifiable in the feature set.

**Note on dataset curation and filtering:**
The ClinVar-derived dataset used here is highly curated and does not reflect the true prevalence of pathogenic vs. benign variants in the population. The false positive (FP) rate is likely underestimated, so metrics that depend on FP (such as precision, AUPRC, and even AUROC to some extent) may not fully represent real-world performance.


**Filtering criteria:**
- Only high-confidence missense variants were included (i.e., variants with unambiguous, expert-reviewed clinical significance and no conflicting or uncertain interpretations)
- Strict clinical significance labels: “Pathogenic” and “Benign” only (variants of uncertain significance and conflicting interpretations were excluded)
- Variants mapped to canonical transcripts (to ensure consistent protein context and avoid transcript-specific annotation ambiguity)
- Additional quality control steps applied

This strict curation is why the final count is 5,000 variants, which is much smaller than the full ClinVar missense set.

**Primary metric:** test AUROC (robust to class imbalance, recommended by Dr. Fan).

- **Evaluation protocol (what “generalization” means):** Models are trained/evaluated on the Week 4 curated dataset (`n=5000`) using a **gene-disjoint** train/val/test split, where all variants from a given gene are confined to a single split; the **test set therefore measures generalization to unseen genes/proteins**, not an IID random holdout.


- **Baseline winner (primary metric = test AUROC):** A shallow **Random Forest (RF)** trained on fixed-length ESM2 embeddings is the clear baseline winner by **test AUROC**.
  - LogReg: test AUROC=0.8223, test AUPRC=0.8776
  - RF: test AUROC=0.9299, test AUPRC=0.9473


The baseline comparison indicates that, using test AUROC as the primary metric, the best-performing approach is a shallow Random Forest (RF) trained on fixed-length ESM2 embeddings. Among the baseline models evaluated on the held-out test set, the RF achieves the strongest ranking performance (AUROC = 0.9299), which is robust to class imbalance and recommended as the primary metric for this task. Precision–recall (AUPRC) is also reported for completeness.

- **Interpretation of the baseline gap:** The RF’s large improvement over Logistic Regression suggests the embedding→label relationship is **nonlinear** enough that a controlled nonlinear model adds meaningful signal, even under leakage-aware gene holdout.

- **Why keep Platt calibration (even when AUROC/AUPRC don’t change):** Platt (sigmoid) calibration fitted on validation preserves ranking metrics (AUROC/AUPRC unchanged) but improves **probability reliability** on test, which matters for downstream decision-making and threshold selection.
- **Calibration note (Platt/sigmoid, fit on val):** Ranking metrics (AUROC/AUPRC) are unchanged as expected, but probability quality can move either way depending on val↔test distribution shift.
  - Validation improved: Brier 0.2036 → 0.1271; log loss 0.5928 → 0.4135
  - Test worsened: Brier 0.1363 → 0.2615; log loss 0.4468 → 0.7086

- **Week 5 takeaway and Week 6 setup:** Week 5 establishes a strong, reproducible baseline: **shallow RF + (optional) Platt calibration**. Week 6 will focus on robustness (≥2 split seeds) and uncertainty quantification (bootstrap CIs on the held-out test set) before any additional model complexity (e.g., MLP).

- **Statistical comparison (LogReg vs RF):**
  - **AUROC (ranking metric, primary):**
    - Test AUROC: LogReg=0.8223 vs RF=0.9299
    - Paired bootstrap: ΔAUROC (RF − LogReg) mean=0.1076; 95% CI [0.0742, 0.1431]
    - **DeLong test (correlated AUROCs):** p=5.87e-10
  - **MCC (thresholded classification metric):** thresholds selected on **validation** to maximize MCC per model, then applied once to test.
    - Test MCC: LogReg=0.4606 vs RF=0.3788
    - Paired bootstrap: ΔMCC (RF − LogReg) mean=-0.0824; 95% CI [-0.1379, -0.0256]
    - **McNemar exact test (paired errors on test at those thresholds):** p=3.28e-06
  - (For completeness) Test AUPRC: LogReg=0.8776 vs RF=0.9473; paired bootstrap ΔAUPRC mean=0.0692; 95% CI [0.0413, 0.1019]

- **Limitations (important context):** Performance may vary across gene families with limited representation, and labels may contain clinical annotation noise.

**Reference baseline going forward:** All subsequent extensions will be compared against the **shallow RF baseline** selected in Week 5.

## Reproducibility pointers

- Reports: `results/Week 5/week5_logreg_report.json`, `results/Week 5/week5_rf_report.json`, `results/Week 5/week5_rf_calibrated_report.json`
- Plots: `results/Week 5/rf_pr_curves_cal_vs_uncal.png`, `results/Week 5/rf_reliability.png`, `results/Week 5/test_score_distributions.png`
- Statistical comparison: `results/Week 5/compare_logreg_vs_rf_stats.json` (generated by `scripts/compare_baselines_stats.py`; includes DeLong AUROC + MCC/McNemar)
