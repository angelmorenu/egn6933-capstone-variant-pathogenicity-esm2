# Weeks 5–8 Execution Checklist

This is a concrete Week 5–8 plan aligned to Phase 2 (Feature Engineering & Baselines): **baseline model development + robust evaluation** using the Week 4 curated dataset artifact.

**Primary artifact for Weeks 5–8:** `data/processed/week4_curated_dataset.parquet`
- Required columns: `chr_pos_ref_alt`, `label` (0/1), `split` (train/val/test), `embedding` (vector)

**Primary entrypoint for baselines:** `scripts/baseline_train_eval.py`

---

## Phase 2 (Weeks 5–8) — High-level deliverables (from README)

- [x] Finalize feature set (ESM2 embedding dimensions and QC)
- [ ] Implement optional missing-feature generation and caching (only if needed)
- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest baseline
- [ ] Initial AUROC/AUPRC evaluation

## Week 5 — Reproduce baselines + lock a “baseline conclusion”

- [x] Confirm Week 4 artifacts exist and are internally consistent
  - [x] `data/processed/week4_curated_dataset.parquet` loads cleanly (n=5000; duplicate keys=0)
  - [x] `scripts/week4_eda.py` passes go/no-go checks (GO)
  - [x] Embedding dimension is consistent (ESM2=2560)

- [x] Run Logistic Regression baseline (uncalibrated)
  - [x] Saved: `results/Week 5/week5_logreg_report.json`
  - [x] Test AUROC=0.7663; Test AUPRC=0.7365

- [x] Run Random Forest baseline (uncalibrated)
  - [x] Saved: `results/Week 5/week5_rf_report.json`
  - [x] Test AUROC=0.9306; Test AUPRC=0.9063

- [x] Run calibration comparison (val-fit calibration; test-evaluated)
  - [x] Platt calibration report: `results/Week 5/week5_rf_calibrated_report.json`
  - [x] Plots: `results/Week 5/week5_rf_pr_cal_vs_uncal.png`, `results/Week 5/week5_rf_reliability.png`, `results/Week 5/week5_test_score_distributions.png`
  - [x] Rank metrics unchanged; probability quality improved (test Brier 0.1378→0.1181; test log loss 0.4513→0.3804)

- [x] Week 5 deliverable: “Baseline conclusion” (short, explicit)
  - [x] Winner by **test AUROC (primary metric; per Dr. Fan)**: shallow RF (AUROC=0.9306) > logreg (AUROC=0.7663)
  - [x] Secondary metric (reported for completeness): AUPRC (RF=0.9063; logreg=0.7365)
  - [x] Point estimates recorded above; confidence intervals planned for Week 6
  - [x] Calibration: Platt improves probability reliability (Brier/log loss) without changing AUROC/AUPRC
  - [x] Evaluation protocol: gene-disjoint holdout; test reflects generalization to unseen genes/proteins

**Recommended commands (Week 5):**
```bash
python scripts/week4_eda.py

python scripts/baseline_train_eval.py \
  --data data/processed/week4_curated_dataset.parquet \
  --out-json "results/Week 5/week5_logreg_report.json"

python scripts/baseline_train_eval.py \
  --model rf \
  --rf-max-depth 4 \
  --rf-n-estimators 200 \
  --data data/processed/week4_curated_dataset.parquet \
  --out-json "results/Week 5/week5_rf_report.json"

python scripts/baseline_train_eval.py \
  --model rf \
  --data data/processed/week4_curated_dataset.parquet \
  --calibration platt \
  --plot-pr "results/Week 5/week5_rf_pr_cal_vs_uncal.png" \
  --plot-reliability "results/Week 5/week5_rf_reliability.png" \
  --out-json "results/Week 5/week5_rf_calibrated_report.json"
```

---

## Week 6 — Robustness: split seeds + bootstrap confidence intervals

- [x] Split seed sensitivity (gene-disjoint)
  - [x] Regenerate Week 3 splits with ≥2 different seeds
    - [x] Seed 13 archived: `data/processed/week2_training_table_strict_seed13_splits.parquet` (+ `*_idx.npy`, `*_splits_meta.json`)
    - [x] Seed 37 archived: `data/processed/week2_training_table_strict_seed37_splits.parquet` (+ `*_idx.npy`, `*_splits_meta.json`)
  - [x] Rebuild Week 4 curated Parquet for each seed (separate output filenames)
    - [x] `data/processed/week4_curated_dataset_seed13.parquet` (+ `_meta.json`)
    - [x] `data/processed/week4_curated_dataset_seed37.parquet` (+ `_meta.json`)
  - [x] Rerun baselines on each seed-specific curated dataset
    - [x] `results/baseline_rf_seed13_bootstrap.json`
    - [x] `results/baseline_rf_seed37_bootstrap.json`
  - [x] Summarize variability of test AUROC/AUPRC across seeds
    - [x] `docs/week6_seed_sensitivity.md`

- [x] Bootstrap confidence intervals on the held-out test set
  - [x] For the best baseline (by AUROC, primary metric), compute bootstrapped 95% CIs for AUROC and AUPRC
  - [x] Report number of bootstrap iterations and any skipped resamples (if a resample contains only one class)

**Recommended commands (Week 6):**
```bash
python scripts/make_week3_splits.py \
  --seed 13 \
  --out-prefix data/processed/week2_training_table_strict
python scripts/make_week4_curated_dataset.py
# Archive the outputs to seed-specific filenames:
cp -f data/processed/week4_curated_dataset.parquet data/processed/week4_curated_dataset_seed13.parquet
cp -f data/processed/week4_curated_dataset_meta.json data/processed/week4_curated_dataset_seed13_meta.json

python scripts/make_week3_splits.py \
  --seed 37 \
  --out-prefix data/processed/week2_training_table_strict
python scripts/make_week4_curated_dataset.py
# Archive the outputs to seed-specific filenames:
cp -f data/processed/week4_curated_dataset.parquet data/processed/week4_curated_dataset_seed37.parquet
cp -f data/processed/week4_curated_dataset_meta.json data/processed/week4_curated_dataset_seed37_meta.json

python scripts/baseline_train_eval.py \
  --model rf \
  --rf-max-depth 4 \
  --rf-n-estimators 200 \
  --data data/processed/week4_curated_dataset_seed13.parquet \
  --bootstrap-iters 1000 \
  --out-json results/baseline_rf_seed13_bootstrap.json

python scripts/baseline_train_eval.py \
  --model rf \
  --rf-max-depth 4 \
  --rf-n-estimators 200 \
  --data data/processed/week4_curated_dataset_seed37.parquet \
  --bootstrap-iters 1000 \
  --out-json results/baseline_rf_seed37_bootstrap.json
```

---

## Week 7 — Feature set + hyperparameter selection discipline

(Goal: make baseline selection defensible and reproducible.)

**Status:** ✅ Completed

- [x] Logistic Regression hyperparameter sweep
  - [x] Sweep `C` using documented grid (0.001–100) and select by validation AUROC (primary metric; per Dr. Fan)
  - [x] Confirm scaling is applied (StandardScaler) in the pipeline
  - [x] Selected C=1.0 (balanced regularization); saved in `docs/week7_hyperparameter_selection.md`

- [x] Random Forest hyperparameters (minimal, controlled)
  - [x] Confirm chosen `max_depth=4`, `n_estimators=200` reflect "shallow RF" intent
  - [x] Finalized shallow RF configuration to prevent overfitting

- [x] Confirm class imbalance handling is explicit
  - [x] Record class weighting: Balanced weights applied to both LogReg and RF ✓
  - [x] Documented in `docs/week7_hyperparameter_selection.md`
  - [x] Validated no changes to weighting between Week 6 and Week 7 (seed tests remain valid)

**Recommended commands (Week 7):**
```bash
# LogReg C-sweep with AUROC selection on validation set
python scripts/baseline_train_eval.py \
  --data data/processed/week4_curated_dataset.parquet \
  --c-grid 0.001,0.01,0.1,1,10,100 \
  --select-metric auroc \
  --out-json results/week7_logreg_c_sweep.json

# Optional: RF grid search (keep it small to save time)
python scripts/baseline_train_eval.py \
  --model rf \
  --rf-max-depth 3,4,5 \
  --rf-n-estimators 100,200 \
  --data data/processed/week4_curated_dataset.parquet \
  --select-metric auroc \
  --out-json results/week7_rf_grid_search.json
```

**Recommended commands (Week 7):**
```bash
python scripts/baseline_train_eval.py \
  --data data/processed/week4_curated_dataset.parquet \
  --c-grid 0.01,0.1,1,10,100 \
  --select-metric auprc \
  --out-json results/week7_logreg_c_sweep.json
```

---

## Week 8 — Thresholding + “initial methods/results” write-up readiness

- [ ] Choose and document an operating threshold
  - [ ] Decide the criterion (e.g., maximize F1 on val; or target recall with minimum precision)
  - [ ] Report val-set threshold and test-set performance at that threshold
  - [ ] State that threshold is tuned on val only (no test peeking)

- [ ] Calibration decision
  - [ ] If using calibration, specify method (Platt vs isotonic), fit split (val), and evaluation split (test)
  - [ ] Include reliability plot and Brier score/log loss (if available in your report)

- [ ] Create a single Week 8 summary artifact (for mid-project review readiness)
  - [ ] One table: model → test AUROC/AUPRC (+ CIs if available)
  - [ ] One figure: PR curves (val vs test) for the selected model
  - [ ] One figure: reliability diagram (if using calibration)
  - [ ] 5–8 sentence written summary of findings and limitations

---

## Definition of Done for Weeks 5–8

- [ ] Baselines reproducible from the curated Parquet with saved JSON reports in `results/`
- [ ] Seed robustness demonstrated (≥2 seeds) under gene-disjoint splits
- [ ] Bootstrapped CIs reported for the selected baseline
- [ ] Calibration and thresholding decisions documented (val-tuned; test-reported)
- [ ] Clear baseline conclusion: selected model and why (AUROC-first; per Dr. Fan)
