# XGBoost Implementation for Capstone (Week 9-10) – Deliverables Summary

## Overview

I stratified cross-validation and class-imbalance handling, created **two complementary deliverables** for implementing gradient boosting in your capstone:

1. **Production Script** (`scripts/xgboost_train_eval.py`)
2. **Interactive Notebook** (`notebooks/04_xgboost_gradient_boosting.ipynb`)

Both implement **identical methodology** but serve different purposes during capstone execution and presentation.

---

## 1. Production Script: `xgboost_train_eval.py`

### Purpose
- **Reproducible training**: Version-controlled, command-line executable
- **Batch processing**: Run multiple hyperparameter configurations without manual intervention
- **CI/CD integration**: Can be called from bash pipelines or cron jobs
- **Documentation**: Comprehensive docstrings and inline comments

### Key Features

**Stratified K-Fold Cross-Validation**
- Splits training set into 5 stratified folds
- Each fold maintains class distribution of training set
- Prevents class imbalance from introducing bias during hyperparameter search
- Reports mean CV AUROC ± std across all folds

**Bayesian Hyperparameter Search (Optuna)**
- TPE sampler: Tree-structured Parzen Estimator for efficient exploration
- MedianPruner: Stops unpromising trials early
- Hyperparameter ranges tailored for ESM2 embeddings (1280+ dims):
  - `max_depth: [4, 5, 6]` — shallow trees for high-dimensional data
  - `min_child_weight: [1, 3, 5]` — regularization for minority class
  - `learning_rate: [0.01, 0.1]` — conservative learning
  - `lambda (L2): [0.1, 10.0]` — strong L2 regularization
  - `subsample, colsample_bytree: [0.7, 1.0]` — stochastic sampling
- Configurable trials (default: 50; recommend 100 for final production)

**Class Imbalance Handling**
```
scale_pos_weight = n_benign / n_pathogenic
```
- Dynamically computed from training set distribution
- XGBoost uses this to weight positive class loss proportionally
- More robust than post-hoc threshold tuning

**Platt Sigmoid Calibration**
- Fit on validation set predictions (sigmoid function)
- Applied to test set for calibrated probability estimates
- Improves probability calibration (Brier score, log loss)
- Does NOT affect ROC-AUC (rank-based metric), but improves reliability

**Metrics Reported**
- k-fold CV AUROC (mean ± std) → generalization estimate
- Test AUROC & AUPRC (uncalibrated and calibrated)
- Calibration quality: Brier score, log loss
- Class weights and sample counts per split

### Usage

```bash
# Basic (all defaults)
python scripts/xgboost_train_eval.py

# Production run with more trials and custom calibration
python scripts/xgboost_train_eval.py \
  --n-trials 100 \
  --cv-folds 5 \
  --calibration platt \
  --out-json results/xgboost_train_eval_report.json \
  --plot-pr results/xgboost_pr_curves.png \
  --plot-roc results/xgboost_roc_curves.png

# Quick validation (fewer trials)
python scripts/xgboost_train_eval.py --n-trials 20
```

### Output
- JSON report: `results/xgboost_train_eval_report.json`
  - Best hyperparameters
  - CV AUROC and test AUROC/AUPRC
  - Calibration metrics
  - Reproducibility info (seed, data path)
- Optional PNG figures:
  - PR curves (val vs test, calibrated)
  - ROC curves (val vs test, calibrated)

---

## 2. Interactive Notebook: `notebooks/04_xgboost_gradient_boosting.ipynb`

### Purpose
- **Exploratory analysis**: Understand data, hyperparameter search, model behavior
- **Capstone presentation**: Narrative walkthrough of methodology
- **Visualization**: Generate publication-quality figures with explanations
- **Debugging**: Inspect predictions, feature importance, calibration quality

### Structure

| Section | Content |
|---------|---------|
| 1. Imports | Libraries, GPU config, seeds |
| 2. Data Loading | Load parquet, verify splits, class distribution |
| 3. Class Weights | Calculate `scale_pos_weight`, analyze imbalance |
| 4. Bayesian Search | Run Optuna with stratified k-fold CV |
| 5. Final Training | Train on full training set with best params |
| 6. Calibration | Fit Platt sigmoid, compare raw vs. calibrated |
| 7. Visualization | PR/ROC curves with calibration comparison |
| 8. Model Comparison | Benchmark vs. LogReg/RF baselines |
| 9. Feature Importance | Top-20 most important ESM2 embedding dimensions |
| 10. Ensemble Strategy | Discussion of stacking/averaging |

### Key Cells to Run

1. **Cell 1-3**: Load data and verify structure
2. **Cell 4**: Run Bayesian search (slow: ~10-20 min for 50 trials on CPU)
3. **Cell 5**: Train final model and get test metrics
4. **Cell 6**: Apply calibration
5. **Cell 7**: View PR/ROC curves
6. **Cell 8**: Compare vs baselines
7. **Cell 9**: Interpret feature importance

### Interactive Features
- Modifiable hyperparameter ranges (try different `max_depth`)
- Adjustable number of Bayesian trials (faster for exploration)
- Baseline comparison table (populate with actual Week 8 results)
- Feature importance visualization

---

## 

✅ **Stratified k-fold CV per split**
- Implemented in `objective()` function
- Uses sklearn's `StratifiedKFold` to maintain class distribution
- Reports mean CV AUROC across all folds

✅ **No threshold tuning**
- XGBoost predicts calibrated probabilities directly
- No MCC-max selection on validation set
- Calibration (Platt) improves probability reliability, not ranking

✅ **Class imbalance handling**
- `scale_pos_weight` dynamically calculated
- XGBoost internally weights positive class loss
- More robust than data resampling for imbalanced splits

✅ **Bayesian search preferred over grid search**
- Optuna TPE sampler efficiently explores hyperparameter space
- Pruning stops unpromising trials early
- Scales better than grid search for high-dimensional space

✅ **Gene-disjoint evaluation maintained**
- Uses existing `week4_curated_dataset.parquet` splits (train/val/test)
- No additional data leakage introduced
- Evaluation protocol consistent with Week 8 baselines

---

## Execution Plan: Week 9-10

### Week 9 (First Half)
1. **Run production script**:
   ```bash
   python scripts/xgboost_train_eval.py \
     --n-trials 50 \
     --out-json results/xgboost_train_eval_report.json
   ```
2. **Inspect results**:
   - Best hyperparameters
   - k-fold CV AUROC (generalization estimate)
   - Test AUROC comparison vs RF baseline
   - If improvement > 1-2%, validate further

### Week 9 (Second Half)
3. **Run interactive notebook**:
   - Execute cells 1-8 to generate visualizations
   - Create performance comparison table
   - Document feature importance findings

### Week 10
4. **Statistical validation**:
   - Run DeLong test comparing XGBoost vs RF AUROC on test set
   - Report confidence intervals and p-values
   - Determine if improvement is statistically significant
5. **Capstone documentation**:
   - Update README with XGBoost methodology
   - Include PR/ROC curves in presentation
   - Highlight key findings (embedding dimensions used, class imbalance impact)

### Week 11+ (Optional)
6. **If XGBoost significantly outperforms RF** (>2%):
   - Use single XGBoost model for final capstone
   - Skip ensemble (Dylan's guidance)
7. **Else if performance is comparable**:
   - Try simple weighted average: 0.3*RF + 0.7*XGBoost
   - Run statistical test on ensemble performance

---

## Dependencies

Install required packages:

```bash
pip install xgboost optuna scikit-learn pandas numpy matplotlib seaborn
```

Versions:
- `xgboost >= 2.0.0`
- `optuna >= 3.0.0`
- `scikit-learn >= 1.3.0`

---

## References

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Calibration (sklearn)**: https://scikit-learn.org/stable/modules/calibration.html
- **Imbalanced Learning**: https://imbalanced-learn.org/ (optional deeper dive)

---

## Next Action

**Choose your starting point**:

1. **If you want to run immediately**:
   ```bash
   python scripts/xgboost_train_eval.py --n-trials 50
   ```
   Takes ~15 min; outputs JSON report and optional figures

2. **If you want to explore interactively first**:
   - Open `notebooks/04_xgboost_gradient_boosting.ipynb`
   - Run cells 1-8 to understand data and hyperparameter search
   - Modify parameters as needed
   - Then run production script for final results


Both deliverables are ready to use and will produce identical results (same random seed, hyperparameter ranges, evaluation protocol). Choose your workflow based on whether you prefer batch processing (script) or interactive exploration (notebook).

