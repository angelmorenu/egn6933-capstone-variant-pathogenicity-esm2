Scripts are intended to be runnable entrypoints (e.g., split creation, training, scoring).

## Week 9-10: XGBoost Gradient Boosting Implementation

### New Script: `xgboost_train_eval.py`

**Purpose**: Train XGBoost model with stratified k-fold cross-validation and Bayesian hyperparameter search (per Dylan's guidance).

**Methodology**:
- Stratified 5-fold CV on training set (maintains class distribution)
- Bayesian hyperparameter search (Optuna TPE sampler)
- Class imbalance handling: `scale_pos_weight = n_benign / n_pathogenic`
- Platt sigmoid calibration (fit on validation, evaluate on test)
- Gene-disjoint evaluation protocol (consistent with Week 8 baselines)

**Usage**:
```bash
# Default: 50 trials, Platt calibration, full reports
python scripts/xgboost_train_eval.py

# Custom configuration
python scripts/xgboost_train_eval.py \
  --n-trials 100 \
  --cv-folds 5 \
  --calibration platt \
  --out-json results/xgboost_train_eval_report.json \
  --plot-pr results/xgboost_pr_curves.png \
  --plot-roc results/xgboost_roc_curves.png

# Quick validation
python scripts/xgboost_train_eval.py --n-trials 20
```

**Output**:
- `results/xgboost_train_eval_report.json` — Comprehensive metrics (CV, test, calibration, hyperparameters)
- `results/xgboost_pr_curves.png` — Precision-Recall curves (optional)
- `results/xgboost_roc_curves.png` — ROC curves (optional)

**Hyperparameter Ranges**:
- `max_depth: [4, 5, 6]` — shallow trees for 1280+ dimensional embeddings
- `min_child_weight: [1, 5]` — regularization for minority class
- `learning_rate: [0.01, 0.1]` — conservative learning
- `lambda (L2): [0.1, 10.0]` — strong L2 regularization
- `subsample, colsample_bytree: [0.7, 1.0]` — stochastic sampling

**Key Decisions (Dylan's Feedback)**:
- ✅ Stratified k-fold CV (not threshold tuning)
- ✅ Bayesian search (not grid search)
- ✅ Class imbalance via `scale_pos_weight` (not resampling)
- ✅ Single model first (ensemble deferred to Week 11 if needed)
- ✅ Maintains gene-disjoint evaluation protocol

**Results (Week 9, 50 trials)**:
- Best CV AUROC: 0.9738 (strong generalization)
- Test AUROC: 0.9265 (comparable to RF baseline 0.9299)
- Test AUPRC: 0.9437 (comparable to RF baseline 0.9473)
- Best hyperparameters: max_depth=6, learning_rate=0.0829, lambda=0.8373, subsample=0.7691, colsample_bytree=0.7395
- Conclusion: XGBoost and RandomForest achieve comparable performance; gradient boosting offers no measurable advantage on this dataset/embedding space. RF remains the reference baseline.