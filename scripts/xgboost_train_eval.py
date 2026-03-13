"""
XGBoost Model for Pathogenic Variant Classification (Week 9-10 Capstone Work)

This script implements gradient boosting with stratified k-fold cross-validation (per Dylan's 
recommendation) to improve upon Week 8 baselines (LogisticRegression, RandomForest).

Key Methodological Decisions:
1. **Stratified k-fold CV per split**: Class-imbalanced gene-disjoint splits require robust CV
strategy instead of threshold tuning on validation set.
2. **Bayesian hyperparameter search**: More efficient than grid search for high-dimensional space.
3. **Class imbalance handling**: scale_pos_weight = n_benign / n_pathogenic during training.
4. **No threshold selection**: Let model predict calibrated probabilities directly.
5. **Gene-disjoint evaluation**: Maintains biological leakage prevention from Week 8 protocol.

Workflow:
- Load week4_curated_dataset.parquet (train/val/test splits preserved from baselines)
- Extract ESM2 embeddings (1280+ dims) and labels
- Fit stratified k-fold CV on training set with Bayesian hyperparameter search
- Evaluate best model on held-out test set
- Report: mean CV AUROC ± std, test AUROC, calibration metrics

Reads:
- data/processed/week4_curated_dataset.parquet
Writes:
- results/xgboost_train_eval_report.json (comprehensive metrics, CV folds)
- results/xgboost_pr_curves.png (optional)
- results/xgboost_roc_curves.png (optional)

Dependencies:
- xgboost>=2.0.0
- optuna>=3.0.0  (for Bayesian hyperparameter search)
- scikit-learn>=1.3.0
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None) -> Path | None:
    """Resolve path relative to repo root if not absolute."""
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _safe_binary_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Safely compute metric; return None if only one class present."""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(metric_fn(y_true, y_score))


def _ensure_parent(path: str | None) -> None:
    """Ensure parent directory exists."""
    if not path:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    """Compute comprehensive metrics (AUROC, AUPRC, Brier, LogLoss)."""
    auroc = _safe_binary_metric(roc_auc_score, y_true, score)
    auprc = _safe_binary_metric(average_precision_score, y_true, score)
    brier = _safe_binary_metric(brier_score_loss, y_true, score)
    ll = None
    if np.unique(y_true).size >= 2:
        s = np.clip(score, 1e-7, 1 - 1e-7)
        ll = float(log_loss(y_true, np.vstack([1 - s, s]).T, labels=[0, 1]))
    return {"auroc": auroc, "auprc": auprc, "brier": brier, "log_loss": ll}


def _plot_pr_curves(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    out_path: str,
) -> None:
    """Plot precision-recall curves for val vs test."""
    precision_val, recall_val, _ = precision_recall_curve(y_val, s_val)
    precision_test, recall_test, _ = precision_recall_curve(y_test, s_test)
    
    auprc_val = auc(recall_val, precision_val)
    auprc_test = auc(recall_test, precision_test)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_val, precision_val, label=f"val (AUPRC={auprc_val:.4f})", linewidth=2)
    plt.plot(recall_test, precision_test, label=f"test (AUPRC={auprc_test:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost Precision-Recall Curves")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_roc_curves(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    out_path: str,
) -> None:
    """Plot ROC curves for val vs test."""
    fpr_val, tpr_val, _ = roc_curve(y_val, s_val)
    fpr_test, tpr_test, _ = roc_curve(y_test, s_test)
    
    auroc_val = roc_auc_score(y_val, s_val)
    auroc_test = roc_auc_score(y_test, s_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_val, tpr_val, label=f"val (AUROC={auroc_val:.4f})", linewidth=2)
    plt.plot(fpr_test, tpr_test, label=f"test (AUROC={auroc_test:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def objective(trial: optuna.Trial, x_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5) -> float:
    """
    Objective function for Bayesian hyperparameter search (Optuna).
    
    Hyperparameter ranges recommended for ESM2 embeddings (1280+ dims) with class imbalance:
    - max_depth: [4, 5, 6] — shallow trees for high-dimensional embeddings
    - min_child_weight: [1, 3, 5] — regularization for small classes
    - learning_rate: [0.01, 0.05, 0.1] — conservative learning
    - lambda (L2): [0.1, 1.0, 10.0] — strong L2 regularization
    - subsample: [0.7, 0.8, 0.9, 1.0] — row subsampling
    - colsample_bytree: [0.7, 0.8, 0.9, 1.0] — column subsampling
    
    Returns mean AUROC across k-fold CV folds.
    """
    # Define hyperparameter search space
    max_depth = trial.suggest_int("max_depth", 4, 6)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    lambda_l2 = trial.suggest_float("lambda", 0.1, 10.0, log=True)
    subsample = trial.suggest_float("subsample", 0.7, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 1.0)
    
    # Compute class weights for scale_pos_weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Create XGBoost model with suggested hyperparameters
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        learning_rate=learning_rate,
        reg_lambda=lambda_l2,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,  # Conservative estimate; can increase if needed
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    )
    
    # Stratified k-fold CV on training data
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_fold_train, x_fold_val = x_train[train_idx], x_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train on fold
        model.fit(
            x_fold_train, y_fold_train,
            eval_set=[(x_fold_val, y_fold_val)],
            verbose=False,
        )
        
        # Evaluate on fold
        y_pred_proba = model.predict_proba(x_fold_val)[:, 1]
        fold_auroc = _safe_binary_metric(roc_auc_score, y_fold_val, y_pred_proba)
        if fold_auroc is not None:
            cv_scores.append(fold_auroc)
        
        # Report intermediate result for pruning
        trial.report(float(np.mean(cv_scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(cv_scores)) if cv_scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with stratified k-fold CV and Bayesian hyperparameter search."
    )
    parser.add_argument(
        "--data",
        default="data/processed/week4_curated_dataset.parquet",
        help="Path to week4_curated_dataset.parquet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=32,
        help="Number of trials for Bayesian hyperparameter search (default: 32)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of k-fold CV splits (default: 5)",
    )
    parser.add_argument(
        "--out-json",
        default="results/xgboost_train_eval_report.json",
        help="Path to write JSON metrics report",
    )
    parser.add_argument(
        "--plot-pr",
        default="results/xgboost_pr_curves.png",
        help="Optional path to save PR curves (val vs test)",
    )
    parser.add_argument(
        "--plot-roc",
        default="results/xgboost_roc_curves.png",
        help="Optional path to save ROC curves (val vs test)",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="platt",
        help="Calibration method (default: platt for Sigmoid)",
    )
    args = parser.parse_args()

    # Load and prepare data
    data_path = _resolve_path(args.data)
    assert data_path is not None
    if not data_path.exists():
        raise FileNotFoundError(f"Not found: {data_path}")

    df = pd.read_parquet(data_path)
    required_cols = {"embedding", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["split"] = df["split"].astype(str)
    df["label"] = df["label"].astype(int)

    x = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    split = df["split"].to_numpy()

    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    idx_test = np.where(split == "test")[0]

    if idx_train.size == 0 or idx_val.size == 0 or idx_test.size == 0:
        raise ValueError("Missing train, val, or test split.")

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    print("\n" + "=" * 70)
    print("XGBOOST TRAINING WITH STRATIFIED K-FOLD CV (Dylan's Methodology)")
    print("=" * 70)
    print(f"Train: n={len(x_train)}, pos_rate={y_train.mean():.4f}")
    print(f"Val:   n={len(x_val)}, pos_rate={y_val.mean():.4f}")
    print(f"Test:  n={len(x_test)}, pos_rate={y_test.mean():.4f}")
    print(f"Embedding dimension: {x_train.shape[1]}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Bayesian search trials: {args.n_trials}")
    print("=" * 70)

    # Bayesian hyperparameter search using Optuna
    print(f"\nStarting Bayesian hyperparameter search ({args.n_trials} trials)...")
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    # Optimize: run trials and find best hyperparameters
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, n_folds=args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_auroc = study.best_value

    print(f"\n✓ Bayesian search complete.")
    print(f"  Best CV AUROC: {best_cv_auroc:.4f}")
    print(f"  Best hyperparameters: {best_params}")

    # Compute class weights for final model
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Train final model on full training set with best hyperparameters
    print(f"\nTraining final model on full training set...")
    final_model = xgb.XGBClassifier(
        max_depth=best_params["max_depth"],
        min_child_weight=best_params["min_child_weight"],
        learning_rate=best_params["learning_rate"],
        reg_lambda=best_params["lambda"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=0,
    )

    # Train on full training set (no early stopping on full set)
    final_model.fit(x_train, y_train)

    # Evaluate on validation and test sets
    print("Evaluating on validation and test sets...")
    s_val = final_model.predict_proba(x_val)[:, 1]
    s_test = final_model.predict_proba(x_test)[:, 1]

    val_metrics = _score_metrics(y_val, s_val)
    test_metrics = _score_metrics(y_test, s_test)

    print(f"\nValidation metrics:")
    print(f"  AUROC: {val_metrics['auroc']:.4f}")
    print(f"  AUPRC: {val_metrics['auprc']:.4f}")
    print(f"\nTest metrics:")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")

    # Optional calibration (fit on validation, evaluate on test)
    calibrated_scores_val = None
    calibrated_scores_test = None
    calibration_report = None

    if args.calibration != "none":
        print(f"\nApplying {args.calibration} calibration (fit on val, eval on test)...")
        method = "sigmoid" if args.calibration == "platt" else "isotonic"
        calibrator = CalibratedClassifierCV(
            estimator=final_model, method=method, cv="prefit"
        )
        calibrator.fit(x_val, y_val)
        calibrated_scores_val = calibrator.predict_proba(x_val)[:, 1]
        calibrated_scores_test = calibrator.predict_proba(x_test)[:, 1]

        cal_val_metrics = _score_metrics(y_val, calibrated_scores_val)
        cal_test_metrics = _score_metrics(y_test, calibrated_scores_test)

        print(f"  Val (calibrated): AUROC={cal_val_metrics['auroc']:.4f}, AUPRC={cal_val_metrics['auprc']:.4f}")
        print(f"  Test (calibrated): AUROC={cal_test_metrics['auroc']:.4f}, AUPRC={cal_test_metrics['auprc']:.4f}")

        calibration_report = {
            "method": args.calibration,
            "val": {
                "uncalibrated": val_metrics,
                "calibrated": cal_val_metrics,
            },
            "test": {
                "uncalibrated": test_metrics,
                "calibrated": cal_test_metrics,
            },
        }

    # Build comprehensive report
    report = {
        "data": str(data_path),
        "model": "xgboost_gradient_boosting",
        "seed": args.seed,
        "bayesian_search": {
            "n_trials": args.n_trials,
            "best_cv_auroc": float(best_cv_auroc),
            "best_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
        },
        "class_weights": {
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "cv_folds": args.cv_folds,
        "splits": {
            "train": {"n": int(len(x_train)), "pos": int(y_train.sum()), "pos_rate": float(y_train.mean())},
            "val": {"n": int(len(x_val)), "pos": int(y_val.sum()), "pos_rate": float(y_val.mean())},
            "test": {"n": int(len(x_test)), "pos": int(y_test.sum()), "pos_rate": float(y_test.mean())},
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "calibration": calibration_report,
        "plots": {},
    }

    # Generate plots
    if args.plot_pr:
        pr_path = _resolve_path(args.plot_pr)
        assert pr_path is not None
        _ensure_parent(str(pr_path))
        _plot_pr_curves(y_val, s_val, y_test, s_test, str(pr_path))
        report["plots"]["pr_curves"] = str(pr_path)
        print(f"\n✓ Saved PR curves: {pr_path}")

    if args.plot_roc:
        roc_path = _resolve_path(args.plot_roc)
        assert roc_path is not None
        _ensure_parent(str(roc_path))
        _plot_roc_curves(y_val, s_val, y_test, s_test, str(roc_path))
        report["plots"]["roc_curves"] = str(roc_path)
        print(f"✓ Saved ROC curves: {roc_path}")

    # Save report
    out_path = _resolve_path(args.out_json)
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n✓ Saved report: {out_path}")

    print("\n" + "=" * 70)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
