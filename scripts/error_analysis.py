"""Week 10 error analysis for RandomForest vs XGBoost misclassifications.

This script trains RF and XGBoost on the train split of the curated dataset,
evaluates on the same held-out test split, and reports:
- confusion summaries per model
- false positive / false negative variant-level rows
- overlap between model errors
- gene-level error rates
- confidence summary for correct vs incorrect predictions

Example:
  python scripts/error_analysis.py \
    --data data/processed/week4_curated_dataset.parquet \
    --rf-report results/baseline_rf_seed37_bootstrap.json \
    --xgb-report results/xgboost_train_eval_report.json \
    --out-json results/error_analysis_report.json \
    --out-csv results/error_analysis_misclassified_variants.csv \
    --plot-confusion results/confusion_matrix_rf_vs_xgb.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import xgboost as xgb


_REPO_ROOT = Path(__file__).resolve().parents[1]

# This script is focused on error analysis of RF vs XGBoost, but it shares some utilities with the
def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p

# Helper to compute confidence of predictions based on predicted probabilities and binary predictions
def _prediction_confidence(score: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Confidence of chosen class in [0.5, 1.0]."""
    score = np.asarray(score)
    pred = np.asarray(pred)
    return np.where(pred == 1, score, 1.0 - score)

# The rest of the code is focused on loading the dataset, training the models, evaluating predictions, and summarizing errors.
def _confusion_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "error_rate": float((fp + fn) / len(y_true)) if len(y_true) > 0 else 0.0,
    }


def _error_type(y_true: int, pred: int) -> str:
    if y_true == pred:
        return "correct"
    if y_true == 0 and pred == 1:
        return "false_positive"
    return "false_negative"


def _gene_error_summary(df: pd.DataFrame, error_col: str, min_rows: int = 5) -> list[dict[str, Any]]:
    if "GeneSymbol" not in df.columns:
        return []

    gene_stats = (
        df.groupby("GeneSymbol", dropna=False)
        .agg(
            n=(error_col, "size"),
            errors=(error_col, "sum"),
        )
        .reset_index()
    )
    gene_stats["error_rate"] = gene_stats["errors"] / gene_stats["n"]
    gene_stats = gene_stats[gene_stats["n"] >= min_rows].sort_values(
        ["error_rate", "errors", "n"], ascending=[False, False, False]
    )

    return [
        {
            "GeneSymbol": (None if pd.isna(r["GeneSymbol"]) else str(r["GeneSymbol"])),
            "n": int(r["n"]),
            "errors": int(r["errors"]),
            "error_rate": float(r["error_rate"]),
        }
        for _, r in gene_stats.head(15).iterrows()
    ]

# Loading the dataset, training the models, evaluating predictions, and summarizing errors.
def _plot_confusion_side_by_side(cm_rf: np.ndarray, cm_xgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cm, title in [
        (axes[0], cm_rf, "RandomForest"),
        (axes[1], cm_xgb, "XGBoost"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# The main function orchestrates the loading of data, training of models, evaluation, and reporting.
def main() -> int:
    parser = argparse.ArgumentParser(description="Week 10 error analysis for RF vs XGBoost")
    parser.add_argument("--data", default="data/processed/week4_curated_dataset.parquet")
    parser.add_argument("--rf-report", default="results/baseline_rf_seed37_bootstrap.json")
    parser.add_argument("--xgb-report", default="results/xgboost_train_eval_report.json")
    parser.add_argument("--rf-threshold", type=float, default=0.5)
    parser.add_argument("--xgb-threshold", type=float, default=0.5)
    parser.add_argument("--out-json", default="results/error_analysis_report.json")
    parser.add_argument("--out-csv", default="results/error_analysis_misclassified_variants.csv")
    parser.add_argument("--plot-confusion", default="results/confusion_matrix_rf_vs_xgb.png")
    args = parser.parse_args()

    data_path = _resolve_path(args.data)
    rf_report_path = _resolve_path(args.rf_report)
    xgb_report_path = _resolve_path(args.xgb_report)
    out_json = _resolve_path(args.out_json)
    out_csv = _resolve_path(args.out_csv)
    plot_path = _resolve_path(args.plot_confusion)

    assert data_path is not None and rf_report_path is not None and xgb_report_path is not None
    assert out_json is not None and out_csv is not None and plot_path is not None

    for p in [data_path, rf_report_path, xgb_report_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_parquet(data_path)
    required = {"embedding", "label", "split", "chr_pos_ref_alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    x = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    split = df["split"].astype(str).to_numpy()

    idx_train = np.where(split == "train")[0]
    idx_test = np.where(split == "test")[0]
    if idx_train.size == 0 or idx_test.size == 0:
        raise ValueError("Dataset must contain train and test splits")

    x_train, y_train = x[idx_train], y[idx_train]
    x_test, y_test = x[idx_test], y[idx_test]

    test_meta_cols = [c for c in ["chr_pos_ref_alt", "GeneSymbol", "label", "split"] if c in df.columns]
    test_df = df.iloc[idx_test][test_meta_cols].copy().reset_index(drop=True)

    rf_report = json.loads(rf_report_path.read_text())
    rf_params = rf_report.get("model_params", {})
    rf_seed = int(rf_report.get("seed", 0))

    xgb_report = json.loads(xgb_report_path.read_text())
    xgb_params = xgb_report.get("bayesian_search", {}).get("best_params", {})
    xgb_seed = int(xgb_report.get("seed", 42))

    # Train RF
    rf_model = RandomForestClassifier(
        n_estimators=int(rf_params.get("n_estimators", 200)),
        max_depth=int(rf_params.get("max_depth", 4)),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 5)),
        max_features=rf_params.get("max_features", "sqrt"),
        class_weight="balanced",
        random_state=rf_seed,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    rf_prob = rf_model.predict_proba(x_test)[:, 1]
    rf_pred = (rf_prob >= float(args.rf_threshold)).astype(int)

    # Train XGBoost
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    xgb_model = xgb.XGBClassifier(
        max_depth=int(float(xgb_params.get("max_depth", 6))),
        min_child_weight=float(xgb_params.get("min_child_weight", 1.0)),
        learning_rate=float(xgb_params.get("learning_rate", 0.08)),
        reg_lambda=float(xgb_params.get("lambda", 1.0)),
        subsample=float(xgb_params.get("subsample", 0.8)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
        scale_pos_weight=float(scale_pos_weight),
        n_estimators=100,
        random_state=xgb_seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(x_train, y_train)
    xgb_prob = xgb_model.predict_proba(x_test)[:, 1]
    xgb_pred = (xgb_prob >= float(args.xgb_threshold)).astype(int)

    # Build detailed table
    test_df["rf_prob"] = rf_prob
    test_df["rf_pred"] = rf_pred
    test_df["rf_correct"] = (rf_pred == y_test)
    test_df["rf_error_type"] = [
        _error_type(int(yt), int(yp)) for yt, yp in zip(y_test, rf_pred)
    ]
    test_df["rf_confidence"] = _prediction_confidence(rf_prob, rf_pred)

    test_df["xgb_prob"] = xgb_prob
    test_df["xgb_pred"] = xgb_pred
    test_df["xgb_correct"] = (xgb_pred == y_test)
    test_df["xgb_error_type"] = [
        _error_type(int(yt), int(yp)) for yt, yp in zip(y_test, xgb_pred)
    ]
    test_df["xgb_confidence"] = _prediction_confidence(xgb_prob, xgb_pred)

    test_df["both_wrong"] = (~test_df["rf_correct"]) & (~test_df["xgb_correct"])
    test_df["disagree"] = test_df["rf_pred"] != test_df["xgb_pred"]

    rf_conf = _confusion_dict(y_test, rf_pred)
    xgb_conf = _confusion_dict(y_test, xgb_pred)

    # Overlap and comparison
    overlap = {
        "both_wrong_n": int(test_df["both_wrong"].sum()),
        "rf_only_wrong_n": int(((~test_df["rf_correct"]) & (test_df["xgb_correct"])).sum()),
        "xgb_only_wrong_n": int(((test_df["rf_correct"]) & (~test_df["xgb_correct"])).sum()),
        "both_correct_n": int((test_df["rf_correct"] & test_df["xgb_correct"]).sum()),
        "prediction_disagreement_n": int(test_df["disagree"].sum()),
    }

    # Confidence summaries
    def _conf_summary(values: pd.Series) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "q10": float(values.quantile(0.10)),
            "q90": float(values.quantile(0.90)),
        }

    confidence = {
        "rf": {
            "correct": _conf_summary(test_df.loc[test_df["rf_correct"], "rf_confidence"]),
            "incorrect": _conf_summary(test_df.loc[~test_df["rf_correct"], "rf_confidence"]),
        },
        "xgb": {
            "correct": _conf_summary(test_df.loc[test_df["xgb_correct"], "xgb_confidence"]),
            "incorrect": _conf_summary(test_df.loc[~test_df["xgb_correct"], "xgb_confidence"]),
        },
    }

    # Gene summaries
    test_df["rf_error"] = (~test_df["rf_correct"]).astype(int)
    test_df["xgb_error"] = (~test_df["xgb_correct"]).astype(int)
    rf_gene_summary = _gene_error_summary(test_df, "rf_error")
    xgb_gene_summary = _gene_error_summary(test_df, "xgb_error")

    # Misclassified rows (any model wrong)
    mis = test_df[(~test_df["rf_correct"]) | (~test_df["xgb_correct"])].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mis.to_csv(out_csv, index=False)

    # Plot confusion matrices
    cm_rf = confusion_matrix(y_test, rf_pred, labels=[0, 1])
    cm_xgb = confusion_matrix(y_test, xgb_pred, labels=[0, 1])
    _plot_confusion_side_by_side(cm_rf, cm_xgb, plot_path)

    report = {
        "data": str(data_path),
        "thresholds": {"rf": float(args.rf_threshold), "xgb": float(args.xgb_threshold)},
        "test_split": {
            "n": int(len(y_test)),
            "pos": int(np.sum(y_test == 1)),
            "neg": int(np.sum(y_test == 0)),
            "pos_rate": float(np.mean(y_test == 1)),
        },
        "confusion": {
            "rf": rf_conf,
            "xgb": xgb_conf,
        },
        "overlap": overlap,
        "confidence": confidence,
        "error_counts": {
            "rf_false_positive": int((test_df["rf_error_type"] == "false_positive").sum()),
            "rf_false_negative": int((test_df["rf_error_type"] == "false_negative").sum()),
            "xgb_false_positive": int((test_df["xgb_error_type"] == "false_positive").sum()),
            "xgb_false_negative": int((test_df["xgb_error_type"] == "false_negative").sum()),
        },
        "gene_error_summary": {
            "rf_top_error_genes": rf_gene_summary,
            "xgb_top_error_genes": xgb_gene_summary,
        },
        "outputs": {
            "misclassified_variants_csv": str(out_csv),
            "confusion_plot": str(plot_path),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    print("Saved:", out_json)
    print("Saved:", out_csv)
    print("Saved:", plot_path)
    print("RF error rate:", rf_conf["error_rate"])
    print("XGB error rate:", xgb_conf["error_rate"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
