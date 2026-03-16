"""Week 10 statistical comparison: XGBoost vs RandomForest.

This script trains both models on the same train split of the curated dataset,
then evaluates them on the same held-out test split to compute:
- DeLong test for correlated AUROCs
- Paired bootstrap deltas for AUROC and AUPRC

Example:
  python scripts/compare_xgboost_vs_rf.py \
    --data data/processed/week4_curated_dataset.parquet \
    --rf-report results/baseline_rf_seed37_bootstrap.json \
    --xgb-report results/xgboost_train_eval_report.json \
    --out-json results/xgboost_vs_rf_statistical_comparison.json \
    --plot-deltas results/xgboost_vs_rf_bootstrap_deltas.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import xgboost as xgb


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Statistical comparison of RF vs XGBoost, but it shares some utilities with the EDA script (e.g. path resolution, confidence computation).
def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _safe_metric(metric_fn: Callable[..., Any], y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(metric_fn(y_true, y_score))

# Helper to compute confidence of predictions based on predicted probabilities and binary predictions
def _paired_bootstrap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: Callable[..., Any],
    rng: np.random.Generator,
    iters: int,
    alpha: float,
) -> dict[str, Any]:
    """Paired bootstrap CI + p-value for delta = metric_b - metric_a."""
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = y_true.shape[0]

    base_a = _safe_metric(metric_fn, y_true, score_a)
    base_b = _safe_metric(metric_fn, y_true, score_b)
    base_delta = None if (base_a is None or base_b is None) else float(base_b - base_a)

    deltas: list[float] = []
    skipped = 0
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        a = _safe_metric(metric_fn, y_true[idx], score_a[idx])
        b = _safe_metric(metric_fn, y_true[idx], score_b[idx])
        if a is None or b is None:
            skipped += 1
            continue
        deltas.append(float(b - a))

    if not deltas:
        return {
            "base_a": base_a,
            "base_b": base_b,
            "base_delta": base_delta,
            "iters": int(iters),
            "skipped": int(skipped),
            "delta_mean": None,
            "delta_ci_low": None,
            "delta_ci_high": None,
            "p_two_sided": None,
            "deltas": [],
        }

    deltas_np = np.asarray(deltas)
    low = float(np.quantile(deltas_np, alpha / 2))
    high = float(np.quantile(deltas_np, 1 - alpha / 2))
    delta_mean = float(deltas_np.mean())

    p_le = float((deltas_np <= 0).mean())
    p_ge = float((deltas_np >= 0).mean())
    p_two_sided = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "base_a": base_a,
        "base_b": base_b,
        "base_delta": base_delta,
        "iters": int(iters),
        "skipped": int(skipped),
        "delta_mean": delta_mean,
        "delta_ci_low": low,
        "delta_ci_high": high,
        "p_two_sided": p_two_sided,
        "deltas": deltas,
    }

# DeLong test for correlated AUROCs, adapted from
def _delong_auc_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, Any]:
    """DeLong test for two correlated ROC AUCs (H0: auc_a == auc_b)."""
    y_true = np.asarray(y_true).astype(int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if np.unique(y_true).size < 2:
        return {"auc_a": None, "auc_b": None, "z": None, "p_two_sided": None}

    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    sa = scores_a[order]
    sb = scores_b[order]

    m = int(np.sum(y_sorted == 1))
    n = int(np.sum(y_sorted == 0))
    if m == 0 or n == 0:
        return {"auc_a": None, "auc_b": None, "z": None, "p_two_sided": None}

    def _midrank(x: np.ndarray) -> np.ndarray:
        idx = np.argsort(x)
        x_sorted = x[idx]
        r = np.empty_like(x_sorted, dtype=float)
        i = 0
        while i < x_sorted.size:
            j = i
            while j < x_sorted.size and x_sorted[j] == x_sorted[i]:
                j += 1
            r[i:j] = 0.5 * ((i + 1) + j)
            i = j
        out = np.empty_like(r)
        out[idx] = r
        return out

    def _fast_delong(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = scores.shape[0]
        tx = np.vstack([_midrank(scores[i, :m]) for i in range(k)])
        ty = np.vstack([_midrank(scores[i, m:]) for i in range(k)])
        tz = np.vstack([_midrank(scores[i, :]) for i in range(k)])
        aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
        v01 = (tz[:, :m] - tx) / n
        v10 = 1.0 - (tz[:, m:] - ty) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        s = sx / m + sy / n
        return aucs, s

    aucs, s = _fast_delong(np.vstack([sa, sb]))
    auc_a = float(aucs[0])
    auc_b = float(aucs[1])
    var = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z": None, "p_two_sided": None}

    z = float((auc_a - auc_b) / math.sqrt(var))
    try:
        from scipy.stats import norm  # type: ignore

        p_two = float(2.0 * norm.sf(abs(z)))
    except Exception:
        p_two = float(math.erfc(abs(z) / math.sqrt(2.0)))

    return {"auc_a": auc_a, "auc_b": auc_b, "z": z, "p_two_sided": p_two}


def _plot_delta_histograms(auroc_deltas: list[float], auprc_deltas: list[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, values, title, xlab in [
        (axes[0], auroc_deltas, "Bootstrap ΔAUROC (XGB - RF)", "ΔAUROC"),
        (axes[1], auprc_deltas, "Bootstrap ΔAUPRC (XGB - RF)", "ΔAUPRC"),
    ]:
        arr = np.asarray(values, dtype=float)
        ax.hist(arr, bins=30, alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.axvline(float(np.mean(arr)), linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# The main function orchestrates the loading of data, training of models, evaluation, and reporting.
def main() -> int:
    parser = argparse.ArgumentParser(description="Week 10: DeLong + paired bootstrap comparison of XGBoost vs RF")
    parser.add_argument("--data", default="data/processed/week4_curated_dataset.parquet")
    parser.add_argument("--rf-report", default="results/baseline_rf_seed37_bootstrap.json")
    parser.add_argument("--xgb-report", default="results/xgboost_train_eval_report.json")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-json", default="results/xgboost_vs_rf_statistical_comparison.json")
    parser.add_argument("--plot-deltas", default="results/xgboost_vs_rf_bootstrap_deltas.png")
    args = parser.parse_args()

    data_path = _resolve_path(args.data)
    rf_report_path = _resolve_path(args.rf_report)
    xgb_report_path = _resolve_path(args.xgb_report)
    out_json = _resolve_path(args.out_json)
    plot_path = _resolve_path(args.plot_deltas)

    for p in [data_path, rf_report_path, xgb_report_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_parquet(data_path)
    required = {"embedding", "label", "split"}
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
    rf_score = rf_model.predict_proba(x_test)[:, 1]

    # Train XGB
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
    xgb_score = xgb_model.predict_proba(x_test)[:, 1]

    # Direct metrics
    rf_auroc = _safe_metric(roc_auc_score, y_test, rf_score)
    xgb_auroc = _safe_metric(roc_auc_score, y_test, xgb_score)
    rf_auprc = _safe_metric(average_precision_score, y_test, rf_score)
    xgb_auprc = _safe_metric(average_precision_score, y_test, xgb_score)

    # DeLong and bootstrap
    delong = _delong_auc_test(y_test, rf_score, xgb_score)
    rng = np.random.default_rng(args.bootstrap_seed)
    auroc_delta = _paired_bootstrap_delta(
        y_true=y_test,
        score_a=rf_score,
        score_b=xgb_score,
        metric_fn=roc_auc_score,
        rng=rng,
        iters=args.bootstrap_iters,
        alpha=args.alpha,
    )
    auprc_delta = _paired_bootstrap_delta(
        y_true=y_test,
        score_a=rf_score,
        score_b=xgb_score,
        metric_fn=average_precision_score,
        rng=rng,
        iters=args.bootstrap_iters,
        alpha=args.alpha,
    )

    if plot_path is not None:
        _plot_delta_histograms(auroc_delta["deltas"], auprc_delta["deltas"], plot_path)

    report = {
        "data": str(data_path),
        "models": {
            "rf": {
                "name": "RandomForest",
                "seed": rf_seed,
                "params": rf_params,
                "test_auroc": rf_auroc,
                "test_auprc": rf_auprc,
            },
            "xgb": {
                "name": "XGBoost",
                "seed": xgb_seed,
                "params": xgb_params,
                "test_auroc": xgb_auroc,
                "test_auprc": xgb_auprc,
            },
        },
        "deltas": {
            "xgb_minus_rf_auroc": None if (rf_auroc is None or xgb_auroc is None) else float(xgb_auroc - rf_auroc),
            "xgb_minus_rf_auprc": None if (rf_auprc is None or xgb_auprc is None) else float(xgb_auprc - rf_auprc),
        },
        "delong_auc_test": delong,
        "paired_bootstrap": {
            "auroc": auroc_delta,
            "auprc": auprc_delta,
            "iters": int(args.bootstrap_iters),
            "alpha": float(args.alpha),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "plots": {
            "bootstrap_deltas": str(plot_path) if plot_path is not None else None,
        },
        "interpretation": {
            "delong_significant_at_0_05": (
                None if delong.get("p_two_sided") is None else bool(delong["p_two_sided"] < 0.05)
            ),
            "auroc_delta_ci_includes_zero": (
                None
                if auroc_delta.get("delta_ci_low") is None
                else bool(auroc_delta["delta_ci_low"] <= 0.0 <= auroc_delta["delta_ci_high"])
            ),
            "auprc_delta_ci_includes_zero": (
                None
                if auprc_delta.get("delta_ci_low") is None
                else bool(auprc_delta["delta_ci_low"] <= 0.0 <= auprc_delta["delta_ci_high"])
            ),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    print("Saved:", out_json)
    if plot_path is not None:
        print("Saved:", plot_path)
    print("RF test AUROC/AUPRC:", rf_auroc, rf_auprc)
    print("XGB test AUROC/AUPRC:", xgb_auroc, xgb_auprc)
    print("DeLong p-value:", delong.get("p_two_sided"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
