"""Statistical comparison: LogReg vs RandomForest on the Week 4 curated dataset.

Trains both models on the gene-disjoint train split and evaluates them on the
same held-out gene-disjoint test split.

Primary analysis: paired bootstrap on test set for ΔAUROC (RF - LogReg).
Also reports ΔAUPRC for completeness.

This avoids IID assumptions by using paired resampling of *the same test
examples* for both models.

Example:
python scripts/compare_baselines_stats.py \
    --data data/processed/week4_curated_dataset.parquet \
        --out "results/Week 5/compare_logreg_vs_rf_stats.json" \
    --bootstrap-iters 10000 \
    --seed 0 \
    --bootstrap-seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Helpers for path resolution, metric safety, and paired bootstrap testing
def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p

# Ensure the parent directory of the given path exists
def _ensure_parent(path: Path) -> None:
    path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

# Safely compute a metric, returning None if the metric is undefined (e.g., only one class present)
def _safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(metric_fn(y_true, y_score))

# Paired bootstrap to compare two sets of scores on the same test examples, returning CI and p-value for the difference in metrics.
def _paired_bootstrap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn,
    rng: np.random.Generator,
    iters: int,
    alpha: float,
) -> dict[str, Any]:
    """Return paired bootstrap CI + two-sided p-value for delta = metric_b - metric_a."""

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
        }

    deltas_np = np.asarray(deltas)
    low = float(np.quantile(deltas_np, alpha / 2))
    high = float(np.quantile(deltas_np, 1 - alpha / 2))
    delta_mean = float(deltas_np.mean())

    # Approximate two-sided p-value for H0: delta == 0
    p_le = float((deltas_np <= 0).mean())
    p_ge = float((deltas_np >= 0).mean())
    p_two_sided = float(2 * min(p_le, p_ge))
    p_two_sided = min(p_two_sided, 1.0)

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
    }


def _mcc_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= float(threshold)).astype(int)
    return float(matthews_corrcoef(y_true, y_pred))


def _choose_threshold_max_mcc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Pick a threshold that maximizes MCC on the provided set.

    Uses a quantile grid for speed/stability.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        raise ValueError("Empty y_true in threshold selection.")

    # Candidate thresholds: include endpoints plus a modest quantile grid.
    qs = np.unique(np.concatenate(([0.0], np.linspace(0.0, 1.0, 201), [1.0])))
    thr = np.unique(np.quantile(y_score, qs))
    # Also consider the canonical 0.5 if in range.
    thr = np.unique(np.concatenate((thr, np.array([0.5], dtype=float))))

    best_thr = float(thr[0])
    best_mcc = -np.inf
    for t in thr:
        m = _mcc_from_scores(y_true, y_score, float(t))
        if m > best_mcc:
            best_mcc = float(m)
            best_thr = float(t)
    return {"threshold": best_thr, "mcc": best_mcc}


def _paired_bootstrap_delta_mcc(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    rng: np.random.Generator,
    iters: int,
    alpha: float,
) -> dict[str, Any]:
    """Paired bootstrap CI + p-value for ΔMCC = mcc_b - mcc_a on same test examples."""

    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = y_true.shape[0]

    base_a = _mcc_from_scores(y_true, score_a, threshold_a)
    base_b = _mcc_from_scores(y_true, score_b, threshold_b)
    base_delta = float(base_b - base_a)

    deltas: list[float] = []
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        a = _mcc_from_scores(y_true[idx], score_a[idx], threshold_a)
        b = _mcc_from_scores(y_true[idx], score_b[idx], threshold_b)
        deltas.append(float(b - a))

    deltas_np = np.asarray(deltas, dtype=float)
    low = float(np.quantile(deltas_np, alpha / 2))
    high = float(np.quantile(deltas_np, 1 - alpha / 2))
    delta_mean = float(deltas_np.mean())
    p_le = float((deltas_np <= 0).mean())
    p_ge = float((deltas_np >= 0).mean())
    p_two_sided = float(2 * min(p_le, p_ge))
    p_two_sided = min(p_two_sided, 1.0)

    return {
        "base_a": base_a,
        "base_b": base_b,
        "base_delta": base_delta,
        "iters": int(iters),
        "delta_mean": delta_mean,
        "delta_ci_low": low,
        "delta_ci_high": high,
        "p_two_sided": p_two_sided,
    }


def _mcnemar_exact_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar test on paired correctness (two-sided).

    Uses an exact binomial test on the discordant pairs.
    """

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": n, "p_two_sided": 1.0}

    # Prefer scipy if present; otherwise compute exact two-sided via binomial tail.
    try:
        from scipy.stats import binomtest  # type: ignore

        p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Two-sided exact p-value: 2 * P(X <= min(b,c)) under X ~ Bin(n, 0.5)
        k = min(b, c)
        # Compute CDF up to k.
        cdf = 0.0
        for i in range(0, k + 1):
            cdf += math.comb(n, i) * (0.5 ** n)
        p = float(min(1.0, 2.0 * cdf))

    return {"b": b, "c": c, "n_discordant": n, "p_two_sided": p}


def _delong_auc_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, Any]:
    """DeLong test for two correlated ROC AUCs (two-sided p-value).

    Returns z-statistic and p-value for H0: auc_a == auc_b.
    Implementation follows the fast DeLong method (Sun & Xu).
    """

    y_true = np.asarray(y_true).astype(int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if np.unique(y_true).size < 2:
        return {"auc_a": None, "auc_b": None, "z": None, "p_two_sided": None}

    # Order by label: positives first.
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    sa = scores_a[order]
    sb = scores_b[order]

    m = int(np.sum(y_sorted == 1))
    n = int(np.sum(y_sorted == 0))
    if m == 0 or n == 0:
        return {"auc_a": None, "auc_b": None, "z": None, "p_two_sided": None}

    def _midrank(x: np.ndarray) -> np.ndarray:
        # Compute midranks (1-indexed) for ties.
        idx = np.argsort(x)
        x_sorted = x[idx]
        r = np.empty_like(x_sorted, dtype=float)
        i = 0
        while i < x_sorted.size:
            j = i
            while j < x_sorted.size and x_sorted[j] == x_sorted[i]:
                j += 1
            # ranks i..j-1 are (i+1)..j in 1-indexed
            r[i:j] = 0.5 * ((i + 1) + j)
            i = j
        out = np.empty_like(r)
        out[idx] = r
        return out

    def _fast_delong(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # scores shape: (k, m+n) where first m are positives.
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
        return aucs, s, float(m)

    scores_mat = np.vstack([sa, sb])
    aucs, s, _ = _fast_delong(scores_mat)
    auc_a = float(aucs[0])
    auc_b = float(aucs[1])
    var = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z": None, "p_two_sided": None}
    z = float((auc_a - auc_b) / math.sqrt(var))

    try:
        from scipy.stats import norm  # type: ignore

        p = float(2.0 * norm.sf(abs(z)))
    except Exception:
        # Normal survival function via erfc.
        p = float(math.erfc(abs(z) / math.sqrt(2.0)))

    return {"auc_a": auc_a, "auc_b": auc_b, "z": z, "p_two_sided": p}

# Main function to parse arguments, load data, train models, compute metrics, perform paired bootstrap, and write results.
def main() -> int:
    parser = argparse.ArgumentParser(description="Paired statistical comparison: LogReg vs RF.")
    parser.add_argument(
        "--data",
        default="data/processed/week4_curated_dataset.parquet",
        help="Path to week4_curated_dataset.parquet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for model training RNG.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Random seed for bootstrap resampling RNG (defaults to --seed).",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        help="Paired bootstrap iterations on test set.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Alpha for CI (default 0.05 => 95%% CI).",
    )
    parser.add_argument(
        "--out",
        default="results/Week 5/compare_logreg_vs_rf_stats.json",
        help="Where to write the JSON summary.",
    )

    parser.add_argument(
        "--mcc-threshold",
        choices=["val-max-mcc", "fixed-0.5"],
        default="val-max-mcc",
        help="How to choose thresholds for MCC evaluation.",
    )

    # RF params (match baseline defaults)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=4)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5)
    parser.add_argument("--rf-max-features", default="sqrt")

    args = parser.parse_args()

    data_path = _resolve_path(args.data)
    out_path = _resolve_path(args.out)
    _ensure_parent(out_path)

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
    if idx_train.size == 0 or idx_test.size == 0:
        raise ValueError("Expected non-empty train and test splits.")
    if args.mcc_threshold == "val-max-mcc" and idx_val.size == 0:
        raise ValueError("--mcc-threshold=val-max-mcc requires a non-empty val split.")

    model_seed = int(args.seed)
    bootstrap_seed = model_seed if args.bootstrap_seed is None else int(args.bootstrap_seed)
    rng = np.random.default_rng(bootstrap_seed)

    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=model_seed,
                    C=1.0,
                ),
            ),
        ]
    )

    max_features = args.rf_max_features
    if str(max_features).strip() not in {"sqrt", "log2"}:
        try:
            max_features = float(max_features)
        except ValueError as e:
            raise ValueError(f"Invalid --rf-max-features: {args.rf_max_features}") from e

    rf = Pipeline(
        steps=[
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(args.rf_n_estimators),
                    max_depth=int(args.rf_max_depth),
                    min_samples_leaf=int(args.rf_min_samples_leaf),
                    max_features=max_features,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=model_seed,
                ),
            )
        ]
    )

    logreg.fit(x[idx_train], y[idx_train])
    rf.fit(x[idx_train], y[idx_train])

    y_test = y[idx_test]
    s_logreg = logreg.predict_proba(x[idx_test])[:, 1]
    s_rf = rf.predict_proba(x[idx_test])[:, 1]

    # Validation scores (for threshold selection)
    if idx_val.size:
        y_val = y[idx_val]
        s_logreg_val = logreg.predict_proba(x[idx_val])[:, 1]
        s_rf_val = rf.predict_proba(x[idx_val])[:, 1]
    else:
        y_val = None
        s_logreg_val = None
        s_rf_val = None

    auroc_logreg = _safe_metric(roc_auc_score, y_test, s_logreg)
    auroc_rf = _safe_metric(roc_auc_score, y_test, s_rf)
    auprc_logreg = _safe_metric(average_precision_score, y_test, s_logreg)
    auprc_rf = _safe_metric(average_precision_score, y_test, s_rf)

    stats_auroc = _paired_bootstrap_delta(
        y_true=y_test,
        score_a=s_logreg,
        score_b=s_rf,
        metric_fn=roc_auc_score,
        rng=rng,
        iters=int(args.bootstrap_iters),
        alpha=float(args.bootstrap_alpha),
    )
    stats_auprc = _paired_bootstrap_delta(
        y_true=y_test,
        score_a=s_logreg,
        score_b=s_rf,
        metric_fn=average_precision_score,
        rng=rng,
        iters=int(args.bootstrap_iters),
        alpha=float(args.bootstrap_alpha),
    )

    # Formal AUROC test: DeLong for correlated AUCs
    delong = _delong_auc_test(y_test, s_logreg, s_rf)

    # MCC comparison (threshold chosen on val, then evaluated on test)
    if args.mcc_threshold == "fixed-0.5":
        thr_logreg = 0.5
        thr_rf = 0.5
        val_choice = None
    else:
        assert y_val is not None and s_logreg_val is not None and s_rf_val is not None
        val_choice = {
            "logreg": _choose_threshold_max_mcc(y_val, s_logreg_val),
            "rf": _choose_threshold_max_mcc(y_val, s_rf_val),
        }
        thr_logreg = float(val_choice["logreg"]["threshold"])
        thr_rf = float(val_choice["rf"]["threshold"])

    mcc_logreg_test = _mcc_from_scores(y_test, s_logreg, thr_logreg)
    mcc_rf_test = _mcc_from_scores(y_test, s_rf, thr_rf)

    pred_logreg = (s_logreg >= thr_logreg).astype(int)
    pred_rf = (s_rf >= thr_rf).astype(int)
    mcnemar = _mcnemar_exact_pvalue(y_test, pred_logreg, pred_rf)
    stats_mcc = _paired_bootstrap_delta_mcc(
        y_true=y_test,
        score_a=s_logreg,
        score_b=s_rf,
        threshold_a=thr_logreg,
        threshold_b=thr_rf,
        rng=rng,
        iters=int(args.bootstrap_iters),
        alpha=float(args.bootstrap_alpha),
    )

    report: dict[str, Any] = {
        "data": str(data_path),
        "seed": {
            "model_seed": int(model_seed),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "bootstrap": {
            "iters": int(args.bootstrap_iters),
            "alpha": float(args.bootstrap_alpha),
            "kind": "paired-resample test examples",
        },
        "models": {
            "logreg": {
                "C": 1.0,
                "class_weight": "balanced",
                "auroc_test": auroc_logreg,
                "auprc_test": auprc_logreg,
                "mcc_test": mcc_logreg_test,
            },
            "rf": {
                "n_estimators": int(args.rf_n_estimators),
                "max_depth": int(args.rf_max_depth),
                "min_samples_leaf": int(args.rf_min_samples_leaf),
                "max_features": args.rf_max_features,
                "class_weight": "balanced",
                "auroc_test": auroc_rf,
                "auprc_test": auprc_rf,
                "mcc_test": mcc_rf_test,
            },
        },
        "tests": {
            "delong_auroc": delong,
            "mcc_threshold": {
                "strategy": args.mcc_threshold,
                "threshold_logreg": float(thr_logreg),
                "threshold_rf": float(thr_rf),
                "val_selected": val_choice,
            },
            "mcnemar_test": mcnemar,
        },
        "delta": {
            "auroc_rf_minus_logreg": stats_auroc,
            "auprc_rf_minus_logreg": stats_auprc,
            "mcc_rf_minus_logreg": stats_mcc,
        },
    }

    out_path.write_text(json.dumps(report, indent=2))

    def fmt(v: float | None) -> str:
        return "NA" if v is None else f"{v:.4f}"

    print(f"test AUROC: logreg={fmt(auroc_logreg)} rf={fmt(auroc_rf)}")
    print(
        "ΔAUROC (RF-LogReg): "
        f"mean={fmt(stats_auroc.get('delta_mean'))} "
        f"CI=[{fmt(stats_auroc.get('delta_ci_low'))},{fmt(stats_auroc.get('delta_ci_high'))}] "
        f"p≈{stats_auroc.get('p_two_sided'):.4g}"
    )
    print(
        "DeLong AUROC test (LogReg vs RF): "
        f"z={delong.get('z'):.3f} p={delong.get('p_two_sided'):.3g}"
        if delong.get("z") is not None
        else "DeLong AUROC test: NA"
    )
    print(f"test MCC (val-chosen thresholds): logreg={mcc_logreg_test:.4f} rf={mcc_rf_test:.4f}")
    print(
        "ΔMCC (RF-LogReg): "
        f"mean={fmt(stats_mcc.get('delta_mean'))} "
        f"CI=[{fmt(stats_mcc.get('delta_ci_low'))},{fmt(stats_mcc.get('delta_ci_high'))}] "
        f"p≈{stats_mcc.get('p_two_sided'):.4g}"
    )
    print(
        "McNemar exact test (thresholded preds on test): "
        f"b={mcnemar.get('b')} c={mcnemar.get('c')} p={mcnemar.get('p_two_sided'):.3g}"
    )
    print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
