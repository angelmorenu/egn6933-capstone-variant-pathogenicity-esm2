"""
Annotation (per Dr. Fan's requirement):
All variants used for model training and evaluation must be mapped to and uniquely identified by Dylan's precomputed ESM2 embeddings. This script expects the input dataset to contain a column 'embedding' with a unique, valid embedding vector for every variant, ensuring full traceability and compliance with project requirements.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p

'''Train and evaluate a simple baseline model on week4_curated_dataset.parquet.
Reports AUROC and AUPRC on val/test splits using the `split` column.

Reads:
- week4_curated_dataset.parquet
Writes:
- Optional JSON metrics report

Usage:
python scripts/baseline_train_eval.py \
--data data/processed/week4_curated_dataset.parquet \
--out-json results/baseline_metrics.json
Optional arguments:
--c-grid '0.01,0.1,1,10' : comma-separated C values for LogisticRegression L2 sweep
--select-metric 'auprc' : metric used to select best C on val when --c-grid is provided
--calibration 'platt' : optional calibration fitted on val (Platt=sigmoid) and evaluated on test
--bootstrap-iters 1000 : if >0, compute bootstrap CIs on test for AUROC/AUPRC
--bootstrap-alpha 0.05 : alpha for bootstrap CI (default 0.05 => 95% CI)
--plot-pr results/pr_curve.png : optional path to save PR curves (val vs test)
--plot-scores-test results/test_score_dists.png : optional path to save test score distributions (pos vs neg)
'''

# Safe metric computation that returns None if only one class is present 
def _safe_binary_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(metric_fn(y_true, y_score))

# Parse a comma-separated list of floats 
def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return []
    return [float(p) for p in parts]

# Bootstrap CI computation for binary metrics 
def _bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_score: np.ndarray,
    rng: np.random.Generator,
    iters: int,
    alpha: float,
) -> dict[str, float | int | None]:
    if iters <= 0:
        return {"iters": 0, "skipped": 0, "low": None, "high": None}

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.shape[0]
    values: list[float] = []
    skipped = 0

    for _ in range(iters):
        idx = rng.integers(0, n, size=n, endpoint=False)
        v = _safe_binary_metric(metric_fn, y_true[idx], y_score[idx])
        if v is None:
            skipped += 1
            continue
        values.append(v)

    if not values:
        return {"iters": iters, "skipped": skipped, "low": None, "high": None}

    low = float(np.quantile(values, alpha / 2))
    high = float(np.quantile(values, 1 - alpha / 2))
    return {"iters": iters, "skipped": skipped, "low": low, "high": high}

# Generator to parse VCF lines into (chrom, info_dict) 
def _log_loss_from_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    s = np.clip(y_score, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, np.vstack([1 - s, s]).T, labels=[0, 1]))

# Helper to ensure parent directory exists for a given path
def _ensure_parent(path: str | None) -> None:
    if not path:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

# Plotting functions 
def _plot_pr_curves(
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    out_path: str,
) -> None:
    precision_val, recall_val, _ = precision_recall_curve(y_val, s_val)
    precision_test, recall_test, _ = precision_recall_curve(y_test, s_test)

    plt.figure(figsize=(7, 5))
    plt.plot(recall_val, precision_val, label="val")
    plt.plot(recall_test, precision_test, label="test")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pr_curves_with_calibration(
    y_val: np.ndarray,
    s_val_uncal: np.ndarray,
    s_val_cal: np.ndarray,
    y_test: np.ndarray,
    s_test_uncal: np.ndarray,
    s_test_cal: np.ndarray,
    out_path: str,
) -> None:
    pv_u, rv_u, _ = precision_recall_curve(y_val, s_val_uncal)
    pv_c, rv_c, _ = precision_recall_curve(y_val, s_val_cal)
    pt_u, rt_u, _ = precision_recall_curve(y_test, s_test_uncal)
    pt_c, rt_c, _ = precision_recall_curve(y_test, s_test_cal)

    plt.figure(figsize=(7, 5))
    plt.plot(rv_u, pv_u, label="val (uncal)")
    plt.plot(rv_c, pv_c, label="val (cal)")
    plt.plot(rt_u, pt_u, label="test (uncal)")
    plt.plot(rt_c, pt_c, label="test (cal)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Calibrated vs Uncalibrated)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_reliability_diagram(
    y_val: np.ndarray,
    s_val_uncal: np.ndarray,
    s_val_cal: np.ndarray,
    y_test: np.ndarray,
    s_test_uncal: np.ndarray,
    s_test_cal: np.ndarray,
    out_path: str,
    n_bins: int,
) -> None:
    def curve(y_true: np.ndarray, s: np.ndarray):
        prob_true, prob_pred = calibration_curve(y_true, s, n_bins=n_bins, strategy="uniform")
        return prob_pred, prob_true

    v_pred_u, v_true_u = curve(y_val, s_val_uncal)
    v_pred_c, v_true_c = curve(y_val, s_val_cal)
    t_pred_u, t_true_u = curve(y_test, s_test_uncal)
    t_pred_c, t_true_c = curve(y_test, s_test_cal)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, title, pred_u, true_u, pred_c, true_c in [
        (axes[0], "Validation", v_pred_u, v_true_u, v_pred_c, v_true_c),
        (axes[1], "Test", t_pred_u, t_true_u, t_pred_c, t_true_c),
    ]:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        ax.plot(pred_u, true_u, marker="o", label="uncal")
        ax.plot(pred_c, true_c, marker="o", label="cal")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(f"Reliability: {title}")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Fraction of positives")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# Plot score distributions for positives vs negatives
def _plot_score_distributions(
    y: np.ndarray,
    score: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    y = np.asarray(y)
    score = np.asarray(score)
    pos = score[y == 1]
    neg = score[y == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=30, alpha=0.6, density=True, label="neg (0)")
    plt.hist(pos, bins=30, alpha=0.6, density=True, label="pos (1)")
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Main function to run the script 
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple baseline model on week4_curated_dataset.parquet and report AUROC/AUPRC "
            "on val/test splits using the `split` column."
        )
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf"],
        default="logreg",
        help="Baseline model to train: logistic regression (logreg) or shallow random forest (rf).",
    )
    parser.add_argument(
        "--data",
        default="data/processed/week4_curated_dataset.parquet",
        help="Path to week4_curated_dataset.parquet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used by the model when applicable)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write a JSON metrics report",
    )
    parser.add_argument(
        "--c-grid",
        default=None,
        help=(
            "Optional comma-separated C values for LogisticRegression L2 sweep (e.g., '0.01,0.1,1,10'). "
            "If provided, selects best C by --select-metric on val."
        ),
    )
    parser.add_argument(
        "--select-metric",
        choices=["auprc", "auroc"],
        default="auprc",
        help="Metric used to select best C on val when --c-grid is provided.",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Optional calibration fitted on val (Platt=sigmoid) and evaluated on test.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="If >0, compute bootstrap CIs on test for AUROC/AUPRC.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Alpha for bootstrap CI (default 0.05 => 95% CI).",
    )
    parser.add_argument(
        "--plot-pr",
        default=None,
        help="Optional path to save PR curves (val vs test).",
    )
    parser.add_argument(
        "--plot-scores-test",
        default=None,
        help="Optional path to save test score distributions (pos vs neg).",
    )
    parser.add_argument(
        "--plot-reliability",
        default=None,
        help="Optional path to save reliability diagram (requires --calibration != none).",
    )
    parser.add_argument(
        "--reliability-bins",
        type=int,
        default=10,
        help="Number of bins for reliability diagram (default: 10).",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=200,
        help="Random Forest: number of trees (only used when --model rf).",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=4,
        help="Random Forest: max tree depth (kept shallow for controlled nonlinearity).",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=5,
        help="Random Forest: min samples per leaf (regularization).",
    )
    parser.add_argument(
        "--rf-max-features",
        default="sqrt",
        help="Random Forest: max_features (e.g., 'sqrt', 'log2', or a float in (0,1]).",
    )
    args = parser.parse_args()

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

    if idx_train.size == 0:
        raise ValueError("No rows found with split == 'train'.")
    if idx_val.size == 0:
        raise ValueError("No rows found with split == 'val'.")
    if idx_test.size == 0:
        raise ValueError("No rows found with split == 'test'.")

    rng = np.random.default_rng(args.seed)

    def _parse_rf_max_features(v: str) -> float | Literal["sqrt", "log2"]:
        vv = str(v).strip()
        if vv == "sqrt":
            return "sqrt"
        if vv == "log2":
            return "log2"
        try:
            f = float(vv)
        except ValueError as e:
            raise ValueError(f"Invalid --rf-max-features: {v}") from e
        if not (0 < f <= 1):
            raise ValueError("--rf-max-features as float must be in (0, 1].")
        return f

    def make_logreg_model(c: float = 1.0) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=args.seed,
                        C=float(c),
                    ),
                ),
            ]
        )

    def make_rf_model() -> Pipeline:
        max_features = _parse_rf_max_features(args.rf_max_features)
        return Pipeline(
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
                        random_state=args.seed,
                    ),
                )
            ]
        )

    chosen_c = 1.0
    sweep_rows: list[dict[str, Any]] = []

    if args.model != "logreg" and args.c_grid:
        raise ValueError("--c-grid is only applicable for --model logreg.")

    if args.model == "logreg" and args.c_grid:
        c_values = _parse_float_list(args.c_grid)
        if not c_values:
            raise ValueError("--c-grid was provided but no valid values were parsed.")

        best_metric = -np.inf
        best_c = None

        for c in c_values:
            m = make_logreg_model(c)
            m.fit(x[idx_train], y[idx_train])
            yv = y[idx_val]
            sv = m.predict_proba(x[idx_val])[:, 1]
            auroc = _safe_binary_metric(roc_auc_score, yv, sv)
            auprc = _safe_binary_metric(average_precision_score, yv, sv)
            row = {"C": float(c), "val_auroc": auroc, "val_auprc": auprc}
            sweep_rows.append(row)
            score = auprc if args.select_metric == "auprc" else auroc
            if score is not None and score > best_metric:
                best_metric = score
                best_c = float(c)

        if best_c is None:
            raise RuntimeError("C-sweep failed: no valid metric values were produced.")
        chosen_c = best_c
        print(f"Selected C={chosen_c} by val {args.select_metric}.")

    if args.model == "logreg":
        model = make_logreg_model(chosen_c)
        model.fit(x[idx_train], y[idx_train])
        model_name = "logreg_standardized_class_weight_balanced"
        model_params: dict[str, Any] = {"C": float(chosen_c)}
    else:
        model = make_rf_model()
        model.fit(x[idx_train], y[idx_train])
        model_name = "random_forest_shallow_class_weight_balanced"
        model_params = {
            "n_estimators": int(args.rf_n_estimators),
            "max_depth": int(args.rf_max_depth),
            "min_samples_leaf": int(args.rf_min_samples_leaf),
            "max_features": str(args.rf_max_features),
        }

    def eval_split(name: str, idx: np.ndarray, score: np.ndarray | None = None) -> dict:
        y_true = y[idx]
        y_score = score if score is not None else model.predict_proba(x[idx])[:, 1]
        auroc = _safe_binary_metric(roc_auc_score, y_true, y_score)
        auprc = _safe_binary_metric(average_precision_score, y_true, y_score)
        return {
            "n": int(idx.size),
            "pos": int(y_true.sum()),
            "neg": int(idx.size - y_true.sum()),
            "pos_rate": float(y_true.mean()) if idx.size else None,
            "auroc": auroc,
            "auprc": auprc,
        }

    report: dict[str, Any] = {
        "data": str(data_path),
        "model": model_name,
        "model_params": model_params,
        "seed": int(args.seed),
        "chosen_C": float(chosen_c) if args.model == "logreg" else None,
        "c_sweep": sweep_rows if (args.model == "logreg" and sweep_rows) else None,
        "splits": {
            "train": eval_split("train", idx_train),
            "val": eval_split("val", idx_val),
            "test": eval_split("test", idx_test),
        },
        "calibration": None,
        "bootstrap": None,
        "plots": None,
    }

    def _score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
        # log_loss requires both classes and probabilities strictly in (0,1)
        auroc = _safe_binary_metric(roc_auc_score, y_true, score)
        auprc = _safe_binary_metric(average_precision_score, y_true, score)
        brier = _safe_binary_metric(brier_score_loss, y_true, score)
        ll = None
        if np.unique(y_true).size >= 2:
            s = np.clip(score, 1e-7, 1 - 1e-7)
            ll = float(log_loss(y_true, np.vstack([1 - s, s]).T, labels=[0, 1]))
        return {"auroc": auroc, "auprc": auprc, "brier": brier, "log_loss": ll}

    # Optional calibration: fit on val, then evaluate calibrated scores on val/test
    calibrated_scores_val = None
    calibrated_scores_test = None
    if args.calibration != "none":
        method = "sigmoid" if args.calibration == "platt" else "isotonic"
        calibrator = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
        calibrator.fit(x[idx_val], y[idx_val])
        calibrated_scores_val = calibrator.predict_proba(x[idx_val])[:, 1]
        calibrated_scores_test = calibrator.predict_proba(x[idx_test])[:, 1]
        # Record calibration-specific metrics (cal vs uncal) for val/test
        uncal_val = model.predict_proba(x[idx_val])[:, 1]
        uncal_test = model.predict_proba(x[idx_test])[:, 1]

        report["calibration"] = {
            "method": args.calibration,
            "val": eval_split("val", idx_val, score=calibrated_scores_val),
            "test": eval_split("test", idx_test, score=calibrated_scores_test),
            "metrics": {
                "val": {
                    "uncalibrated": _score_metrics(y[idx_val], uncal_val),
                    "calibrated": _score_metrics(y[idx_val], calibrated_scores_val),
                },
                "test": {
                    "uncalibrated": _score_metrics(y[idx_test], uncal_test),
                    "calibrated": _score_metrics(y[idx_test], calibrated_scores_test),
                },
            },
        }

    # Optional bootstrap CIs on test (uses calibrated scores if calibration enabled)
    if args.bootstrap_iters and args.bootstrap_iters > 0:
        test_scores = (
            calibrated_scores_test
            if calibrated_scores_test is not None
            else model.predict_proba(x[idx_test])[:, 1]
        )
        y_test = y[idx_test]
        ci_auroc = _bootstrap_ci(
            roc_auc_score, y_test, test_scores, rng=rng, iters=args.bootstrap_iters, alpha=args.bootstrap_alpha
        )
        ci_auprc = _bootstrap_ci(
            average_precision_score,
            y_test,
            test_scores,
            rng=rng,
            iters=args.bootstrap_iters,
            alpha=args.bootstrap_alpha,
        )

        # Brier does not require both classes, but we keep skip behavior consistent if a degenerate bootstrap sample occurs.
        ci_brier = _bootstrap_ci(
            brier_score_loss,
            y_test,
            test_scores,
            rng=rng,
            iters=args.bootstrap_iters,
            alpha=args.bootstrap_alpha,
        )

        ci_logloss = _bootstrap_ci(
            _log_loss_from_score,
            y_test,
            test_scores,
            rng=rng,
            iters=args.bootstrap_iters,
            alpha=args.bootstrap_alpha,
        )
        report["bootstrap"] = {
            "alpha": float(args.bootstrap_alpha),
            "iters": int(args.bootstrap_iters),
            "score_source": "calibrated" if calibrated_scores_test is not None else "uncalibrated",
            "test": {"auroc": ci_auroc, "auprc": ci_auprc, "brier": ci_brier, "log_loss": ci_logloss},
        }

    # Optional plots
    plots: dict[str, str] = {}
    if args.plot_pr:
        plot_pr_path = _resolve_path(args.plot_pr)
        assert plot_pr_path is not None
        _ensure_parent(str(plot_pr_path))
        val_uncal = model.predict_proba(x[idx_val])[:, 1]
        test_uncal = model.predict_proba(x[idx_test])[:, 1]
        if calibrated_scores_val is not None and calibrated_scores_test is not None:
            _plot_pr_curves_with_calibration(
                y[idx_val],
                val_uncal,
                calibrated_scores_val,
                y[idx_test],
                test_uncal,
                calibrated_scores_test,
                str(plot_pr_path),
            )
        else:
            _plot_pr_curves(y[idx_val], val_uncal, y[idx_test], test_uncal, str(plot_pr_path))
        plots["pr_curves"] = str(plot_pr_path)

    if args.plot_scores_test:
        plot_scores_path = _resolve_path(args.plot_scores_test)
        assert plot_scores_path is not None
        _ensure_parent(str(plot_scores_path))
        test_scores = (
            calibrated_scores_test
            if calibrated_scores_test is not None
            else model.predict_proba(x[idx_test])[:, 1]
        )
        _plot_score_distributions(
            y[idx_test],
            test_scores,
            str(plot_scores_path),
            title="Test score distributions (pos vs neg)",
        )
        plots["test_score_distributions"] = str(plot_scores_path)

    if args.plot_reliability:
        if calibrated_scores_val is None or calibrated_scores_test is None:
            raise ValueError("--plot-reliability requires --calibration platt|isotonic")
        plot_rel_path = _resolve_path(args.plot_reliability)
        assert plot_rel_path is not None
        _ensure_parent(str(plot_rel_path))
        val_uncal = model.predict_proba(x[idx_val])[:, 1]
        test_uncal = model.predict_proba(x[idx_test])[:, 1]
        _plot_reliability_diagram(
            y[idx_val],
            val_uncal,
            calibrated_scores_val,
            y[idx_test],
            test_uncal,
            calibrated_scores_test,
            str(plot_rel_path),
            n_bins=int(args.reliability_bins),
        )
        plots["reliability"] = str(plot_rel_path)

    report["plots"] = plots if plots else None

    def fmt(v: float | None) -> str:
        return "NA" if v is None else f"{v:.4f}"

    for s in ("train", "val", "test"):
        d = report["splits"][s]
        print(
            f"{s:5s}  n={d['n']:4d}  pos_rate={d['pos_rate']:.4f}  "
            f"AUROC={fmt(d['auroc'])}  AUPRC={fmt(d['auprc'])}"
        )

    if report.get("calibration"):
        dval = report["calibration"]["val"]
        dtest = report["calibration"]["test"]
        print(
            f"calib(val)   AUROC={fmt(dval['auroc'])}  AUPRC={fmt(dval['auprc'])}  method={report['calibration']['method']}"
        )
        print(
            f"calib(test)  AUROC={fmt(dtest['auroc'])}  AUPRC={fmt(dtest['auprc'])}  method={report['calibration']['method']}"
        )

        m = report["calibration"].get("metrics")
        if m:
            tvu = m["test"]["uncalibrated"]
            tvc = m["test"]["calibrated"]
            print(
                "calib(test metrics)  "
                f"Brier uncal={fmt(tvu.get('brier'))} cal={fmt(tvc.get('brier'))}  "
                f"LogLoss uncal={fmt(tvu.get('log_loss'))} cal={fmt(tvc.get('log_loss'))}"
            )

    if args.out_json:
        out_path = _resolve_path(args.out_json)
        assert out_path is not None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
