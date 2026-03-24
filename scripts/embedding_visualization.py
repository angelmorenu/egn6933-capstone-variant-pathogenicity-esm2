"""Week 11 embedding-space visualization (UMAP/t-SNE).

Builds 2D projections from ESM2 embeddings and saves publication-ready figures.

Supports coloring by:
- label
- split
- model agreement (RF vs XGBoost correctness overlap)

Example:
  python scripts/embedding_visualization.py \
    --data data/processed/week4_curated_dataset.parquet \
    --method umap \
    --color-by label \
    --out-png results/embedding_umap_by_label.png

  python scripts/embedding_visualization.py \
    --data data/processed/week4_curated_dataset.parquet \
    --method umap \
    --color-by split \
    --out-png results/embedding_umap_by_split.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

import xgboost as xgb


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Resolve paths relative to the repo root, 
# ensuring consistent file handling across environments.
def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path

# 2D projections using UMAP or t-SNE, with optional PCA preprocessing for t-SNE.
def _prepare_agreement_labels(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    rf_report_path: Path,
    xgb_report_path: Path,
    threshold: float,
) -> pd.Series:
    y = df["label"].to_numpy(dtype=int)
    split = df["split"].astype(str).to_numpy()

    idx_train = np.where(split == "train")[0]
    if idx_train.size == 0:
        raise ValueError("Dataset must contain train split for agreement mode")

    x_train = embeddings[idx_train]
    y_train = y[idx_train]

    rf_report = json.loads(rf_report_path.read_text())
    rf_params = rf_report.get("model_params", {})
    rf_seed = int(rf_report.get("seed", 0))

    xgb_report = json.loads(xgb_report_path.read_text())
    xgb_params = xgb_report.get("bayesian_search", {}).get("best_params", {})
    xgb_seed = int(xgb_report.get("seed", 42))

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
    rf_prob = rf_model.predict_proba(embeddings)[:, 1]
    rf_pred = (rf_prob >= float(threshold)).astype(int)

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
    xgb_prob = xgb_model.predict_proba(embeddings)[:, 1]
    xgb_pred = (xgb_prob >= float(threshold)).astype(int)

    rf_correct = rf_pred == y
    xgb_correct = xgb_pred == y

    agreement = np.where(
        rf_correct & xgb_correct,
        "both_correct",
        np.where(
            (~rf_correct) & (~xgb_correct),
            "both_wrong",
            np.where((~rf_correct) & xgb_correct, "rf_only_wrong", "xgb_only_wrong"),
        ),
    )

    return pd.Series(agreement, index=df.index, name="agreement")

# Format p-values for display, including handling bootstrap bounds and scientific notation.
def _compute_projection(
    x: np.ndarray,
    method: str,
    random_state: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    tsne_perplexity: float,
    tsne_learning_rate: float,
) -> tuple[np.ndarray, str]:
    if method == "umap":
        try:
            from umap import UMAP  # type: ignore
        except Exception:
            print("Warning: UMAP unavailable in this environment; falling back to t-SNE.")
        else:
            reducer = UMAP(
                n_components=2,
                n_neighbors=int(umap_n_neighbors),
                min_dist=float(umap_min_dist),
                metric="euclidean",
                random_state=int(random_state),
            )
            return reducer.fit_transform(x), "umap"

    # t-SNE path
    pca_dims = min(50, x.shape[1], x.shape[0] - 1)
    if pca_dims >= 2:
        x_reduced = PCA(n_components=pca_dims, random_state=int(random_state)).fit_transform(x)
    else:
        x_reduced = x

    tsne = TSNE(
        n_components=2,
        perplexity=float(tsne_perplexity),
        learning_rate=float(tsne_learning_rate),
        init="pca",
        random_state=int(random_state),
        n_iter=1000,
    )
    return tsne.fit_transform(x_reduced), "tsne"

# Generate a scatter plot of the 2D projections, 
# coloring points by the specified category (label, split, or agreement).
def _scatter_plot(vis_df: pd.DataFrame, color_by: str, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    marker_map = {"train": "o", "val": "s", "test": "^"}

    if color_by == "label":
        color_map = {"benign": "#1f77b4", "pathogenic": "#d62728"}
        vis_df = vis_df.copy()
        vis_df["color_group"] = vis_df["label"].map({0: "benign", 1: "pathogenic"})
    elif color_by == "split":
        color_map = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
        vis_df = vis_df.copy()
        vis_df["color_group"] = vis_df["split"].astype(str)
    else:
        color_map = {
            "both_correct": "#2ca02c",
            "both_wrong": "#d62728",
            "rf_only_wrong": "#9467bd",
            "xgb_only_wrong": "#ff7f0e",
        }
        vis_df = vis_df.copy()
        vis_df["color_group"] = vis_df["agreement"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 7))

    for split_name in ["train", "val", "test"]:
        split_df = vis_df[vis_df["split"].astype(str) == split_name]
        if split_df.empty:
            continue
        for color_group, sub_df in split_df.groupby("color_group"):
            ax.scatter(
                sub_df["dim1"],
                sub_df["dim2"],
                s=12,
                alpha=0.70,
                marker=marker_map.get(split_name, "o"),
                c=color_map.get(str(color_group), "#7f7f7f"),
                label=f"{color_group} | {split_name}",
                linewidths=0,
            )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.15)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # remove duplicates while preserving order
        seen = set()
        uniq = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                uniq.append((handle, label))
        ax.legend(
            [h for h, _ in uniq],
            [l for _, l in uniq],
            fontsize=8,
            loc="best",
            frameon=True,
            ncol=1,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 11 embedding visualization")
    parser.add_argument("--data", default="data/processed/week4_curated_dataset.parquet")
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--color-by", choices=["label", "split", "agreement"], default="label")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.10)

    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0)

    parser.add_argument("--agreement-threshold", type=float, default=0.5)
    parser.add_argument("--rf-report", default="results/baseline_rf_seed37_bootstrap.json")
    parser.add_argument("--xgb-report", default="results/xgboost_train_eval_report.json")

    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-json", default="results/embedding_visualization_summary.json")
    args = parser.parse_args()

    data_path = _resolve_path(args.data)
    out_png = _resolve_path(args.out_png)
    out_json = _resolve_path(args.out_json)

    assert data_path is not None and out_png is not None and out_json is not None

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_parquet(data_path)
    required = {"embedding", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    sample_df = df.copy()
    if len(sample_df) > int(args.max_points):
        sample_df = sample_df.sample(n=int(args.max_points), random_state=int(args.seed)).reset_index(drop=True)

    x = np.asarray(sample_df["embedding"].tolist(), dtype=np.float32)

    if args.color_by == "agreement":
        rf_path = _resolve_path(args.rf_report)
        xgb_path = _resolve_path(args.xgb_report)
        if rf_path is None or xgb_path is None or not rf_path.exists() or not xgb_path.exists():
            raise FileNotFoundError("Agreement mode requires valid --rf-report and --xgb-report files")
        sample_df["agreement"] = _prepare_agreement_labels(
            df=sample_df,
            embeddings=x,
            rf_report_path=rf_path,
            xgb_report_path=xgb_path,
            threshold=float(args.agreement_threshold),
        )

    projection, method_used = _compute_projection(
        x=x,
        method=args.method,
        random_state=int(args.seed),
        umap_n_neighbors=int(args.umap_n_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        tsne_perplexity=float(args.tsne_perplexity),
        tsne_learning_rate=float(args.tsne_learning_rate),
    )

    vis_df = sample_df.copy()
    vis_df["dim1"] = projection[:, 0]
    vis_df["dim2"] = projection[:, 1]

    title = f"{method_used.upper()} projection colored by {args.color_by}"
    _scatter_plot(vis_df=vis_df, color_by=args.color_by, out_path=out_png, title=title)

    summary: dict[str, Any] = {
        "data": str(data_path),
        "method_requested": args.method,
        "method_used": method_used,
        "color_by": args.color_by,
        "seed": int(args.seed),
        "n_points": int(len(vis_df)),
        "split_counts": {str(k): int(v) for k, v in vis_df["split"].value_counts().to_dict().items()},
        "label_counts": {str(k): int(v) for k, v in vis_df["label"].value_counts().to_dict().items()},
        "dim1_range": [float(vis_df["dim1"].min()), float(vis_df["dim1"].max())],
        "dim2_range": [float(vis_df["dim2"].min()), float(vis_df["dim2"].max())],
        "out_png": str(out_png),
    }

    if args.color_by == "agreement":
        summary["agreement_counts"] = {
            str(k): int(v) for k, v in vis_df["agreement"].value_counts().to_dict().items()
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")
    print(f"n_points: {len(vis_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
