r"""Plot a compact summary of the baseline comparison stats JSON.

Creates an advisor-friendly "effect size" plot:
- ΔAUROC (RF − LogReg) with 95% CI + p-values (bootstrap + DeLong)
- ΔMCC (RF − LogReg) with 95% CI + p-values (bootstrap + McNemar)

Usage:
    python scripts/plot_compare_baselines_stats.py \
        --json "results/Week 5/compare_logreg_vs_rf_stats.json" \
        --json "results/Week 5/compare_logreg_vs_rf_stats_bseed1.json" \
        --json "results/Week 5/compare_logreg_vs_rf_stats_bseed2.json" \
        --out "results/Week 5/compare_logreg_vs_rf_stats_summary.png"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _ensure_parent(path: Path) -> None:
    path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _fmt_p(p: float | None, *, iters: int | None = None) -> str:
    if p is None:
        return "NA"
    if p == 0.0 and iters:
        # Bootstrap p-values can be 0.0 when no resamples support H0.
        # Report a conservative bound based on the bootstrap resolution.
        bound = 1.0 / float(iters)
        return f"<{bound:.0e}"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


@dataclass(frozen=True)
class Run:
    label: str
    boot_iters: int | None
    delta_auroc_mean: float
    delta_auroc_low: float
    delta_auroc_high: float
    p_auroc_boot: float | None
    p_auroc_delong: float | None
    delta_mcc_mean: float
    delta_mcc_low: float
    delta_mcc_high: float
    p_mcc_boot: float | None
    p_mcc_mcnemar: float | None


def _load_run(path: Path) -> Run:
    d: dict[str, Any] = json.loads(path.read_text())

    seed = d.get("seed", {})
    if isinstance(seed, dict):
        label = f"bseed={seed.get('bootstrap_seed', 'NA')}"
    else:
        label = f"seed={seed}"

    delta = d["delta"]
    au = delta["auroc_rf_minus_logreg"]
    mc = delta["mcc_rf_minus_logreg"]

    boot = d.get("bootstrap", {})
    boot_iters = boot.get("iters")
    if not isinstance(boot_iters, int):
        iters_from_delta = au.get("iters")
        boot_iters = iters_from_delta if isinstance(iters_from_delta, int) else None

    tests = d.get("tests", {})
    delong_p = (tests.get("delong_auroc") or {}).get("p_two_sided")
    mcnemar_p = (tests.get("mcnemar_test") or {}).get("p_two_sided")

    return Run(
        label=label,
        boot_iters=boot_iters,
        delta_auroc_mean=float(au["delta_mean"]),
        delta_auroc_low=float(au["delta_ci_low"]),
        delta_auroc_high=float(au["delta_ci_high"]),
        p_auroc_boot=au.get("p_two_sided"),
        p_auroc_delong=delong_p,
        delta_mcc_mean=float(mc["delta_mean"]),
        delta_mcc_low=float(mc["delta_ci_low"]),
        delta_mcc_high=float(mc["delta_ci_high"]),
        p_mcc_boot=mc.get("p_two_sided"),
        p_mcc_mcnemar=mcnemar_p,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ΔAUROC/ΔMCC summary from compare_baselines_stats outputs.")
    parser.add_argument(
        "--json",
        action="append",
        required=True,
        help="Path to a compare_logreg_vs_rf_stats*.json (repeatable).",
    )
    parser.add_argument(
        "--out",
        default="results/Week 5/compare_logreg_vs_rf_stats_summary.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Baseline comparison (RF − LogReg) on held-out gene-disjoint test set",
        help="Figure title.",
    )

    args = parser.parse_args()

    json_paths = [_resolve_path(p) for p in args.json]
    out_path = _resolve_path(args.out)
    _ensure_parent(out_path)

    runs = [_load_run(p) for p in json_paths]

    # Sort by bootstrap seed if available (keeps row labels stable).
    def _sort_key(r: Run) -> tuple[int, str]:
        try:
            return (int(r.label.split("=")[-1]), r.label)
        except Exception:
            return (10**9, r.label)

    runs = sorted(runs, key=_sort_key)

    # Plot: two rows (AUROC, MCC), multiple runs as CI bars.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 5.8), sharex=True)

    for ax, metric_name in zip(axes, ["ΔAUROC", "ΔMCC"], strict=True):
        ax.axvline(0.0, color="0.5", lw=1.0, ls="--")
        ax.set_ylabel(metric_name)
        ax.grid(True, axis="x", color="0.9")

    y_positions = list(range(len(runs)))

    note_bbox = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25)

    # AUROC row
    ax = axes[0]
    for y, r in zip(y_positions, runs, strict=True):
        ax.plot([r.delta_auroc_low, r.delta_auroc_high], [y, y], color="C0", lw=3)
        ax.plot(r.delta_auroc_mean, y, marker="o", color="C0")
        ax.text(
            0.99,
            y,
            f"boot p={_fmt_p(r.p_auroc_boot, iters=r.boot_iters)}  DeLong p={_fmt_p(r.p_auroc_delong)}",
            va="center",
            ha="right",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            bbox=note_bbox,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.label for r in runs])

    # MCC row
    ax = axes[1]
    for y, r in zip(y_positions, runs, strict=True):
        ax.plot([r.delta_mcc_low, r.delta_mcc_high], [y, y], color="C1", lw=3)
        ax.plot(r.delta_mcc_mean, y, marker="o", color="C1")
        ax.text(
            0.99,
            y,
            f"boot p={_fmt_p(r.p_mcc_boot, iters=r.boot_iters)}  McNemar p={_fmt_p(r.p_mcc_mcnemar)}",
            va="center",
            ha="right",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            bbox=note_bbox,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.label for r in runs])

    # X label and title
    axes[1].set_xlabel("Effect size (RF − LogReg)")
    fig.suptitle(args.title)

    # Footnote clarifying thresholding and why only bootstrap varies.
    fig.text(
        0.01,
        0.005,
        "Note: ΔMCC uses per-model validation-selected thresholds (val-max-MCC). Bootstrap seed affects CI/boot p only.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
