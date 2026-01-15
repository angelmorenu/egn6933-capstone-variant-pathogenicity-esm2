from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ChromosomeSplitPlan:
    train_chroms: tuple[str, ...]
    val_chroms: tuple[str, ...]
    test_chroms: tuple[str, ...]


def _normalize_chrom(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text


def make_chromosome_holdout_plan(
    chromosomes: Iterable[object],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    exclude: Iterable[str] = ("", "na", "none", "un", "mt"),
) -> ChromosomeSplitPlan:
    """Create a simple chromosome-based split plan.

    Notes:
    - This is leakage-aware but not label-stratified (chromosomes are the unit).
    - Use `search_balanced_chromosome_holdout_plan` if you want a balance-aware search.
    """

    exclude_set = {e.lower() for e in exclude}
    chroms = sorted(
        {
            _normalize_chrom(c)
            for c in chromosomes
            if _normalize_chrom(c).lower() not in exclude_set
        }
    )
    if not chroms:
        raise ValueError("No chromosomes found after normalization/exclusion")

    rng = np.random.default_rng(seed)
    shuffled = chroms.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    n_train = max(1, n - n_test - n_val)

    train = tuple(sorted(shuffled[:n_train]))
    val = tuple(sorted(shuffled[n_train : n_train + n_val]))
    test = tuple(sorted(shuffled[n_train + n_val :]))

    return ChromosomeSplitPlan(train_chroms=train, val_chroms=val, test_chroms=test)


def search_balanced_chromosome_holdout_plan(
    chroms: np.ndarray,
    labels: np.ndarray,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    max_iters: int = 2000,
) -> ChromosomeSplitPlan:
    """Search for a chromosome split that approximately matches dataset fractions and label rates.

    This does *not* guarantee perfect balance; it picks the best plan found.

    Parameters
    - chroms: chromosome per row
    - labels: binary labels per row (0/1)
    """

    if chroms.shape[0] != labels.shape[0]:
        raise ValueError("chroms and labels must have the same length")

    chroms_norm = np.array([_normalize_chrom(c) for c in chroms], dtype=object)
    unique_chroms = sorted({c for c in chroms_norm if c and c.lower() not in {"na", "none", "un", "mt"}})
    if len(unique_chroms) < 3:
        raise ValueError("Need at least 3 chromosomes to form train/val/test")

    total_n = labels.shape[0]
    target_test = test_fraction
    target_val = val_fraction
    overall_pos_rate = float(np.mean(labels))

    # Precompute per-chrom stats
    chrom_to_n = {}
    chrom_to_pos = {}
    for c in unique_chroms:
        mask = chroms_norm == c
        chrom_to_n[c] = int(mask.sum())
        chrom_to_pos[c] = int(labels[mask].sum())

    def score(plan: ChromosomeSplitPlan) -> float:
        def agg(chrom_list: tuple[str, ...]) -> tuple[int, int]:
            n = sum(chrom_to_n[c] for c in chrom_list)
            p = sum(chrom_to_pos[c] for c in chrom_list)
            return n, p

        n_train, p_train = agg(plan.train_chroms)
        n_val, p_val = agg(plan.val_chroms)
        n_test, p_test = agg(plan.test_chroms)

        frac_val = n_val / total_n
        frac_test = n_test / total_n

        def rate(n: int, p: int) -> float:
            return float(p / n) if n > 0 else 0.0

        # Penalize deviation from target fractions and overall pos rate
        s = 0.0
        s += abs(frac_val - target_val) * 4.0
        s += abs(frac_test - target_test) * 4.0
        s += abs(rate(n_train, p_train) - overall_pos_rate) * 1.0
        s += abs(rate(n_val, p_val) - overall_pos_rate) * 2.0
        s += abs(rate(n_test, p_test) - overall_pos_rate) * 2.0

        # Penalize tiny val/test
        if n_val < 500:
            s += 2.0
        if n_test < 500:
            s += 2.0
        return s

    rng = np.random.default_rng(seed)
    best_plan = None
    best_score = float("inf")

    chrom_list = unique_chroms
    n_chrom = len(chrom_list)
    n_test_chrom = max(1, int(round(n_chrom * test_fraction)))
    n_val_chrom = max(1, int(round(n_chrom * val_fraction)))

    for _ in range(max_iters):
        shuffled = chrom_list.copy()
        rng.shuffle(shuffled)

        test = tuple(sorted(shuffled[:n_test_chrom]))
        val = tuple(sorted(shuffled[n_test_chrom : n_test_chrom + n_val_chrom]))
        train = tuple(sorted(shuffled[n_test_chrom + n_val_chrom :]))
        if not train:
            continue

        plan = ChromosomeSplitPlan(train_chroms=train, val_chroms=val, test_chroms=test)
        s = score(plan)
        if s < best_score:
            best_score = s
            best_plan = plan

    if best_plan is None:
        raise RuntimeError("Failed to find a valid chromosome split plan")
    return best_plan
