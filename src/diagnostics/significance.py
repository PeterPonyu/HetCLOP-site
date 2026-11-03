"""Statistical hardening for encoder-comparison diagnostics, per the
independent design review's requirement to pair every point-estimate ratio
with a permutation null and a bootstrap confidence interval."""
from __future__ import annotations

from typing import Callable

import numpy as np


def permutation_null(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    labels,
    n_perm: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(labels)
        null_vals[i] = stat_fn(X, shuffled)
    return null_vals


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    labels,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    types = np.unique(labels)
    idx_by_type = {t: np.where(labels == t)[0] for t in types}
    boot_vals = np.empty(n_boot)
    for i in range(n_boot):
        resampled_idx = np.concatenate(
            [rng.choice(idx_by_type[t], size=len(idx_by_type[t]), replace=True) for t in types]
        )
        boot_vals[i] = stat_fn(X[resampled_idx], labels[resampled_idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_vals, [alpha, 1 - alpha])
    return float(lo), float(hi)
