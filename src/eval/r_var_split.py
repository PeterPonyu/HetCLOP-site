"""Latent-scale vs decoded-expression-scale per-gene/per-dimension variance
correlation, kept as two distinct functions per independent design review Q5:
the earlier results show these two scales tell very different stories
(latent r_var = +0.201 with real headroom to +0.813; decoded r_var near
zero with the oracle itself capped near 1.0 on a different statistic) and
must never be reported as a single conflated number."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def r_var_decoded(real_expr: np.ndarray, gen_expr: np.ndarray) -> float:
    real_var = np.asarray(real_expr).var(axis=0)
    gen_var = np.asarray(gen_expr).var(axis=0)
    r, _ = pearsonr(real_var, gen_var)
    return float(r)


def r_var_latent(real_latent: np.ndarray, gen_latent: np.ndarray, labels) -> float:
    labels = np.asarray(labels)
    real_vars, gen_vars = [], []
    for t in np.unique(labels):
        mask = labels == t
        if mask.sum() < 2:
            continue
        real_vars.append(np.asarray(real_latent)[mask].var(axis=0))
        gen_vars.append(np.asarray(gen_latent)[mask].var(axis=0))
    if not real_vars:
        raise ValueError("no cell type has >= 2 cells; latent r_var is undefined")
    real_pooled = np.concatenate(real_vars)
    gen_pooled = np.concatenate(gen_vars)
    # pearsonr raises ValueError on length-1 input; with only one qualifying
    # group and a 1-D latent space, the pooled arrays can have length 1, making
    # correlation mathematically undefined — return nan in that case rather
    # than crashing.
    try:
        r, _ = pearsonr(real_pooled, gen_pooled)
    except ValueError:
        r = float('nan')
    return float(r)
