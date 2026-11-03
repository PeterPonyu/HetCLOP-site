"""Paper-level multiplicity machinery for Phase 2h (T2).

Two corrections applied across an arbitrary set of hypothesis tests:

  * ``holm``           -- Holm-Bonferroni step-down (controls FWER).
  * ``benjamini_hochberg`` -- BH step-up (controls FDR).

Both take a list of raw two-sided p-values and return adjusted p-values in the
SAME order as the input (the standard convention: adjusted_p[i] corresponds to
raw_p[i]). "Survives at level q" means adjusted_p <= q.

The adjusted values are enforced monotone in the sorted order (the usual
step-down / step-up cumulative max / min), so a test never reports an adjusted
p smaller than a more-significant test in its own family.

"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def holm(pvals: Sequence[float]):
    """Holm-Bonferroni step-down adjusted p-values (FWER control).

    Sort ascending; the k-th smallest (0-based j) is multiplied by (m - j);
    enforce a running cumulative maximum so adjusted values are monotone in the
    sorted order; clip to 1. Returned in input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    factors = m - np.arange(m)  # m, m-1, ..., 1
    adj_sorted = ranked * factors
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out


def benjamini_hochberg(pvals: Sequence[float]):
    """Benjamini-Hochberg step-up adjusted p-values (FDR control).

    Sort ascending; the k-th smallest (1-based rank i) is scaled by m/i; enforce
    a running cumulative minimum from the largest downward so adjusted values are
    monotone; clip to 1. Returned in input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    adj_sorted = ranked * m / ranks
    # step-up: enforce monotone non-decreasing by taking min from the top down
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out


def apply_corrections(tests, p_key="raw_p"):
    """Attach paper-level ``holm_p_paper`` and ``bh_q_paper`` to each test dict.

    ``tests`` is a list of dicts each carrying a raw two-sided p under ``p_key``.
    Returns a NEW list of dicts (shallow-copied) with the two adjusted fields and
    boolean ``survives_holm_0.05`` / ``survives_bh_0.05`` added. Order preserved.
    """
    pvals = [float(t[p_key]) for t in tests]
    h = holm(pvals)
    b = benjamini_hochberg(pvals)
    out = []
    for t, hp, bq in zip(tests, h, b):
        rec = dict(t)
        rec["holm_p_paper"] = float(hp)
        rec["bh_q_paper"] = float(bq)
        rec["survives_holm_0.05"] = bool(hp <= 0.05)
        rec["survives_bh_0.05"] = bool(bq <= 0.05)
        out.append(rec)
    return out
