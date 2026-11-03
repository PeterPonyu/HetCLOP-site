"""Acceptance gate for stochastic / count decoders.

The gate exists to catch ONE specific pathology: a stochastic decoder whose MEAN
UNDERFITS relative to the deterministic baseline -- the heteroscedastic-NLL
collapse documented in Phase 2b v1 (mu Pearson 0.445 vs 0.742 deterministic),
where a runaway variance/dispersion head starves the mean's gradient. If the mean
underfits, the model's distributional metrics compare a broken mean to a working
control and prove nothing.

It is therefore a ONE-SIDED underfit guard, NOT a two-sided equivalence test. A
model whose mean fits AS WELL AS or BETTER THAN the deterministic baseline exhibits
none of the collapse pathology and must PASS -- even if it is better by a wide
margin. An earlier two-sided ``abs(gap) <= tol`` formulation wrongly FAILED healthy
models (e.g. an NB decoder whose count-likelihood + library-size offset yields a
BETTER conditional mean than MSE-on-log1p: randproj_512 NB mean 0.785 vs det 0.752,
gap -0.033) -- a false exclusion. The gate must only fire when the mean is WORSE.

Sign convention (matches every call site): ``gap = det_pearson - model_pearson``.
Positive gap => model mean is WORSE than deterministic. The gate fails iff
``gap > tol``.
"""
from __future__ import annotations


def mean_acceptance_gate(det_pearson: float, model_pearson: float,
                         tol: float = 0.02) -> bool:
    """Return True (PASS) iff the model's mean Pearson does not UNDERFIT the
    deterministic baseline by more than ``tol``.

    One-sided: PASS when ``det_pearson - model_pearson <= tol`` (the model is at
    least as good, or worse by no more than tol). FAIL only when the model mean is
    worse by more than tol (the underfit pathology the gate guards against). A
    model that beats the baseline by any margin PASSES."""
    return bool((det_pearson - model_pearson) <= tol)
