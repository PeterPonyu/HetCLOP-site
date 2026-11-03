"""Distributional-fidelity metrics for stochastic-decoder evaluation (Phase 2b).

These are the ACTUAL success criteria for judging a stochastic decoder against
a real cell population -- geometry-ratio metrics (tightness) must NOT be used
for that, because Phase 2a (PHASE2A_REPORT.md, finding C3) showed underfitting
INFLATES decoded tightness: blurry / mean-regressed / noisy output scores best.

All three functions are pure (numpy in, float/dict out) and are meant to be
computed PER (dataset, cell type) group with >= 20 real test cells, then
aggregated (median + IQR) across groups by the caller.

Design notes that matter for interpretation:

* ``energy_distance`` uses the standard V-statistic estimator (diagonal
  included), so it is exactly 0 for two identical sample sets and >= 0 always.
  It is a genuine distributional distance: it does NOT reward re-injected noise
  the way a variance ratio does -- a decoder that restores marginal variance but
  gets the location/shape wrong is still penalised.

* ``r_var_per_gene`` is the per-gene within-type variance ratio decoded/real.
  A trivial matched-noise control restores this to ~1 BY CONSTRUCTION, so a
  near-1 r_var is necessary but NOT sufficient evidence of a good stochastic
  decoder. Report it, but do not treat it as the discriminating metric.

* ``covariance_preservation`` was INTENDED as the discriminating metric, but it
  is a scale-invariant Pearson between the two correlation-matrix upper
  triangles: independent per-gene noise attenuates all correlations by a roughly
  common factor, PRESERVING their rank order, so the trivial matched-noise
  control scores HIGH here. It fails to discriminate. Kept for continuity, not
  as a success criterion.

* ``correlation_attenuation`` is the metric independent noise CANNOT fake:
  the ratio of decoded-to-real MEAN ABSOLUTE gene-gene correlation. Independent
  noise dilutes every correlation toward 0, driving this below 1; a model that
  captures real cell-to-cell covariance keeps it near 1.

* ``r_var_dispersion`` is the IQR width of the per-group signed r_var
  distribution. A global-per-gene noise control (one residual std per gene for
  all cell types) is systematically miscalibrated per type -> wide dispersion;
  a per-cell variance-calibrated p(y|z) -> narrow dispersion.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Standard energy distance ``2*E||X-Y|| - E||X-X'|| - E||Y-Y'||``.

    V-statistic estimator (means over the FULL pairwise distance matrices,
    diagonal included). Exactly 0.0 when X and Y are the same sample set,
    strictly >= 0 in general. Lower = the two empirical distributions are
    closer. Rows are samples, columns are features."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2-D with matching feature dimension")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("energy distance is undefined for an empty sample set")
    d_xy = cdist(X, Y).mean()
    d_xx = cdist(X, X).mean()
    d_yy = cdist(Y, Y).mean()
    return float(2.0 * d_xy - d_xx - d_yy)


def r_var_per_gene(real: np.ndarray, decoded: np.ndarray, eps: float = 1e-12) -> dict:
    """Per-gene within-type variance ratio decoded/real.

    Returns a dict with:
      * ``median_signed_ratio``  -- median across genes of var(decoded)/var(real)
        (1.0 = perfectly calibrated marginal variance; >1 over-dispersed,
        <1 under-dispersed). Exactly 1.0 when decoded == real.
      * ``median_abs_log2_ratio`` -- median across genes of |log2(ratio)|
        (0.0 = perfectly calibrated; larger = more miscalibrated in either
        direction). Exactly 0.0 when decoded == real.
      * ``n_genes_scored`` -- genes with real variance > eps (others undefined).

    Sample variance (ddof=1) so a single-cell group is undefined; caller should
    restrict to groups with >= 2 (in practice >= 20) cells."""
    real = np.asarray(real, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if real.shape != decoded.shape:
        raise ValueError("real and decoded must have the same shape")
    if real.shape[0] < 2:
        raise ValueError("variance ratio needs >= 2 cells")
    vr = real.var(axis=0, ddof=1)
    vd = decoded.var(axis=0, ddof=1)
    valid = vr > eps
    if not valid.any():
        return {"median_signed_ratio": float("nan"),
                "median_abs_log2_ratio": float("nan"), "n_genes_scored": 0}
    ratio = vd[valid] / vr[valid]
    # guard log2 against a decoded gene with exactly zero variance
    safe = np.clip(ratio, eps, None)
    return {
        "median_signed_ratio": float(np.median(ratio)),
        "median_abs_log2_ratio": float(np.median(np.abs(np.log2(safe)))),
        "n_genes_scored": int(valid.sum()),
    }


def r_var_per_gene_expressed(
    real: np.ndarray, decoded: np.ndarray,
    min_mean_count: float = 0.5, top_k: int = 200, eps: float = 1e-12,
) -> dict:
    """Within-type variance ratio decoded/real, restricted to genes that actually
    carry signal in the group -- the fix for the sparsity degeneracy of
    ``r_var_per_gene``.

    On a sparse count panel most genes have within-type mean count << 1; the NB
    variance law ``mu + mu^2/theta`` barely moves their variance with theta, and
    discrete low counts pin many per-gene ratios at exactly 1.0, so the
    unrestricted median-across-genes ratio is dominated by uninformative
    near-zero genes and sits at 1.0 for ANY theta (see
    ``test_r_var_unrestricted_median_pinned_but_restricted_recovers``). This
    function restricts to expressed genes so the DIRECTION of the miscalibration
    (over- vs under-inflation) is measurable.

    ``real`` / ``decoded`` are on the log1p scale (metric space); gene selection
    uses the REAL data only. Two restrictions are computed and returned side by
    side so no conclusion hinges on one arbitrary cut:
      * ``expressed`` -- genes whose real within-type MEAN COUNT (recovered as
        ``expm1(real).mean``) is >= ``min_mean_count``.
      * ``topk`` -- the ``top_k`` genes by real within-type variance (log1p), the
        same support energy_distance and the covariance metrics effectively use.

    For each restriction returns:
      * ``median_signed_ratio`` -- median over genes of var(dec)/var(real)
        (>1 over-inflated, <1 under-inflated). A point mass at exactly 1 can bias
        this, so it is reported alongside:
      * ``mean_log2_ratio``     -- mean over genes of log2(ratio), signed; robust
        to a point mass at 1 (log2(1)=0) and symmetric in over/under.
      * ``frac_ratio_gt1``      -- fraction of genes with ratio strictly > 1; the
        cleanest direction statistic -- a point mass at exactly 1 cannot bias a
        strict inequality toward either side (>0.5 => net over-inflation).
      * ``median_abs_log2_ratio`` -- median over genes of |log2(ratio)|, the
        expression-restricted analogue of ``r_var_per_gene``'s unrestricted
        ``median_abs_log2_ratio`` (per-group variance-calibration ERROR, lower =
        better, 0 iff decoded==real). This is the magnitude statistic the paired
        Wilcoxon calibration tests use; restricting it to expressed genes removes
        the sparsity degeneracy (near-zero-count genes pinned at |log2 1|=0 that
        dominate the unrestricted median) that made the unrestricted number
        directionally uninformative -- see the E4/E5 finding.
      * ``n_genes_scored``      -- genes surviving the restriction with real
        variance > eps.
    Sample variance uses ddof=1; caller restricts to groups with >= 20 cells."""
    real = np.asarray(real, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if real.shape != decoded.shape:
        raise ValueError("real and decoded must have the same shape")
    if real.shape[0] < 2:
        raise ValueError("variance ratio needs >= 2 cells")
    vr = real.var(axis=0, ddof=1)
    vd = decoded.var(axis=0, ddof=1)
    mean_count = np.expm1(real).mean(axis=0)  # real is log1p(counts)
    finite_var = vr > eps

    def _stats(sel):
        sel = sel & finite_var
        n = int(sel.sum())
        if n == 0:
            return {"median_signed_ratio": float("nan"),
                    "mean_log2_ratio": float("nan"),
                    "frac_ratio_gt1": float("nan"),
                    "median_abs_log2_ratio": float("nan"), "n_genes_scored": 0}
        ratio = vd[sel] / vr[sel]
        safe = np.clip(ratio, eps, None)
        return {
            "median_signed_ratio": float(np.median(ratio)),
            "mean_log2_ratio": float(np.mean(np.log2(safe))),
            "frac_ratio_gt1": float(np.mean(ratio > 1.0)),
            "median_abs_log2_ratio": float(np.median(np.abs(np.log2(safe)))),
            "n_genes_scored": n,
        }

    expressed = _stats(mean_count >= min_mean_count)
    # top_k by real within-type variance
    k = int(min(top_k, real.shape[1]))
    order = np.argsort(vr)[::-1][:k]
    topk_mask = np.zeros(real.shape[1], dtype=bool)
    topk_mask[order] = True
    topk = _stats(topk_mask)
    return {"expressed": expressed, "topk": topk,
            "min_mean_count": float(min_mean_count), "top_k": k}


def covariance_preservation(real: np.ndarray, decoded: np.ndarray, top_k: int = 200) -> float:
    """Gene-gene co-expression preservation within a cell-type group.

    Selects the ``top_k`` most-variable genes from the REAL data, builds the
    Pearson gene-gene correlation matrix for real and for decoded over those
    genes, and returns the Pearson correlation between the vectorised upper
    triangles. 1.0 = decoded reproduces real co-expression structure exactly;
    ~0 = decoded correlations are unrelated to real (what independent per-gene
    noise injection produces, because it destroys cross-gene covariance).

    Returns nan if fewer than 2 finite off-diagonal correlation pairs exist."""
    real = np.asarray(real, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if real.shape != decoded.shape:
        raise ValueError("real and decoded must have the same shape")
    if real.shape[0] < 3:
        raise ValueError("gene-gene correlation needs >= 3 cells")
    n_genes = real.shape[1]
    k = int(min(top_k, n_genes))
    v = real.var(axis=0, ddof=1)
    idx = np.argsort(v)[::-1][:k]
    R = real[:, idx]
    D = decoded[:, idx]
    cr = np.corrcoef(R, rowvar=False)
    cd = np.corrcoef(D, rowvar=False)
    iu = np.triu_indices(k, k=1)
    a = cr[iu]
    b = cd[iu]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    return float(pearsonr(a[valid], b[valid])[0])


def correlation_attenuation(real: np.ndarray, decoded: np.ndarray, top_k: int = 200) -> float:
    """Ratio ``mean(|r_decoded|) / mean(|r_real|)`` over the gene-gene correlation
    off-diagonals of the ``top_k`` most-variable REAL genes, within a cell-type
    group.

    Unlike ``covariance_preservation`` (a scale-invariant Pearson of the two
    correlation-matrix upper triangles, which independent noise can fake because
    it merely attenuates all correlations by a roughly common factor and so
    preserves their rank order), this measures the ABSOLUTE strength of decoded
    co-expression relative to real. Independent per-gene noise injection drives
    this BELOW 1 (it dilutes every correlation toward 0); a decoder that captures
    real cell-to-cell covariance keeps it near 1. Values > 1 mean the decoded
    cells are MORE correlated than real. Exactly 1.0 when decoded == real.

    Returns nan if fewer than 2 finite off-diagonal pairs exist, or if the real
    mean absolute correlation is ~0 (ratio undefined)."""
    real = np.asarray(real, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if real.shape != decoded.shape:
        raise ValueError("real and decoded must have the same shape")
    if real.shape[0] < 3:
        raise ValueError("gene-gene correlation needs >= 3 cells")
    n_genes = real.shape[1]
    k = int(min(top_k, n_genes))
    v = real.var(axis=0, ddof=1)
    idx = np.argsort(v)[::-1][:k]
    cr = np.corrcoef(real[:, idx], rowvar=False)
    cd = np.corrcoef(decoded[:, idx], rowvar=False)
    iu = np.triu_indices(k, k=1)
    a = np.abs(cr[iu])
    b = np.abs(cd[iu])
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    mean_real = a[valid].mean()
    if mean_real <= 1e-12:
        return float("nan")
    return float(b[valid].mean() / mean_real)


def r_var_dispersion(signed_ratios) -> float:
    """IQR width (Q75 - Q25) of a collection of per-group signed ``r_var``
    (variance-ratio) values.

    A trivial global-per-gene noise control uses ONE residual std per gene across
    all cell types, so it is systematically miscalibrated per type (over-inflating
    intrinsically low-variance types, under-inflating high-variance ones) and its
    signed r_var spreads WIDE across groups. A per-cell variance-calibrated
    ``p(y|z)`` should keep the spread NARROW. Smaller is better. Non-finite
    values are dropped; returns nan if fewer than 2 finite values remain.

    WARNING -- dispersion is CONFOUNDED BY MAGNITUDE and must NEVER be quoted
    alone. It is the spread of the ratio, so a predictor whose within-type
    variance is near ZERO has r_var ~ 0 in every group and therefore a
    deceptively SMALL dispersion -- not because it is well calibrated but because
    the spread of a near-zero quantity is small. A mean-regressed / deterministic
    decoder (r_var median ~ 0.12-0.26 here) posts the LOWEST dispersion of all
    while being the WORST variance restorer. Dispersion is only interpretable
    JOINTLY WITH the r_var median being near 1: restrict any dispersion
    comparison to variants that actually restore variance (e.g. r_var median in
    [0.8, 1.25]) and explicitly EXCLUDE near-zero-variance predictors from it.
    See ``test_r_var_dispersion_is_deceptively_small_for_near_zero_variance``."""
    vals = np.asarray(list(signed_ratios), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


def _iqr(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 25)), float(np.percentile(values, 75))


def aggregate_over_groups(
    real: np.ndarray,
    decoded: np.ndarray,
    labels,
    dataset_ids,
    min_cells: int = 20,
    energy_subsample: int = 300,
    top_k: int = 200,
    seed: int = 0,
) -> dict:
    """Compute all three distributional metrics per (dataset, cell type) group
    with >= ``min_cells`` real cells, then aggregate as median + IQR across
    groups. ``real`` and ``decoded`` are aligned row-for-row (same cells).

    Energy distance subsamples each side to <= ``energy_subsample`` cells with a
    fixed RNG for tractability. Returns per-group records and the aggregate."""
    real = np.asarray(real, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    labels = np.asarray(labels)
    dataset_ids = np.asarray(dataset_ids)
    rng = np.random.default_rng(seed)

    per_group = []
    for d in np.unique(dataset_ids):
        dm = dataset_ids == d
        for t in np.unique(labels[dm]):
            mask = dm & (labels == t)
            n = int(mask.sum())
            if n < min_cells:
                continue
            R = real[mask]
            D = decoded[mask]
            rv = r_var_per_gene(R, D)
            rvx = r_var_per_gene_expressed(R, D, min_mean_count=0.5, top_k=top_k)
            cov = covariance_preservation(R, D, top_k=top_k)
            corr_att = correlation_attenuation(R, D, top_k=top_k)
            if n > energy_subsample:
                ridx = rng.choice(n, energy_subsample, replace=False)
                didx = rng.choice(n, energy_subsample, replace=False)
                ed = energy_distance(R[ridx], D[didx])
            else:
                ed = energy_distance(R, D)
            per_group.append({
                "dataset": str(d), "cell_type": str(t), "n_cells": n,
                "r_var_signed": rv["median_signed_ratio"],
                "r_var_abs_log2": rv["median_abs_log2_ratio"],
                # expression-restricted signed variance-ratio direction stats
                # (fix for the sparsity degeneracy of the unrestricted median)
                "r_var_signed_expr": rvx["expressed"]["median_signed_ratio"],
                "r_var_meanlog2_expr": rvx["expressed"]["mean_log2_ratio"],
                "r_var_frac_gt1_expr": rvx["expressed"]["frac_ratio_gt1"],
                # expression-restricted |log2 r_var| calibration ERROR (lower=better);
                # the restricted analogue of the unrestricted r_var_abs_log2 the
                # paired calibration tests use (E5).
                "r_var_abs_log2_expr": rvx["expressed"]["median_abs_log2_ratio"],
                "n_expr": rvx["expressed"]["n_genes_scored"],
                "r_var_signed_topk": rvx["topk"]["median_signed_ratio"],
                "r_var_meanlog2_topk": rvx["topk"]["mean_log2_ratio"],
                "r_var_frac_gt1_topk": rvx["topk"]["frac_ratio_gt1"],
                "r_var_abs_log2_topk": rvx["topk"]["median_abs_log2_ratio"],
                "n_topk": rvx["topk"]["n_genes_scored"],
                "energy_distance": ed,
                "cov_preservation": cov,
                "correlation_attenuation": corr_att,
            })

    def summarise(key):
        vals = np.array([g[key] for g in per_group if np.isfinite(g[key])], dtype=np.float64)
        if vals.size == 0:
            return {"median": float("nan"), "iqr": [float("nan"), float("nan")], "n": 0}
        q25, q75 = _iqr(vals)
        return {"median": float(np.median(vals)), "iqr": [q25, q75], "n": int(vals.size)}

    return {
        "n_groups": len(per_group),
        "r_var_signed": summarise("r_var_signed"),
        "r_var_abs_log2": summarise("r_var_abs_log2"),
        "r_var_dispersion": r_var_dispersion(g["r_var_signed"] for g in per_group),
        # expression-restricted direction stats (median across groups)
        "r_var_signed_expressed": summarise("r_var_signed_expr"),
        "r_var_meanlog2_expressed": summarise("r_var_meanlog2_expr"),
        "r_var_frac_gt1_expressed": summarise("r_var_frac_gt1_expr"),
        "r_var_abs_log2_expressed": summarise("r_var_abs_log2_expr"),
        "n_expr": summarise("n_expr"),
        "r_var_signed_topk": summarise("r_var_signed_topk"),
        "r_var_meanlog2_topk": summarise("r_var_meanlog2_topk"),
        "r_var_frac_gt1_topk": summarise("r_var_frac_gt1_topk"),
        "r_var_abs_log2_topk_agg": summarise("r_var_abs_log2_topk"),
        "n_topk": summarise("n_topk"),
        "energy_distance": summarise("energy_distance"),
        "cov_preservation": summarise("cov_preservation"),
        "correlation_attenuation": summarise("correlation_attenuation"),
        "per_group": per_group,
    }
