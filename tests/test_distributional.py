import numpy as np
import pytest

from src.evaluation.distributional import (
    aggregate_over_groups,
    correlation_attenuation,
    covariance_preservation,
    energy_distance,
    r_var_dispersion,
    r_var_per_gene,
    r_var_per_gene_expressed,
)


def test_energy_distance_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    assert energy_distance(X, X) == pytest.approx(0.0, abs=1e-9)


def test_energy_distance_positive_for_shifted_samples():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 8))
    Y = rng.normal(size=(80, 8)) + 5.0  # location shift
    ed = energy_distance(X, Y)
    assert ed > 0.0
    # a big shift should dominate within-set spread
    assert ed > energy_distance(X, X + 0.01)


def test_energy_distance_never_negative():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5))
    Y = rng.normal(size=(40, 5)) * 2.0 + 1.0
    assert energy_distance(X, Y) >= -1e-9


def test_r_var_returns_one_when_decoded_equals_real():
    rng = np.random.default_rng(3)
    real = rng.normal(size=(100, 12))
    out = r_var_per_gene(real, real.copy())
    assert out["median_signed_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert out["median_abs_log2_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert out["n_genes_scored"] == 12


def test_r_var_detects_inflated_variance():
    rng = np.random.default_rng(4)
    real = rng.normal(size=(200, 10))
    decoded = real * 2.0  # variance x4 per gene
    out = r_var_per_gene(real, decoded)
    assert out["median_signed_ratio"] == pytest.approx(4.0, rel=1e-6)
    assert out["median_abs_log2_ratio"] == pytest.approx(2.0, rel=1e-6)


def test_cov_preservation_one_when_decoded_equals_real():
    rng = np.random.default_rng(5)
    # low-rank factor structure so genes are genuinely correlated
    factors = rng.normal(size=(80, 3))
    loadings = rng.normal(size=(3, 40))
    real = factors @ loadings + 0.1 * rng.normal(size=(80, 40))
    assert covariance_preservation(real, real.copy(), top_k=40) == pytest.approx(1.0, abs=1e-9)


def test_cov_preservation_drops_with_independent_noise():
    rng = np.random.default_rng(6)
    factors = rng.normal(size=(80, 3))
    loadings = rng.normal(size=(3, 40))
    real = factors @ loadings + 0.1 * rng.normal(size=(80, 40))
    # independent per-gene noise that dominates the signal destroys the
    # gene-gene co-expression structure the factor model created
    per_gene_std = real.std(axis=0)
    noisy = real + 20.0 * per_gene_std * rng.normal(size=real.shape)
    intact = covariance_preservation(real, real.copy(), top_k=40)
    destroyed = covariance_preservation(real, noisy, top_k=40)
    assert intact == pytest.approx(1.0, abs=1e-9)
    assert destroyed < 0.5
    assert destroyed < intact - 0.4


def test_correlation_attenuation_one_when_decoded_equals_real():
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(80, 3))
    loadings = rng.normal(size=(3, 40))
    real = factors @ loadings + 0.1 * rng.normal(size=(80, 40))
    assert correlation_attenuation(real, real.copy(), top_k=40) == pytest.approx(1.0, abs=1e-9)


def test_correlation_attenuation_below_one_for_independent_noise():
    rng = np.random.default_rng(8)
    factors = rng.normal(size=(120, 3))
    loadings = rng.normal(size=(3, 40))
    real = factors @ loadings + 0.1 * rng.normal(size=(120, 40))
    # independent per-gene noise dilutes every gene-gene correlation toward 0,
    # so absolute decoded correlation strength is a fraction of real -> ratio < 1.
    # This is the trap covariance_preservation (rank-based, scale-invariant)
    # misses: additive independent noise leaves the correlation RANKING intact.
    per_gene_std = real.std(axis=0)
    noisy = real + 5.0 * per_gene_std * rng.normal(size=real.shape)
    att = correlation_attenuation(real, noisy, top_k=40)
    assert att < 0.9
    # and the scale-invariant metric is fooled: it stays much higher than att
    cov = covariance_preservation(real, noisy, top_k=40)
    assert cov > att


def test_r_var_dispersion_zero_for_identical_values():
    assert r_var_dispersion([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-12)


def test_r_var_dispersion_wider_for_more_spread():
    narrow = r_var_dispersion([0.95, 1.0, 1.02, 1.05, 0.98])
    wide = r_var_dispersion([0.2, 0.7, 1.0, 1.8, 3.0])
    assert wide > narrow


def test_r_var_dispersion_ignores_nan_and_needs_two():
    assert np.isnan(r_var_dispersion([1.0]))
    assert np.isnan(r_var_dispersion([np.nan, 1.0]))
    assert r_var_dispersion([np.nan, 0.5, 1.5, np.nan]) == pytest.approx(0.5, abs=1e-12)


def test_r_var_dispersion_is_deceptively_small_for_near_zero_variance():
    """Dispersion is confounded by magnitude: a near-zero-variance predictor
    (r_var ~ 0 in every group, like a mean-regressed deterministic decoder)
    gets a DECEPTIVELY SMALL dispersion, SMALLER than a genuinely
    variance-restoring predictor whose ratios scatter around 1. Low dispersion
    is therefore NOT evidence of good calibration and must never be quoted
    without the r_var median. Guards the docstring warning."""
    rng = np.random.default_rng(11)
    # near-zero-variance predictor: ratios tiny, tightly clustered near 0
    near_zero = 0.02 * np.abs(rng.normal(size=200)) + 1e-3
    # genuinely calibrated predictor: ratios scatter around 1
    calibrated = 1.0 + 0.25 * rng.normal(size=200)
    disp_near_zero = r_var_dispersion(near_zero)
    disp_calibrated = r_var_dispersion(calibrated)
    # the WORSE (near-zero-variance) predictor posts the SMALLER dispersion
    assert disp_near_zero < disp_calibrated
    # ...yet its variance calibration is far worse: median ratio nowhere near 1
    assert np.median(near_zero) < 0.1
    assert abs(np.median(calibrated) - 1.0) < 0.1


def test_r_var_unrestricted_median_pinned_but_restricted_recovers():
    """Reproduces the Phase 2d degeneracy: a group where the vast MAJORITY of
    genes are near-zero-expression (sparse counts) and a small MINORITY are
    high-expression and genuinely over-inflated. The unrestricted median signed
    ratio is pinned at ~1.0 by the sparse majority, while the expression- and
    top-k-restricted statistics correctly recover the known over-inflation."""
    rng = np.random.default_rng(0)
    n_cells = 200
    n_sparse, n_high = 1900, 100
    # sparse genes: real ~ Poisson(0.05); decoded ~ Poisson(0.05) (same mean, no
    # inflation) -> ratio distribution log-symmetric about 1, median ~1.
    mu_sparse = 0.05
    real_sparse = rng.poisson(mu_sparse, size=(n_cells, n_sparse))
    dec_sparse = rng.poisson(mu_sparse, size=(n_cells, n_sparse))
    # high genes: real ~ Poisson(20); decoded = OVER-inflated (extra multiplicative
    # noise) so decoded within-type variance is clearly > real.
    mu_high = 20.0
    real_high = rng.poisson(mu_high, size=(n_cells, n_high)).astype(float)
    infl = rng.normal(1.0, 0.6, size=(n_cells, n_high)).clip(0.05, None)
    dec_high = (real_high.mean(0, keepdims=True) * infl)  # same mean, inflated var
    real = np.log1p(np.hstack([real_sparse, real_high]).astype(float))
    decoded = np.log1p(np.hstack([dec_sparse, dec_high]).clip(0, None))

    unrestricted = r_var_per_gene(real, decoded)["median_signed_ratio"]
    # unrestricted median is pinned near 1 by the sparse majority
    assert abs(unrestricted - 1.0) < 0.1

    rvx = r_var_per_gene_expressed(real, decoded, min_mean_count=0.5, top_k=100)
    # both restrictions must select the ~100 high-expression genes only
    assert rvx["expressed"]["n_genes_scored"] <= n_high + 5
    assert rvx["topk"]["n_genes_scored"] == 100
    # and must recover clear OVER-inflation (ratio > 1, frac>1 well above 0.5)
    assert rvx["expressed"]["median_signed_ratio"] > 1.2
    assert rvx["expressed"]["frac_ratio_gt1"] > 0.7
    assert rvx["expressed"]["mean_log2_ratio"] > 0.2
    assert rvx["topk"]["median_signed_ratio"] > 1.2
    assert rvx["topk"]["frac_ratio_gt1"] > 0.7
    # the restricted |log2 r_var| calibration ERROR (E5's paired statistic) must
    # register the clear over-inflation carried by the expressed/top-k genes
    # (log2(1.2+) > 0.26), and be finite.
    assert np.isfinite(rvx["expressed"]["median_abs_log2_ratio"])
    assert rvx["expressed"]["median_abs_log2_ratio"] > 0.2
    assert rvx["topk"]["median_abs_log2_ratio"] > 0.2


def test_r_var_expressed_calibrated_is_neutral():
    """When decoded == real on the expressed genes, restricted stats are neutral:
    ratio 1, frac>1 ~ 0.5-ish (here exactly at 1 -> frac_gt1 == 0), log2 ~ 0."""
    rng = np.random.default_rng(1)
    real = np.log1p(rng.poisson(10.0, size=(150, 300)).astype(float))
    rvx = r_var_per_gene_expressed(real, real.copy(), min_mean_count=0.5, top_k=100)
    assert rvx["expressed"]["median_signed_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert rvx["expressed"]["mean_log2_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert rvx["expressed"]["median_abs_log2_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert rvx["topk"]["median_signed_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert rvx["topk"]["median_abs_log2_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_aggregate_emits_expression_restricted_abs_log2_per_group():
    """aggregate_over_groups must expose the expression-restricted |log2 r_var|
    calibration-error statistic per group (E5's paired-test statistic) under
    r_var_abs_log2_expr / r_var_abs_log2_topk, alongside the unrestricted
    r_var_abs_log2, and each restricted per-group value must equal the direct
    r_var_per_gene_expressed median_abs_log2_ratio on that group's cells."""
    rng = np.random.default_rng(3)
    # two datasets x two cell types, >= 20 cells each, on a log1p count panel
    n_per, n_genes = 40, 400
    reals, decs, labs, dss = [], [], [], []
    for ds in ("d0", "d1"):
        for ct in ("A", "B"):
            counts = rng.poisson(rng.uniform(0.02, 8.0, size=n_genes), size=(n_per, n_genes))
            real = np.log1p(counts.astype(float))
            dec = np.log1p((counts * rng.normal(1.0, 0.3, size=counts.shape)).clip(0, None))
            reals.append(real); decs.append(dec)
            labs += [ct] * n_per; dss += [ds] * n_per
    real = np.vstack(reals); decoded = np.vstack(decs)
    labels = np.array(labs); dataset_ids = np.array(dss)

    agg = aggregate_over_groups(real, decoded, labels, dataset_ids,
                                min_cells=20, energy_subsample=300, top_k=200, seed=0)
    assert agg["n_groups"] == 4
    for g in agg["per_group"]:
        assert "r_var_abs_log2" in g
        assert "r_var_abs_log2_expr" in g and "r_var_abs_log2_topk" in g
        mask = (dataset_ids == g["dataset"]) & (labels == g["cell_type"])
        direct = r_var_per_gene_expressed(real[mask], decoded[mask],
                                          min_mean_count=0.5, top_k=200)
        assert g["r_var_abs_log2_expr"] == pytest.approx(
            direct["expressed"]["median_abs_log2_ratio"], rel=1e-9, nan_ok=True)
        assert g["r_var_abs_log2_topk"] == pytest.approx(
            direct["topk"]["median_abs_log2_ratio"], rel=1e-9, nan_ok=True)
