"""Unit tests for Phase 2j (external-decoder F4 test).

Fast, CPU-only, no full corpus / no GPU. They pin (a) the within-decoder
Poisson-vs-NB control mechanics and sign conventions in phase2j_lib, (b) that a
heavy-dispersion NB over-inflates expressed-gene variance relative to a
dispersion-free Poisson on the SAME mean (the F4 mechanism), (c) the energy
count-redraw sweep shape, and (d) that CFGen's real (vendored) EncoderModel has a
GLOBAL gene-wise theta and a size-factor-summing softmax mean.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
PHASE2J = REPO / "experiments" / "phase2j_external_2026-07-09"
sys.path.insert(0, str(PHASE2J))
sys.path.insert(0, str(PHASE2J / "cfgen_vendored"))

import phase2j_lib as L  # noqa: E402
from src.architecture.stochastic_decoder import (  # noqa: E402
    sample_nb, sample_nb_fixed_theta,
)


def _synthetic_corpus(seed=0, n_ds=6, n_types=3, cells_per_group=40, n_genes=40):
    """Poisson-generated corpus with a clean (dataset, cell_type) group structure,
    all groups >= 20 cells. Returns a data dict compatible with
    evaluate_dispersion_control plus the true per-cell mean used to build it."""
    rng = np.random.default_rng(seed)
    labels, dsids, mus = [], [], []
    for d in range(n_ds):
        for t in range(n_types):
            base = rng.uniform(0.1, 4.0, size=n_genes)  # per-group gene means (counts)
            mu = np.broadcast_to(base, (cells_per_group, n_genes)).copy()
            mu = mu * rng.uniform(0.7, 1.3, size=(cells_per_group, 1))  # per-cell libsize wobble
            mus.append(mu)
            labels += [f"t{t}"] * cells_per_group
            dsids += [f"d{d}"] * cells_per_group
    mu_all = np.vstack(mus)
    raw = rng.poisson(mu_all).astype(np.float32)
    logX = np.log1p(raw).astype(np.float32)
    n = raw.shape[0]
    te = np.ones(n, dtype=bool)  # treat all as test for the control
    data = {
        "raw": raw, "logX": logX,
        "labels": np.array(labels), "dataset_ids": np.array(dsids),
        "te": te, "tr": np.zeros(n, bool), "va": np.zeros(n, bool),
        "lib": raw.sum(1).astype(np.float32),
    }
    return data, mu_all.astype(np.float64)


def test_poisson_on_mean_is_theta_inf():
    """poisson_on_mean must be exactly Poisson(mu): sample_nb_fixed_theta(mu, inf)."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.1, 5.0, size=(200, 30))
    a = sample_nb_fixed_theta(mu, float("inf"), seed=7)
    b = np.random.default_rng(7).poisson(np.clip(mu, 0, None)).astype(np.float32)
    assert np.array_equal(a, b)


def test_heavy_dispersion_nb_over_inflates_vs_poisson():
    """The F4 mechanism, end-to-end through the lib: with the model's mean held at
    the true Poisson mean, a heavy learned dispersion (small theta) over-inflates
    expressed-gene variance, so the dispersion-free Poisson draw is BETTER
    calibrated (|log2 r_var| lower) and WINS the paired contrast."""
    data, mu_true = _synthetic_corpus(seed=2)
    theta = np.full_like(mu_true, 0.5)  # heavy overdispersion (var = mu + 2 mu^2)
    res = L.evaluate_dispersion_control(
        "unit_heavy", mu_true, theta, data, det_log_pearson=0.9,
        seed=0, n_energy_redraws=3)
    c = res["contrasts"]["rvar_expressed"]
    # Poisson wins the per-gene calibration contrast in a clear majority of groups.
    assert c["poisson_win_fraction_groups"] > 0.7, c
    # diff = nb - poisson on a lower-is-better metric; NB worse => positive median.
    assert c["median_paired_diff_nb_minus_pois"] > 0
    # Direction: NB over-inflates expressed variance more than Poisson.
    d = res["direction"]
    assert (d["nb_stochastic"]["r_var_signed_expr_median"]
            > d["poisson_on_mean"]["r_var_signed_expr_median"])


def test_contrast_structure_and_gate():
    data, mu_true = _synthetic_corpus(seed=3)
    theta = np.full_like(mu_true, 1.3)
    res = L.evaluate_dispersion_control(
        "unit_struct", mu_true, theta, data, det_log_pearson=0.5,
        seed=0, n_energy_redraws=4)
    assert set(res["contrasts"]) == {"rvar_expressed", "rvar_topk", "energy"}
    for label, c in res["contrasts"].items():
        for key in ("naive_p", "clustered_wilcoxon_p", "clustered_sign_p",
                    "poisson_win_fraction_groups"):
            assert key in c, (label, key)
    assert "gate_passed" in res["gate"]
    # gate is a one-sided underfit guard: a near-perfect mean must PASS.
    assert res["gate"]["gate_passed"] is True


def test_energy_redraw_sweep_shape():
    data, mu_true = _synthetic_corpus(seed=4)
    theta = np.full_like(mu_true, 1.0)
    res = L.evaluate_dispersion_control(
        "unit_energy", mu_true, theta, data, det_log_pearson=0.5,
        seed=0, n_energy_redraws=5)
    er = res["energy_redraw"]
    assert er["n_redraws"] == 5
    assert len(er["clustered_wilcoxon_p"]) == 5
    assert len(er["naive_p"]) == 5
    assert er["clustered_wilcoxon_p_min"] <= er["clustered_wilcoxon_p_max"]


def test_amortized_detection():
    """theta_summary.is_amortized: True for per-cell varying theta, False for a
    globally-broadcast gene-wise theta."""
    data, mu_true = _synthetic_corpus(seed=5)
    g = np.broadcast_to(np.linspace(0.5, 2.0, mu_true.shape[1]), mu_true.shape).copy()
    res_g = L.evaluate_dispersion_control("g", mu_true, g, data, 0.5, 0, 2)
    assert res_g["theta_summary"]["is_amortized"] is False
    gc = g * np.random.default_rng(0).uniform(0.5, 1.5, size=mu_true.shape)
    res_gc = L.evaluate_dispersion_control("gc", mu_true, gc, data, 0.5, 0, 2)
    assert res_gc["theta_summary"]["is_amortized"] is True


def test_cfgen_real_encoder_global_theta_and_softmax_mean():
    """CFGen's vendored EncoderModel: global gene-wise theta (shape (G,)) and a
    decode whose row sums equal the size factor (softmax * size_factor)."""
    import torch
    from cfgen.models.base.encoder_model import EncoderModel
    torch.manual_seed(0)
    G = 50
    enc = {"rna": {"dims": [32, 16, 8], "batch_norm": True,
                   "dropout": False, "dropout_p": 0.0}}
    m = EncoderModel(in_dim={"rna": G}, encoder_kwargs={k: dict(v) for k, v in enc.items()},
                     learning_rate=1e-3, weight_decay=1e-5,
                     covariate_specific_theta=False, conditioning_covariate="c",
                     n_cat=3, is_binarized=False)
    assert tuple(m.theta.shape) == (G,)  # GLOBAL gene-wise dispersion
    m.eval()
    raw = torch.poisson(torch.rand(16, G) * 3) + 1.0
    xn = torch.log1p(raw)
    sf = raw.sum(1, keepdim=True)
    with torch.no_grad():
        z = m.encode({"X_norm": {"rna": xn}})
        mu = m.decode(z, {"rna": sf})["rna"]
    assert z["rna"].shape == (16, 8)
    assert torch.allclose(mu.sum(1, keepdim=True), sf, rtol=1e-4, atol=1e-3)


# --- theta-conditionality probe (run_theta_probe.py) --------------------------

def test_sample_nb_broadcast_theta_matches_fixed_theta_when_uniform():
    """sample_nb(mu, theta_g_broadcast) must reduce EXACTLY to
    sample_nb_fixed_theta(mu, scalar) when theta_g happens to be uniform across
    genes -- pins that the array-capable substitution used for (ii) in the theta
    probe (sample_nb_fixed_theta only accepts a true scalar) is the same
    Gamma-Poisson mixture, not a different sampling path."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.1, 5.0, size=(100, 20))
    theta_g = np.full(20, 2.5)
    a = sample_nb(mu, np.broadcast_to(theta_g, mu.shape), seed=11)
    b = sample_nb_fixed_theta(mu, 2.5, seed=11)
    assert np.array_equal(a, b)


def test_theta_probe_isolates_conditionality_from_deficit():
    """Coordinator's confound probe, end-to-end: build a synthetic corpus where
    Poisson(mu) under-inflates (a real variance deficit), then compare
    nb_amortized (true per-cell theta_ic, correctly sized per group) against
    nb_global_theta (per-gene MEDIAN theta collapsed across cells) on the SAME
    mu. If the amortized draw calibrates variance and the global draw does not
    (because the "right" dispersion differs by group but the median smears
    across groups), conditionality is doing real work -- confirming the
    mechanism the probe is designed to detect, not merely asserting a
    p-value plumbing check."""

    rng = np.random.default_rng(0)
    n_ds, n_types, cells_per_group, n_genes = 4, 3, 40, 30
    labels, dsids, mu_rows, theta_rows = [], [], [], []
    # Each (dataset, cell_type) group needs a DIFFERENT true dispersion so a
    # single global (median) theta cannot calibrate every group at once, while
    # the per-cell (amortized) theta -- set here to the group's true value --
    # calibrates every group exactly.
    group_thetas = rng.uniform(0.3, 6.0, size=(n_ds, n_types))
    for d in range(n_ds):
        for t in range(n_types):
            base = rng.uniform(2.0, 8.0, size=n_genes)  # deficit-prone mean scale
            mu = np.broadcast_to(base, (cells_per_group, n_genes)).copy()
            true_theta = group_thetas[d, t]
            th = np.full((cells_per_group, n_genes), true_theta)
            mu_rows.append(mu)
            theta_rows.append(th)
            labels += [f"t{t}"] * cells_per_group
            dsids += [f"d{d}"] * cells_per_group
    mu_ic = np.vstack(mu_rows)
    theta_ic = np.vstack(theta_rows)  # "amortized": exactly the group's true theta
    real_counts = sample_nb(mu_ic, theta_ic, seed=99)  # ground truth draw
    real_log = np.log1p(real_counts).astype(np.float32)

    theta_g = np.median(theta_ic, axis=0)  # collapsed across ALL cells/groups
    pois = sample_nb_fixed_theta(mu_ic, float("inf"), seed=1)
    nb_global = sample_nb(mu_ic, np.broadcast_to(theta_g, mu_ic.shape), seed=1)
    nb_amort = sample_nb(mu_ic, theta_ic, seed=1)

    from src.evaluation.distributional import aggregate_over_groups
    labels_arr, ds_arr = np.array(labels), np.array(dsids)
    agg_amort = aggregate_over_groups(real_log, np.log1p(nb_amort).astype(np.float32),
                                      labels_arr, ds_arr, min_cells=20, top_k=n_genes)
    agg_global = aggregate_over_groups(real_log, np.log1p(nb_global).astype(np.float32),
                                       labels_arr, ds_arr, min_cells=20, top_k=n_genes)
    agg_pois = aggregate_over_groups(real_log, np.log1p(pois).astype(np.float32),
                                     labels_arr, ds_arr, min_cells=20, top_k=n_genes)

    # The mechanism the confound probe exists to detect, stated as the aggregate
    # calibration ordering the |log2 r_var| statistic can actually resolve on this
    # fixture:  Poisson leaves the largest deficit; a global-median theta fills it
    # ON AVERAGE but the "right" dispersion differs by group so it stays
    # mis-calibrated; the per-cell amortized theta (set to each group's true value)
    # calibrates every group.  Hence |log2|(amort) < |log2|(global) < |log2|(pois).
    #
    # NOTE (deliberate): a PER-GROUP clustered win-fraction is NOT asserted here.
    # With 30 genes x 40 cells/group the per-group median is too noisy -- amort-vs-
    # global is a coin-flip (win 0.50, clustered p ~ 0.63) despite the clean
    # aggregate gap. The real probe (314 real groups) has the power for the
    # per-group instrument; this synthetic unit test asserts only what it can bear.
    m_amort = agg_amort["r_var_abs_log2_expressed"]["median"]
    m_global = agg_global["r_var_abs_log2_expressed"]["median"]
    m_pois = agg_pois["r_var_abs_log2_expressed"]["median"]
    assert m_amort < m_global < m_pois, (m_amort, m_global, m_pois)
    # conditionality is not negligible: the amort->global step (theta made
    # per-group) is a real fraction of the global->pois step (deficit merely
    # filled on average). Observed ~0.17 on this fixture; assert a floor with margin.
    assert (m_global - m_amort) > 0.1 * (m_pois - m_global)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
