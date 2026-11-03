"""Unit tests for the Phase 2i G5 zero-inflated negative-binomial (ZINB) pieces.

Three properties matter for trusting the ZINB arm:
  1. ``zinb_nll`` reduces EXACTLY to ``nb_nll`` as the zero-inflation logit
     -> -inf (pi -> 0), so ZINB nests NB and a null (pi ~ 0) means "no benefit".
  2. ``zinb_nll`` is minimised at the data-generating zero-inflation probability
     (with mu, theta fixed at truth) -- i.e. it is a correct NLL, not an arbitrary
     loss.
  3. ``sample_zinb`` reproduces the ZINB mean/zero-fraction: pi=0 matches
     ``sample_nb``; a positive pi lowers the mean by ~(1-pi) and raises the zero
     fraction.
"""
import numpy as np
import pytest
import torch

from src.architecture.stochastic_decoder import (
    nb_nll, sample_nb, sample_zinb, zi_logit_stats, zinb_nll,
)


def _nb_nll_np(y, mu, theta):
    return float(nb_nll(
        torch.from_numpy(np.asarray(y, dtype=np.float64)),
        torch.tensor(float(mu), dtype=torch.float64),
        torch.tensor(float(theta), dtype=torch.float64),
    ).item())


def _zinb_nll_np(y, mu, theta, zi_logit):
    return float(zinb_nll(
        torch.from_numpy(np.asarray(y, dtype=np.float64)),
        torch.tensor(float(mu), dtype=torch.float64),
        torch.tensor(float(theta), dtype=torch.float64),
        torch.tensor(float(zi_logit), dtype=torch.float64),
    ).item())


def test_zinb_reduces_to_nb_when_pi_zero():
    """zi_logit -> large negative (pi -> 0): ZINB NLL == NB NLL exactly."""
    y = np.array([0.0, 1.0, 2.0, 5.0, 0.0, 3.0])
    mu, theta = 3.0, 2.0
    got = _zinb_nll_np(y, mu, theta, -40.0)  # sigmoid(-40) ~ 4e-18
    expected = _nb_nll_np(y, mu, theta)
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_zinb_nll_minimised_at_true_pi():
    """With mu, theta fixed at truth, the empirical mean NLL bottoms out nearest
    the data-generating zero-inflation probability among a coarse logit grid."""
    mu0, theta0, pi0 = 5.0, 3.0, 0.3
    zi0 = float(np.log(pi0 / (1 - pi0)))  # logit(0.3)
    y = sample_zinb(np.full((400_000, 1), mu0), np.full((400_000, 1), theta0),
                    np.array([zi0]), seed=0).ravel()
    grid_pi = np.array([0.0, 0.1, 0.2, pi0, 0.4, 0.5, 0.6])
    grid_zi = np.where(grid_pi <= 0, -40.0, np.log(np.clip(grid_pi, 1e-6, 1 - 1e-6)
                                                  / (1 - np.clip(grid_pi, 1e-6, 1 - 1e-6))))
    nlls = np.array([_zinb_nll_np(y, mu0, theta0, z) for z in grid_zi])
    assert grid_pi[int(np.argmin(nlls))] == pytest.approx(pi0, abs=1e-9)


def test_sample_zinb_pi_zero_matches_nb():
    """pi=0 (zi_logit very negative): ZINB samples match NB mean and variance."""
    mu0, theta0 = 5.0, 2.0
    n = 500_000
    y = sample_zinb(np.full((n, 1), mu0), np.full((n, 1), theta0),
                    np.array([-40.0]), seed=3).ravel()
    assert y.mean() == pytest.approx(mu0, rel=0.02)
    var_target = mu0 + mu0 ** 2 / theta0  # 17.5
    assert y.var() == pytest.approx(var_target, rel=0.04)


def test_sample_zinb_positive_pi_lowers_mean_and_adds_zeros():
    """A positive per-gene pi zeros ~pi of the entries: mean scales by ~(1-pi)
    and the zero fraction rises above the NB baseline."""
    mu0, theta0, pi0 = 8.0, 3.0, 0.4
    zi0 = float(np.log(pi0 / (1 - pi0)))
    n = 500_000
    nb = sample_nb(np.full((n, 1), mu0), np.full((n, 1), theta0), seed=5).ravel()
    zi = sample_zinb(np.full((n, 1), mu0), np.full((n, 1), theta0),
                     np.array([zi0]), seed=5).ravel()
    assert zi.mean() == pytest.approx((1 - pi0) * mu0, rel=0.03)
    assert (zi == 0).mean() > (nb == 0).mean() + 0.2  # clearly more zeros


def test_sample_zinb_per_gene_pi_independent():
    """A per-gene zi vector applies gene-wise: gene 0 (pi~0) keeps the NB mean,
    gene 1 (pi=0.5) halves it."""
    n = 300_000
    mu = np.full((n, 2), 10.0)
    theta = np.full((n, 2), 4.0)
    zi = np.array([-40.0, 0.0])  # pi = [~0, 0.5]
    y = sample_zinb(mu, theta, zi, seed=7)
    assert y[:, 0].mean() == pytest.approx(10.0, rel=0.03)
    assert y[:, 1].mean() == pytest.approx(5.0, rel=0.03)


def test_zi_logit_stats_reports_pi():
    zi = np.array([-40.0, 0.0, 40.0])  # pi = [~0, 0.5, ~1]
    s = zi_logit_stats(zi)
    assert s["pi_min"] == pytest.approx(0.0, abs=1e-6)
    assert s["pi_median"] == pytest.approx(0.5, abs=1e-6)
    assert s["pi_max"] == pytest.approx(1.0, abs=1e-6)
    assert s["frac_genes_pi_gt_0.5"] == pytest.approx(1 / 3, abs=1e-9)
