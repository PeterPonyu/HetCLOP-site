"""Unit tests for the Phase 2c negative-binomial decoder pieces.

Two properties matter for trusting the NB arm:
  1. ``nb_nll`` is minimised at the data-generating (mu, theta) -- i.e. it is a
     correct NLL, not an arbitrary loss. Checked by sampling a large NB dataset
     and confirming the empirical mean NLL bottoms out at the true parameters
     along both the mu and the theta axes.
  2. ``sample_nb`` reproduces the target NB mean and variance
     (E=mu, Var=mu+mu^2/theta) within Monte-Carlo tolerance.
"""
import numpy as np
import pytest
import torch

from src.architecture.stochastic_decoder import nb_nll, sample_nb, sample_nb_fixed_theta


def _nll_np(y, mu, theta):
    """numpy wrapper around the torch nb_nll, returning a python float."""
    return float(nb_nll(
        torch.from_numpy(np.asarray(y, dtype=np.float64)),
        torch.tensor(float(mu), dtype=torch.float64),
        torch.tensor(float(theta), dtype=torch.float64),
    ).item())


def test_nb_nll_zero_dispersion_matches_closed_form():
    """Spot-check the NB NLL against a hand-computed value on tiny input, so a
    future refactor of the formula is caught."""
    y = np.array([0.0, 1.0, 2.0, 5.0])
    mu, theta = 2.0, 3.0
    from scipy.special import gammaln
    ll = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
          + theta * (np.log(theta) - np.log(theta + mu))
          + y * (np.log(mu) - np.log(theta + mu)))
    expected = float(-ll.mean())
    assert _nll_np(y, mu, theta) == pytest.approx(expected, rel=1e-5)


def test_nb_nll_minimised_at_true_mu():
    """Along the mu axis (theta fixed at truth), the empirical mean NLL is
    minimised at the data-generating mu."""
    mu0, theta0 = 4.0, 2.5
    y = sample_nb(np.full(400_000, mu0), np.full(400_000, theta0), seed=0)
    grid = np.array([1.0, 2.0, 3.0, 3.5, mu0, 4.5, 5.0, 6.0, 8.0])
    nlls = np.array([_nll_np(y, m, theta0) for m in grid])
    assert grid[int(np.argmin(nlls))] == pytest.approx(mu0, abs=1e-9)


def test_nb_nll_minimised_at_true_theta():
    """Along the theta axis (mu fixed at truth), the empirical mean NLL is
    minimised nearest the data-generating theta among a coarse grid."""
    mu0, theta0 = 4.0, 2.5
    y = sample_nb(np.full(400_000, mu0), np.full(400_000, theta0), seed=1)
    grid = np.array([0.5, 1.0, 1.5, 2.0, theta0, 3.0, 4.0, 6.0, 12.0])
    nlls = np.array([_nll_np(y, mu0, t) for t in grid])
    assert grid[int(np.argmin(nlls))] == pytest.approx(theta0, abs=1e-9)


def test_nb_nll_finer_theta_minimum_is_close():
    """A finer theta grid places the argmin within 15% of the true theta."""
    mu0, theta0 = 6.0, 3.0
    y = sample_nb(np.full(600_000, mu0), np.full(600_000, theta0), seed=2)
    grid = np.linspace(1.5, 5.0, 36)
    nlls = np.array([_nll_np(y, mu0, t) for t in grid])
    assert grid[int(np.argmin(nlls))] == pytest.approx(theta0, rel=0.15)


def test_sample_nb_reproduces_mean_and_variance():
    """A large NB sample matches E=mu and Var=mu+mu^2/theta within MC tolerance."""
    mu0, theta0 = 5.0, 2.0
    y = sample_nb(np.full(500_000, mu0), np.full(500_000, theta0), seed=3)
    assert y.mean() == pytest.approx(mu0, rel=0.02)
    var_target = mu0 + mu0 ** 2 / theta0  # = 5 + 25/2 = 17.5
    assert y.var() == pytest.approx(var_target, rel=0.03)
    assert (y >= 0).all()


def test_sample_nb_large_theta_approaches_poisson():
    """As theta -> large, NB variance approaches the Poisson variance (= mu)."""
    mu0, theta0 = 5.0, 5000.0
    y = sample_nb(np.full(500_000, mu0), np.full(500_000, theta0), seed=4)
    # Poisson variance is mu=5; NB variance is mu+mu^2/theta = 5+25/5000 ~ 5.005
    assert y.var() == pytest.approx(mu0, rel=0.03)


def test_sample_nb_elementwise_heterogeneous_params():
    """Element-wise mu/theta arrays sample independently with the right per-cell
    mean (checked by replicating each parameter pair many times)."""
    mu = np.repeat([2.0, 20.0], 200_000)
    theta = np.repeat([1.0, 5.0], 200_000)
    y = sample_nb(mu, theta, seed=5)
    assert y[:200_000].mean() == pytest.approx(2.0, rel=0.03)
    assert y[200_000:].mean() == pytest.approx(20.0, rel=0.02)


def test_sample_nb_fixed_theta_imposes_variance():
    """A scalar imposed theta yields NB variance mu+mu^2/theta at the given mean,
    matching sample_nb with a constant theta array (Phase 2d theta-sweep needs a
    single dispersion applied to every element around a fixed mean)."""
    mu0, theta = 5.0, 1.3
    y = sample_nb_fixed_theta(np.full(500_000, mu0), theta, seed=6)
    assert y.mean() == pytest.approx(mu0, rel=0.02)
    var_target = mu0 + mu0 ** 2 / theta
    assert y.var() == pytest.approx(var_target, rel=0.04)
    assert (y >= 0).all()


def test_sample_nb_fixed_theta_inf_is_poisson():
    """theta=inf is the exact Poisson limit: variance equals the mean, not
    mu+mu^2/theta (which is singular there)."""
    mu0 = 7.0
    y = sample_nb_fixed_theta(np.full(500_000, mu0), np.inf, seed=7)
    assert y.mean() == pytest.approx(mu0, rel=0.02)
    assert y.var() == pytest.approx(mu0, rel=0.03)  # Poisson: Var == mean
    # integer-valued counts
    assert np.allclose(y, np.round(y))


def test_sample_nb_fixed_theta_monotone_variance_in_theta():
    """Smaller imposed theta -> larger variance around the SAME mean (the core
    monotonicity the theta-sweep relies on)."""
    mu = np.full(300_000, 6.0)
    variances = [sample_nb_fixed_theta(mu, t, seed=8).var()
                 for t in (0.25, 1.0, 5.0, 100.0)]
    assert variances[0] > variances[1] > variances[2] > variances[3]


def test_sample_nb_fixed_theta_rejects_nonpositive():
    with pytest.raises(ValueError):
        sample_nb_fixed_theta(np.full(10, 3.0), 0.0, seed=0)
