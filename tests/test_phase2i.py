"""Unit tests for the Phase 2i shared encoder builders (phase2i_lib).

The G1/G4 conclusions rest on two constructive properties of the encoders:
  1. ``pca_k_encoder`` returns a k-dim embedding whose captured variance is
     MONOTONE-NON-DECREASING in k (more components -> at least as much information
     about the input) -- the premise of G1's "known information gradient".
  2. ``noise_corrupted_pca_encoder`` monotonically DESTROYS information: the
     canonical correlation between the clean-PCA and corrupted-PCA embeddings
     falls as the injected noise std rises, and noise_std=0 reproduces the clean
     encoder exactly.
Plus a guard that the reused exact-permutation Spearman p is not the scipy
degeneracy for a perfectly monotone n=7 relationship (the G4 statistic).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "experiments" / "phase2i_extensions_2026-07-09"))
sys.path.insert(0, str(HERE / "experiments" / "phase2g_armb_f4_2026-07-09"))

from phase2i_lib import (  # noqa: E402
    adjacent_pairs, noise_corrupted_pca_encoder, pca_k_encoder, randproj_encoder,
)
from phase2g_lib import exact_spearman_perm_p  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    # 400 cells, 30 genes with low-rank structure + noise
    latent = rng.standard_normal((400, 5))
    load = rng.standard_normal((5, 30))
    X = (latent @ load + 0.3 * rng.standard_normal((400, 30))).astype(np.float32)
    tr = np.zeros(400, dtype=bool)
    tr[:300] = True
    return X, tr


def test_pca_k_shapes_and_dim_capping():
    X, tr = _toy()
    for k in (2, 5, 10):
        Z = pca_k_encoder(X, tr, k)
        assert Z.shape == (400, k)
    # k larger than n_features is capped
    Z = pca_k_encoder(X, tr, 1000)
    assert Z.shape[1] <= X.shape[1]


def test_pca_captured_variance_monotone_in_k():
    """Total captured variance (sum of embedding column variances) is
    non-decreasing in k -- the G1 information-gradient premise."""
    X, tr = _toy()
    ks = [2, 5, 10, 20, 30]
    captured = [float(pca_k_encoder(X, tr, k).var(axis=0).sum()) for k in ks]
    assert all(captured[i + 1] >= captured[i] - 1e-5 for i in range(len(captured) - 1))


def test_noise_zero_matches_clean_pca():
    X, tr = _toy()
    clean = pca_k_encoder(X, tr, 5)
    corrupt0 = noise_corrupted_pca_encoder(X, tr, 5, noise_std=0.0)
    assert np.allclose(clean, corrupt0, atol=1e-4)


def test_noise_monotonically_destroys_information():
    """Higher noise_std -> the corrupted PCA embedding is less predictable from
    the clean one (lower captured-by-regression R^2), monotone in the noise."""
    X, tr = _toy()
    clean = pca_k_encoder(X, tr, 5)

    def r2_recover(noise_std):
        Zc = noise_corrupted_pca_encoder(X, tr, 5, noise_std=noise_std, seed=1)
        # least-squares predict each clean component from corrupted embedding
        A = np.linalg.lstsq(Zc, clean, rcond=None)[0]
        pred = Zc @ A
        ss_res = ((clean - pred) ** 2).sum()
        ss_tot = ((clean - clean.mean(0)) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    r2 = [r2_recover(s) for s in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert r2[0] == pytest.approx(1.0, abs=1e-6)
    assert all(r2[i + 1] <= r2[i] + 1e-6 for i in range(len(r2) - 1))
    assert r2[-1] < r2[0]  # strictly destroyed by the largest noise


def test_randproj_shape_and_determinism():
    X, tr = _toy()
    Z1 = randproj_encoder(X, 8, seed=0)
    Z2 = randproj_encoder(X, 8, seed=0)
    assert Z1.shape == (400, 8)
    assert np.array_equal(Z1, Z2)


def test_adjacent_pairs():
    assert adjacent_pairs(["a", "b", "c"]) == [("a", "b"), ("b", "c")]
    assert adjacent_pairs([1]) == []


def test_exact_perm_p_n7_not_degenerate():
    """For a perfectly monotone n=7 relationship the exact two-sided permutation
    p is 2/5040, NOT scipy's degenerate 0.0 -- and it reaches 0.05."""
    x = np.arange(7.0)
    y = 2.0 * x + 1.0
    r = exact_spearman_perm_p(x, y)
    assert r["rho"] == pytest.approx(1.0)
    assert r["n_perm"] == 5040
    assert r["p_two_sided"] == pytest.approx(2 / 5040, abs=1e-9)
    assert r["p_two_sided"] < 0.05
