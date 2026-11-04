"""Unit tests for the Phase 2g (Arm-B F4 replication) helpers.

Covers the two load-bearing new pieces:
  * gin_library_size -- the anti-leak library-size offset (G_in only).
  * exact_spearman_perm_p -- the honest 4!-permutation p that replaces scipy's
    degenerate p=0.0 for a perfect Spearman rho.
plus assert_log1p_scale (the control-space guard) and build_armB_encoders shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PHASE2G = HERE.parent / "experiments" / "phase2g_armb_f4_2026-07-09"
sys.path.insert(0, str(PHASE2G))

from phase2g_lib import (  # noqa: E402
    assert_log1p_scale, build_armB_encoders, exact_spearman_perm_p, gin_library_size,
)


# --------------------------------------------------------------------------- #
# gin_library_size: the offset must be G_in depth and never leak G_out.
# --------------------------------------------------------------------------- #
def test_gin_library_size_uses_gin_only():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 50, size=(200, 10)).astype(np.float32)
    gin_idx = np.array([0, 1, 2, 3, 4])
    gout_idx = np.array([5, 6, 7, 8, 9])
    lib = gin_library_size(raw, gin_idx, gout_idx)
    # must equal the G_in depth exactly ...
    assert np.allclose(lib, raw[:, gin_idx].sum(axis=1))
    # ... and must NOT equal the G_out depth (the leak the arm must avoid)
    assert not np.allclose(lib, raw[:, gout_idx].sum(axis=1))


def test_gin_library_size_rejects_overlapping_split():
    raw = np.ones((10, 6), dtype=np.float32)
    with pytest.raises(AssertionError):
        gin_library_size(raw, [0, 1, 2, 3], [3, 4, 5])  # index 3 in both


def test_gin_library_size_rejects_zero_depth_cell():
    raw = np.ones((5, 6), dtype=np.float32)
    raw[2, :3] = 0.0  # cell 2 has zero G_in depth
    with pytest.raises(AssertionError):
        gin_library_size(raw, [0, 1, 2], [3, 4, 5])


# --------------------------------------------------------------------------- #
# exact_spearman_perm_p: honest p for a perfect monotone (scipy gives 0.0).
# --------------------------------------------------------------------------- #
def test_exact_perm_p_perfect_increasing():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])  # rho = +1
    res = exact_spearman_perm_p(x, y)
    assert res["rho"] == pytest.approx(1.0)
    assert res["n_perm"] == 24
    assert res["p_one_sided"] == pytest.approx(1 / 24)   # 0.0417
    assert res["p_two_sided"] == pytest.approx(2 / 24)   # 0.0833


def test_exact_perm_p_perfect_decreasing_matches_spec_defect12():
    # the Arm-A F4 case: rho = -1 -> one-sided 1/24=0.042, two-sided 2/24=0.083.
    # full-precision Arm-A values (spec §20: "no ties at full precision")
    x = np.array([-0.033, -0.006, 0.022, 0.031])   # mean-fit gaps
    y = np.array([0.0147, 0.0045, -0.0210, -0.0211])  # energy effect (nb-pois)
    res = exact_spearman_perm_p(x, y)
    assert res["rho"] == pytest.approx(-1.0)
    assert res["p_one_sided"] == pytest.approx(1 / 24, abs=1e-9)
    assert res["p_two_sided"] == pytest.approx(2 / 24, abs=1e-9)


def test_exact_perm_p_is_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5)
    y = rng.normal(size=5)
    res = exact_spearman_perm_p(x, y)
    assert res["n_perm"] == 120
    assert 0.0 < res["p_two_sided"] <= 1.0
    assert 0.0 < res["p_one_sided"] <= 1.0
    # two-sided p is at least the one-sided tail
    assert res["p_two_sided"] >= res["p_one_sided"] - 1e-9


# --------------------------------------------------------------------------- #
# assert_log1p_scale: catches a count-space control (the recurring bug).
# --------------------------------------------------------------------------- #
def test_assert_log1p_scale_passes_for_log1p_control():
    rng = np.random.default_rng(1)
    real = np.log1p(rng.integers(0, 100, size=(50, 8)).astype(np.float64))
    decoded = real + rng.normal(0, 0.3, size=real.shape)  # log1p-scale noise
    chk = assert_log1p_scale(decoded, real, "ok_control")
    assert chk["ratio"] < 4.0


def test_assert_log1p_scale_rejects_count_space_control():
    rng = np.random.default_rng(2)
    real = np.log1p(rng.integers(0, 100, size=(50, 8)).astype(np.float64))
    # a control accidentally built in COUNT space: values in the hundreds
    decoded = rng.integers(0, 500, size=real.shape).astype(np.float64)
    with pytest.raises(AssertionError):
        assert_log1p_scale(decoded, real, "count_space_bug")


# --------------------------------------------------------------------------- #
# build_armB_encoders: G_in-restricted shapes.
# --------------------------------------------------------------------------- #
def test_build_armB_encoders_are_gin_restricted(tmp_path):
    rng = np.random.default_rng(0)
    n, n_genes = 300, 40
    logX = rng.normal(size=(n, n_genes)).astype(np.float32)
    gin_idx = np.arange(0, 20)
    tr = np.zeros(n, dtype=bool)
    tr[:200] = True
    enc = build_armB_encoders(logX, gin_idx, tr, tmp_path, pca_dim=8, seed=0)
    assert set(enc) == {"pca_512", "randproj_512"}  # no FM caches in tmp_path
    for k, z in enc.items():
        assert z.shape == (n, 8), k
    # randproj must be a linear map of the G_in slice ONLY (independent of G_out)
    logX2 = logX.copy()
    logX2[:, 20:] = rng.normal(size=(n, 20))  # perturb G_out only
    enc2 = build_armB_encoders(logX2, gin_idx, tr, tmp_path, pca_dim=8, seed=0)
    assert np.allclose(enc["randproj_512"], enc2["randproj_512"], atol=1e-5)
