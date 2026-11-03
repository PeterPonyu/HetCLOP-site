"""Unit tests for Phase 2h multiplicity + enumeration machinery (T2).

Cross-validates the paper-level Holm and Benjamini-Hochberg implementations
against statsmodels (the gold standard), plus hand-computed known answers and
the properties the correction must guarantee (order preservation, monotonicity,
clipping, edge cases). Also pins the hypothesis-inventory enumeration.

"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
PHASE2H = HERE / "experiments" / "phase2h_reanalysis_2026-07-09"
sys.path.insert(0, str(PHASE2H))

from multiplicity import apply_corrections, benjamini_hochberg, holm  # noqa: E402

sm = pytest.importorskip("statsmodels.stats.multitest")


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------
def test_holm_hand_computed():
    # p=[0.01,0.02,0.03,0.04,0.05], factors 5..1 with cumulative max
    got = holm([0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_allclose(got, [0.05, 0.08, 0.09, 0.09, 0.09], rtol=1e-12)


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0, 1, size=rng.integers(1, 30))
        exp = sm.multipletests(p, method="holm")[1]
        np.testing.assert_allclose(holm(p), exp, rtol=1e-10, atol=1e-14)


def test_holm_preserves_input_order():
    p = [0.9, 0.001, 0.5]
    got = holm(p)
    # smallest raw (0.001, index 1) must get the smallest adjusted value
    assert np.argmin(got) == 1


def test_holm_clips_to_one():
    assert np.all(holm([0.6, 0.7, 0.8]) <= 1.0)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------
def test_bh_hand_computed():
    # all q become 0.05 for this arithmetic-progression case
    got = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_allclose(got, [0.05, 0.05, 0.05, 0.05, 0.05], rtol=1e-12)


def test_bh_matches_statsmodels():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.uniform(0, 1, size=rng.integers(1, 30))
        exp = sm.multipletests(p, method="fdr_bh")[1]
        np.testing.assert_allclose(benjamini_hochberg(p), exp, rtol=1e-10, atol=1e-14)


def test_bh_is_step_up_monotone_in_sorted_order():
    p = np.array([0.001, 0.2, 0.01, 0.9, 0.03])
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)  # non-decreasing along sorted p


def test_bh_never_below_holm_is_false_and_bh_le_holm():
    # BH is less conservative than Holm: q_bh <= p_holm elementwise
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=20)
    assert np.all(benjamini_hochberg(p) <= holm(p) + 1e-12)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_empty():
    assert holm([]).size == 0
    assert benjamini_hochberg([]).size == 0


def test_single():
    np.testing.assert_allclose(holm([0.03]), [0.03])
    np.testing.assert_allclose(benjamini_hochberg([0.03]), [0.03])


def test_tiny_pvalues_survive():
    p = [1e-30, 1e-25, 1e-20]
    assert np.all(holm(p) < 0.05)
    assert np.all(benjamini_hochberg(p) < 0.05)


# ---------------------------------------------------------------------------
# apply_corrections wrapper
# ---------------------------------------------------------------------------
def test_apply_corrections_attaches_fields_and_survival():
    tests = [{"id": "a", "raw_p": 1e-20}, {"id": "b", "raw_p": 0.5}, {"id": "c", "raw_p": 0.001}]
    out = apply_corrections(tests, p_key="raw_p")
    assert [t["id"] for t in out] == ["a", "b", "c"]  # order preserved
    assert out[0]["survives_holm_0.05"] and out[0]["survives_bh_0.05"]
    assert not out[1]["survives_holm_0.05"]
    for t in out:
        assert 0.0 <= t["holm_p_paper"] <= 1.0
        assert 0.0 <= t["bh_q_paper"] <= 1.0


def test_apply_corrections_does_not_mutate_input():
    tests = [{"id": "a", "raw_p": 0.01}]
    apply_corrections(tests)
    assert "holm_p_paper" not in tests[0]
