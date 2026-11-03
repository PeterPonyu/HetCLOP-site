"""Pins the expression-cut sensitivity sweep (Tier 3, spec §45).

Closes the manuscript's own caveat that the expression cut (mean count >= 0.5) is "arbitrary".
F4's primary claim -- a dispersion-free Poisson draw beats the learned-theta negative binomial on
the expression-restricted |log2 r_var|, Arm A, PCA-512, mean held fixed -- must keep the same
DIRECTION and remain dataset-clustered significant at every informative cut.

The cut 0.0 (unrestricted) is deliberately excluded from the gate: F3 establishes that the
unrestricted statistic is degenerate (sparsity pins the per-gene median at 1.0). It is stored for
contrast, and this test asserts it is NOT used as support.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "phase2r_exprcut_2026-07-10" / "results" / "exprcut.json"

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="expression-cut sweep not yet run")


def _d():
    return json.loads(OUT.read_text())


def _informative(d):
    return [c for c in d["cuts"] if float(c) > 0.0]


def test_direction_stable_across_every_informative_cut():
    d = _d()
    assert d["direction_stable_across_informative_cuts"] is True
    for c in _informative(d):
        pw = d["per_cut"][str(c)]["poisson_win_fraction_groups"]
        assert pw > 0.5, f"cut {c}: Poisson no longer wins ({pw})"


def test_significance_stable_across_every_informative_cut():
    d = _d()
    assert d["significance_stable_across_informative_cuts"] is True
    for c in _informative(d):
        p = d["per_cut"][str(c)]["clustered_wilcoxon_p"]
        assert p < 0.05, f"cut {c}: clustered p = {p} not significant"


def test_paper_cut_is_within_the_swept_range_and_not_extremal():
    """The 0.5 used in the paper must not be the cut that maximises the effect."""
    d = _d()
    assert d["paper_cut"] in [float(c) for c in d["cuts"]]
    wins = {float(c): d["per_cut"][str(c)]["poisson_win_fraction_groups"] for c in _informative(d)}
    best = max(wins, key=lambda k: wins[k])
    assert best != d["paper_cut"] or len(wins) == 1, (
        f"the paper's cut {d['paper_cut']} is the single most favourable one: {wins}")


def test_unrestricted_cut_is_recorded_but_not_claimed():
    """cut 0.0 is the statistic F3 discredits; it must be recorded, excluded from the gates, and
    explicitly disclaimed as evidence. (Assert the substance, not a particular wording.)"""
    d = _d()
    assert "0.0" in d["per_cut"], "the unrestricted contrast must be recorded"
    assert 0.0 not in [float(c) for c in _informative(d)], "0.0 must not enter the claim gates"
    note = d["note"].lower()
    assert "not evidence" in note and ("discredit" in note or "degenerate" in note), (
        f"the note must disclaim cut 0.0 as evidence: {d['note']}")
