"""Pins the leave-two-out robustness result now that it is cited in the manuscript (spec §52).

The Limitations section claims the two species-misnamed (mouse) datasets do not affect any
conclusion. That claim is backed by this experiment: dropping both datasets (50 -> 48) must leave
every reported leg dataset-clustered-significant in all three seeds, and the bit-level sanity
assertion must pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
L2O = ROOT / "experiments" / "phase2h_reanalysis_2026-07-09" / "results" / "t1_leave_two_out.json"

pytestmark = pytest.mark.skipif(not L2O.exists(), reason="leave-two-out result absent")


def _d():
    return json.loads(L2O.read_text())


def test_dropping_both_datasets_keeps_every_leg_significant():
    d = _d()
    for leg, splits in d["per_leg_summary"].items():
        for split in ("full50", "leave2out48"):
            assert splits[split]["seeds_clustered_wilcoxon_sig_0.05"] == 3, f"{leg}/{split}"


def test_leave_two_out_actually_drops_two_datasets():
    d = _d()
    any_leg = next(iter(d["per_leg_summary"].values()))
    assert any_leg["full50"]["n_datasets"] == 50
    assert any_leg["leave2out48"]["n_datasets"] == 48


def test_primary_claim_stays_significant_after_drop():
    """The F4 primary leg (pca_512 r_var) worst clustered p must remain well below 0.05 after the drop."""
    d = _d()
    leg = d["per_leg_summary"]["r_var_abs_log2_expr__pca_512"]
    assert leg["full50"]["max_clustered_wilcoxon_p"] < 0.05
    assert leg["leave2out48"]["max_clustered_wilcoxon_p"] < 0.05


def test_sanity_assertion_passed():
    d = _d()
    assert d["sanity_assert"]["status"] == "PASS"
    assert d["sanity_assert"]["n_checks"] > 0
