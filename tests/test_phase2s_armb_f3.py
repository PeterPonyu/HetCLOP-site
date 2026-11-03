"""Pins the Arm-B replication of F3 (Tier 3, spec §46).

Closes the manuscript's last stated limitation: "F3 runs in Arm A only ... its Arm-B replication
is left to future work."

F3's claim (i) -- deterministic decoding destroys within-type variance -- should be
geometry-INDEPENDENT (it follows from emitting E[Y|Z]). Claim (ii) -- Gaussian-on-log1p noise
under-inflates -- had never been measured in Arm B. Both replicate here, in every seed, on the
expression-restricted signed variance ratio.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "phase2s_armb_f3_2026-07-10" / "results" / "armb_f3.json"

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="Arm-B F3 replication not yet run")


def _d():
    return json.loads(OUT.read_text())


def test_arm_b_is_genuinely_circularity_broken():
    d = _d()
    assert d["g_in"] > 0 and d["g_out"] > 0
    assert d["arm"].startswith("B")


def test_deterministic_destroys_variance_in_arm_b():
    """Claim (i): geometry-independent. Every seed's median signed r_var is far below 1."""
    d = _d()
    assert d["claims"]["det_destroys_variance_every_seed"] is True
    for s, v in d["per_seed"].items():
        assert v["deterministic_signed_rvar_median"] < 0.5, f"seed {s}"
        assert v["deterministic_frac_groups_below_one"] > 0.95, f"seed {s}"


def test_gaussian_matched_noise_under_inflates_in_arm_b():
    """Claim (ii): the Gaussian arm under-inflates in Arm B, as it does in Arm A."""
    d = _d()
    assert d["claims"]["gaussian_under_inflates_every_seed"] is True
    for s, v in d["per_seed"].items():
        assert v["matched_noise_signed_rvar_median"] < 1.0, f"seed {s}"


def test_noise_restores_some_but_not_all_variance():
    """matched_noise sits strictly between the deterministic collapse and calibration."""
    d = _d()
    assert d["claims"]["noise_restores_some_variance_every_seed"] is True
    det = d["deterministic_median_across_seeds"]
    mat = d["matched_noise_median_across_seeds"]
    assert det < mat < 1.0, (det, mat)
