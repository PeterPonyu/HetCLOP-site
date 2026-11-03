"""Pins the Tier-1 biological readout and, crucially, the claims it does NOT license (spec §43).

Mean-fixed design (one model's own mu; only the dispersion parameterisation varies), 3 seeds.

What the data supports:
  * a GLOBAL dispersion is worst on BOTH biological readouts in every seed, despite having the
    BEST marginal variance calibration (signed r_var 1.036 in the theta probe) -- a placement
    effect, not a magnitude effect;
  * a CONDITIONAL dispersion beats Poisson on marker-gene recovery in every seed.

What it does NOT support, and these tests enforce that the wording stays honest:
  * conditional does NOT beat Poisson on TSTR: the gap is smaller than the across-seed sd, so the
    manuscript must state a tie. If a future run genuinely separates them, the last test fails and
    forces the claim to be revisited rather than silently upgraded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "experiments" / "phase2q_biological_2026-07-10"
AGG = DIR / "results" / "biological_agg.json"
PROBE = ROOT / "experiments" / "phase2j_external_2026-07-09" / "results" / "scvi_theta_probe.json"

pytestmark = pytest.mark.skipif(
    not (DIR / "results" / "biological_seed2.json").exists(),
    reason="3-seed biological readout not present",
)


def _agg():
    if not AGG.exists():
        subprocess.run([sys.executable, str(DIR / "aggregate.py")], check=True, capture_output=True)
    return json.loads(AGG.read_text())


def test_global_dispersion_is_biologically_worst():
    d = _agg()
    assert d["claims"]["global_is_worst_on_both_readouts_every_seed"] is True
    mk = d["marker_recall_mean_sd"]
    assert mk["nb_global"][0] < mk["poisson"][0] and mk["nb_global"][0] < mk["nb_conditional"][0]
    rare = d["tstr_rare_f1_mean_sd"]
    # global more than halves rare-cell-type recovery relative to both others
    assert rare["nb_global"][0] < 0.5 * rare["poisson"][0]
    assert rare["nb_global"][0] < 0.5 * rare["nb_conditional"][0]


def test_conditional_beats_poisson_on_marker_recovery():
    d = _agg()
    assert d["claims"]["conditional_beats_poisson_on_marker_every_seed"] is True
    p = d["marker_paired_worst_across_seeds"]["nb_conditional_vs_poisson"]
    assert p["worst_wilcoxon_p"] < 0.05 and p["mean_diff"] > 0


def test_conditional_beats_global_on_every_cell_type():
    d = _agg()
    p = d["marker_paired_worst_across_seeds"]["nb_conditional_vs_nb_global"]
    assert p["min_win_fraction_types"] > 0.95, "conditional should beat global on nearly every type"
    assert p["worst_wilcoxon_p"] < 1e-6


def test_tstr_conditional_vs_poisson_is_a_tie_not_a_win():
    """Guards the honest wording: the TSTR gap must stay within across-seed noise. If it ever
    separates, this fails and the claim must be re-examined -- never silently upgraded."""
    d = _agg()
    assert d["claims"]["conditional_beats_poisson_on_tstr_beyond_noise"] is False
    assert abs(d["tstr_cond_minus_pois_macro"]) <= d["tstr_across_seed_sd"]


def test_global_is_worst_despite_best_marginal_calibration():
    """The placement-not-magnitude argument: under the same mean-fixed design the theta probe
    shows nb_global has the signed r_var CLOSEST to 1.0 of the three draws."""
    pr = list(json.loads(PROBE.read_text())["results"].values())[0]["direction"]
    dev = {k: abs(pr[k]["r_var_signed_expr_median"] - 1.0)
           for k in ("poisson_on_mean", "nb_global_theta", "nb_amortized")}
    assert dev["nb_global_theta"] == min(dev.values()), (
        f"nb_global should be best-calibrated on the marginal statistic: {dev}")
    # and yet it is biologically worst
    assert _agg()["claims"]["global_is_worst_on_both_readouts_every_seed"] is True
