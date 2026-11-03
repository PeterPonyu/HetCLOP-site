"""Pins the theta-conditionality probe -- the decisive test of law factor (b) (spec §36).

The probe holds mu IDENTICAL across three arms drawn from the trained scVI gene-cell model
(poisson(mu), NB(mu, global theta), NB(mu, amortized theta_ic)), so any difference is due to
the DISPERSION parameterization alone -- the worse-mean confound cannot leak. Registered
CONFIRM prediction (§27): amortized beats poisson, but a global (median-collapse) theta does
NOT -- conditionality does the work. This test pins that outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "experiments" / "phase2j_external_2026-07-09" / "results" / "scvi_theta_probe.json"

pytestmark = pytest.mark.skipif(not PROBE.exists(), reason="theta probe not yet run")


def _seed0():
    d = json.loads(PROBE.read_text())
    r = d["results"]
    return r["0"] if "0" in r else list(r.values())[0]


def _all_seeds():
    return list(json.loads(PROBE.read_text())["results"].values())


def _c(r, key):
    return r["contrasts"][key]


def test_verdict_holds_in_every_seed():
    """The mean-fixed ablation is now run over multiple re-fits; factor (b) must confirm in ALL of
    them: global loses to Poisson, amortised beats Poisson, amortised beats the global collapse."""
    seeds = _all_seeds()
    assert len(seeds) >= 3, f"expected >=3 seeds, got {len(seeds)}"
    for i, r in enumerate(seeds):
        assert _c(r, "nb_global_vs_poisson__energy")["median_paired_diff_a_minus_b"] > 0, f"seed {i}"
        assert _c(r, "nb_amortized_vs_poisson__energy")["median_paired_diff_a_minus_b"] < 0, f"seed {i}"
        assert _c(r, "nb_amortized_vs_nb_global__energy")["median_paired_diff_a_minus_b"] < 0, f"seed {i}"
        assert r["mu_identical_across_arms"] is True, f"seed {i}"


def test_mu_identical_across_arms():
    """The confound control: mu is the SAME in all three arms (poisson/global/amortized),
    so the gate 'failing' on the worse mean is irrelevant to this contrast."""
    r = _seed0()
    assert r["mu_identical_across_arms"] is True


def test_poisson_leaves_a_deficit():
    """Factor (a): the Poisson-on-mean draw under-inflates expressed-gene variance."""
    r = _seed0()
    assert r["direction"]["poisson_on_mean"]["r_var_signed_expr_median"] < 0.95


def test_amortized_theta_beats_poisson():
    """(iii) beats (i): the conditional per-cell theta calibrates and wins."""
    r = _seed0()
    for metric in ("rvar_expressed", "rvar_topk", "energy"):
        c = _c(r, f"nb_amortized_vs_poisson__{metric}")
        # a = nb_amortized, b = poisson; a wins => median diff (a-b) < 0, clustered sig
        assert c["median_paired_diff_a_minus_b"] < 0, f"amortized not better on {metric}: {c}"
        assert c["clustered_wilcoxon_p"] < 0.05, f"amortized-vs-poisson n.s. on {metric}"


def test_global_theta_does_NOT_beat_poisson():
    """(ii) does NOT beat (i): with the SAME mu and the SAME deficit, a global (median-collapse)
    theta is WORSE than Poisson -- filling the deficit on average backfires per-group. This is
    the crux that isolates conditionality from the deficit."""
    r = _seed0()
    for metric in ("rvar_expressed", "rvar_topk", "energy"):
        c = _c(r, f"nb_global_vs_poisson__{metric}")
        # a = nb_global, b = poisson; global WORSE => median diff (a-b) > 0, Poisson wins
        assert c["median_paired_diff_a_minus_b"] > 0, f"global unexpectedly not worse on {metric}: {c}"
        assert c["clustered_wilcoxon_p"] < 0.05, f"global-vs-poisson n.s. on {metric}"
        assert c["b_win_fraction_groups"] > 0.5, f"Poisson should win most groups vs global on {metric}"


def test_amortized_beats_global_decisively():
    """The direct contrast: conditional theta beats the global collapse on every metric,
    at the Wilcoxon floor -- conditionality is what calibrates, given the deficit."""
    r = _seed0()
    for metric in ("rvar_expressed", "rvar_topk", "energy"):
        c = _c(r, f"nb_amortized_vs_nb_global__{metric}")
        assert c["median_paired_diff_a_minus_b"] < 0, f"amortized not better than global on {metric}"
        assert c["clustered_wilcoxon_p"] < 1e-6, f"amortized-vs-global not decisive on {metric}"


def test_matches_synthetic_forecast():
    """The real-weight result must reproduce the synthetic phase diagram's mixed-regime
    forecast (global does NOT beat Poisson; conditional wins). Both point the same way."""
    r = _seed0()
    global_worse = _c(r, "nb_global_vs_poisson__energy")["median_paired_diff_a_minus_b"] > 0
    amort_wins = _c(r, "nb_amortized_vs_poisson__energy")["median_paired_diff_a_minus_b"] < 0
    assert global_worse and amort_wins
