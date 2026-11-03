"""Pins the synthetic 2x2 phase-diagram experiment for the dispersion law (spec §29).

The experiment demonstrates, with known ground truth, that extra dispersion helps iff
(a) Poisson-on-mean leaves a deficit AND (b) theta is conditional enough -- isolating
the contestable factor (b) via a homogeneous-vs-heterogeneous true-theta contrast. These
tests re-run the grid once and assert the five CORE predictions hold in every one of five
seeds, plus the load-bearing calibration ordering from the stored per-regime levels. The two
seed-sensitive items are checked as reported (non-asserted) observations.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "phase2m_lawgrid_2026-07-09" / "run_lawgrid.py"
OUT = ROOT / "experiments" / "phase2m_lawgrid_2026-07-09" / "results" / "lawgrid.json"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="lawgrid script absent")


@functools.lru_cache(maxsize=1)
def _load():
    # deterministic (fixed internal seeds); run the multi-seed grid ONCE per test session
    # and memoize -- each test re-invoking it would multiply the 5-seed cost needlessly.
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, f"grid script must exit 0 (all predictions hold):\n{r.stdout}\n{r.stderr}"
    return json.loads(OUT.read_text())


def test_all_core_predictions_hold():
    d = _load()
    assert d["all_predictions_hold"] is True
    for name, ok in d["predictions_checked"].items():
        assert ok, f"prediction failed: {name}"


def test_core_predictions_robust_across_seeds():
    """The load-bearing factor-(a)/(b) claims must hold in EVERY seed, not just one draw."""
    d = _load()
    rob = d["seed_robustness"]
    assert rob["n_seeds"] >= 5
    assert rob["all_core_predictions_hold_every_seed"] is True
    for name, ok in rob["per_prediction_holds_all_seeds"].items():
        assert ok, f"core prediction not seed-robust: {name}"


def test_no_deficit_all_three_calibrated_and_tied():
    """No deficit => Poisson is calibrated and all three models tie (so which is strictly
    'best' is a coin flip -- asserted only as a near-tie, never as a strict argmin)."""
    d = _load()
    lv = d["results"]["no_deficit"]["levels"]
    assert 0.95 <= lv["poisson"]["r_var_signed_expr_median"] <= 1.05
    errs = [lv[m]["r_var_abs_log2_expr_median"] for m in ("poisson", "nb_global", "nb_conditional")]
    assert max(errs) - min(errs) < 0.02, f"expected a 3-way tie, got {errs}"


def test_homogeneous_global_equals_conditional():
    """The crux of factor (b): with a homogeneous true theta a global value calibrates
    every group, so the conditional oracle buys nothing."""
    d = _load()
    lv = d["results"]["deficit_homog"]["levels"]
    g = lv["nb_global"]["r_var_abs_log2_expr_median"]
    c = lv["nb_conditional"]["r_var_abs_log2_expr_median"]
    assert abs(g - c) < 0.02, (g, c)


def test_heterogeneous_conditional_strictly_better():
    """With heterogeneous true theta the conditional oracle strictly beats the global
    ablation -- conditionality earns its keep only here."""
    d = _load()
    lv = d["results"]["deficit_hetero"]["levels"]
    g = lv["nb_global"]["r_var_abs_log2_expr_median"]
    c = lv["nb_conditional"]["r_var_abs_log2_expr_median"]
    assert c < g - 0.05, (c, g)
    v = d["verdicts"]["deficit_hetero"]
    assert v["conditional_beats_global"] and v["conditional_vs_global_clustered_p"] < 0.05


def test_mixed_conditional_wins_and_global_contest_is_borderline():
    """Bimodal true theta (near-Poisson + overdispersed cell types). The conditional oracle
    winning is the ROBUST claim (asserted). Whether the global ablation beats Poisson is
    deliberately BORDERLINE and seed-sensitive -- it is a reported observation, and its
    non-unanimity across seeds is itself the finding (it mirrors the near-noise-floor scVI
    `gene` paired energy leg), so we assert it is tracked as such, not that it is always
    False."""
    d = _load()
    assert d["verdicts"]["deficit_mixed"]["conditional_beats_global"] is True
    counts = d["seed_robustness"]["observation_seed_counts_out_of_n"]
    n = d["seed_robustness"]["n_seeds"]
    # the global-vs-Poisson observation must be genuinely seed-sensitive, not unanimous
    assert 0 < counts["mixed__global_theta_does_not_beat_poisson"] < n
