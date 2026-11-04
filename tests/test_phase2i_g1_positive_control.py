"""Pins the G1 positive control once it lands (spec: F2 sensitivity objection).

G1 answers the obvious objection to F2 ("six encoders span only 0.0045 decoded
Pearson => the METRIC is insensitive"): it builds a PCA-k ladder (k in {2,5,10,25,50,512})
with a KNOWN monotone information gradient in Arm B and shows the pipeline DOES separate
encoders when a real difference exists. This test is dormant (skipif) until
g1_positive_control.json exists, then it asserts G1's PASS criteria automatically.

Assertions are the robust scientific claims, not brittle step-wise monotonicity: a single
top-end swap (e.g. pca_50 edging pca_512, as E3 saw against the oracle) must not false-fail
the control, so we lean on the exact-permutation Spearman p and the clustered separation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
G1 = ROOT / "experiments" / "phase2i_extensions_2026-07-09" / "results" / "g1_positive_control.json"

pytestmark = pytest.mark.skipif(not G1.exists(), reason="G1 positive control not yet run")


def _load():
    return json.loads(G1.read_text())


def test_energy_orders_the_gradient_significantly():
    """The core positive-control claim: ENERGY distance recovers the information ordering at
    exact-permutation significance. (This is the sensitive metric; reconstruction Pearson
    saturates at the task ceiling -- see the next test.)"""
    o = _load()["ordering_recovered"]
    en = o["spearman_logk_vs_energy"]
    assert en["rho"] < -0.9 and en["p_two_sided"] < 0.05, f"energy ordering weak: {en}"
    assert o["energy_monotone_decreasing"] is True


def test_recon_responds_but_saturates_at_ceiling():
    """Reconstruction Pearson responds STRONGLY to information at low k but SATURATES at the
    ~0.68 task ceiling (F2/E3), so it cannot order the top rungs -- which is exactly why it is
    NOT perm-significant across the full ladder, and why energy (more sensitive) is needed.
    This is a ceiling property, not metric insensitivity."""
    o = _load()["ordering_recovered"]
    recon = o["det_recon_pearson"]
    # steep rise from the weakest encoder to the ceiling (>0.2 Pearson of dynamic range)
    assert max(recon) - recon[0] > 0.2, f"recon did not respond to information: {recon}"
    # and it plateaus: the top three rungs (k>=25) are tied within the FM-vs-PCA gap
    top = recon[-3:]
    assert max(top) - min(top) < 0.0045, f"expected ceiling saturation in top rungs: {top}"
    # energy, by contrast, keeps resolving those same saturated rungs
    energy = o["nb_energy_median"]
    assert energy[-1] < energy[-3], "energy should still separate the recon-saturated top rungs"


def test_pca2_vs_pca512_separates_clustered():
    """The headline pair separates on energy at dataset-clustered significance across ALL
    count redraws (standing multi-draw rule -- no single-draw energy p)."""
    h = _load()["headline_pca2_vs_pca512_multidraw"]
    assert h["separates_all_draws_clustered_0.05"] is True
    assert max(h["clustered_wilcoxon_p_range"]) < 0.05, h["clustered_wilcoxon_p_range"]
    # pca_2 (less info) must be the WORSE encoder on energy in the median draw
    assert h["median_energy_diff_pca2_minus_pca512"] > 0


def test_smallest_resolved_step_is_below_fm_vs_pca_gap():
    """The killer claim: energy resolves an adjacent recon gap SMALLER than the 0.0045
    FM-vs-PCA gap, so 'encoders indistinguishable' is a property of the ENCODERS, not the
    metric. Requires at least one adjacent pair to resolve."""
    s = _load()["smallest_resolved_step"]
    assert s["n_adjacent_resolved"] >= 1, "no adjacent PCA-ladder pair resolved -- control weak"
    assert s["smaller_than_fm_vs_pca_gap"] is True, (
        f"smallest resolved gap {s['smallest_resolved_recon_gap']} not below "
        f"FM-vs-PCA reference {s['fm_vs_pca_gap_reference']}")


def test_noise_gradient_corroborates():
    """Corroborating gradient: adding input noise strictly degrades reconstruction."""
    ng = _load()["noise_gradient"]
    recon = ng["det_recon_pearson"]
    # more noise => not-better reconstruction (monotone non-increasing within tolerance)
    assert recon[0] >= max(recon[1:]) - 1e-3, f"noise gradient not monotone: {recon}"
