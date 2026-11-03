"""Pins the graded-conditionality result (Tier 2, spec §42): scVI gene / gene-label / gene-cell.

Factor (b) is not binary. As scVI's dispersion becomes more conditional
(gene = global -> gene-label = per cell type -> gene-cell = per cell), the dispersion-free
Poisson draw's win-fraction must fall MONOTONICALLY on both the expression-restricted |log2 r_var|
and the energy distance, across all three seeds.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GL = ROOT / "experiments" / "phase2o_genelabel_2026-07-10" / "results" / "scvi_genelabel.json"
SV = ROOT / "experiments" / "phase2j_external_2026-07-09" / "results" / "scvi.json"

pytestmark = pytest.mark.skipif(not (GL.exists() and SV.exists()), reason="gene-label/scvi results absent")


def _mean_poiswin(results_block, metric):
    return st.mean(r["contrasts"][metric]["poisson_win_fraction_groups"] for r in results_block.values())


def _axis(metric):
    gl = json.loads(GL.read_text())["results"]
    sv = json.loads(SV.read_text())["results"]
    return (_mean_poiswin(sv["gene"], metric),
            _mean_poiswin(gl, metric),
            _mean_poiswin(sv["gene-cell"], metric))


def test_conditionality_is_monotone_on_both_metrics():
    for metric in ("rvar_expressed", "energy"):
        g, glab, gc = _axis(metric)
        assert g > glab > gc, f"{metric}: not monotone (gene {g:.3f} > label {glab:.3f} > cell {gc:.3f})"


def test_gene_label_is_the_energy_tipping_point():
    """On energy, gene-label already flips the winner to NB (Poisson wins < 0.5), significantly
    in every seed -- cell-type conditioning is enough for the holistic metric."""
    gl = json.loads(GL.read_text())["results"]
    for s, r in gl.items():
        e = r["contrasts"]["energy"]
        assert e["poisson_win_fraction_groups"] < 0.5, f"seed {s}: energy not flipped to NB"
        assert e["clustered_wilcoxon_p"] < 0.05


def test_gene_label_reproducible_across_seeds():
    gl = json.loads(GL.read_text())["results"]
    rv = [r["contrasts"]["rvar_expressed"]["poisson_win_fraction_groups"] for r in gl.values()]
    assert len(rv) == 3 and (max(rv) - min(rv)) < 0.1, f"gene-label rvar win not reproducible: {rv}"
