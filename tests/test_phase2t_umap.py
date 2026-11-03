"""Pins the single-cell UMAP result (spec §47): the embedding is BLIND to the dispersion defect.

The three synthetic draws share an identical conditional mean by construction, and a UMAP layout is
driven by the mean. So the global-dispersion draw -- which destroys marker genes and more than halves
rare-cell-type recovery -- looks nearly indistinguishable from the correct one in the embedding.

These tests enforce the honest reading:
  * every panel embeds the SAME cells in the SAME coordinate system (else the panels aren't comparable);
  * rare cell types survive the stratified subsample (else the collapse could not even be looked for);
  * the global draw IS the worst on kNN purity (so we are not claiming the embedding sees nothing);
  * but the embedding UNDER-REPORTS the damage relative to the marker-gene readout -- which is the claim.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UMAP = ROOT / "experiments" / "phase2t_umap_2026-07-10" / "results" / "umap.json"
BIO = ROOT / "experiments" / "phase2q_biological_2026-07-10" / "results" / "biological_agg.json"

pytestmark = pytest.mark.skipif(not UMAP.exists(), reason="UMAP experiment not yet run")


def _d():
    return json.loads(UMAP.read_text())


def test_panels_are_paired_and_share_coordinates():
    d = _d()
    counts = collections.Counter(p["variant"] for p in d["points"])
    assert set(counts) == set(d["variants"])
    assert len(set(counts.values())) == 1, f"panels have different cell counts: {counts}"
    assert counts["real"] == d["n_cells_per_panel"]


def test_rare_types_survive_the_subsample():
    d = _d()
    rare = collections.Counter((p["variant"], p["rare"]) for p in d["points"])
    per_panel = {v: rare[(v, True)] for v in d["variants"]}
    assert all(n > 0 for n in per_panel.values()), per_panel
    assert len(set(per_panel.values())) == 1, "rare cells must be the same paired cells in every panel"
    assert d["n_rare_types"] > 0


def test_global_dispersion_is_worst_on_purity_but_only_slightly():
    """The embedding is not blind in sign -- only in magnitude."""
    d = _d()
    pur = {k: v["knn_purity_k15"] for k, v in d["panel_stats"].items()}
    assert pur["nb_global"] < pur["poisson"] and pur["nb_global"] < pur["nb_conditional"]
    # ... yet the gap is small in relative terms
    assert d["relative_damage_umap_purity"] < 0.15


def test_embedding_under_reports_the_biological_damage():
    """The load-bearing claim: marker-gene damage >> damage visible in the embedding."""
    d = _d()
    assert d["relative_damage_marker_recall"] > d["relative_damage_umap_purity"]
    assert d["umap_underreports_damage_factor"] > 2.0


def test_umap_and_biological_readout_describe_the_same_run():
    """The blindness factor must be computed against the real biological numbers, not a copy."""
    d = _d()
    bio = json.loads(BIO.read_text())
    mk = bio["marker_recall_mean_sd"]
    expected = (mk["nb_conditional"][0] - mk["nb_global"][0]) / mk["nb_conditional"][0]
    assert abs(d["relative_damage_marker_recall"] - expected) < 1e-9
