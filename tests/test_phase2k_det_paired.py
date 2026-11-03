"""Pins the paired deterministic-vs-stochastic re-analysis (defect #16, paper.tex:436).

The point of these tests is narrow and load-bearing: the sign convention must be right
(energy is lower-is-better; diff = det - opponent; diff>0 => deterministic worse), and the
falsified claim must stay falsified against the real stored numbers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "phase2k_det_paired_2026-07-09" / "run_det_paired.py"
OUT = ROOT / "experiments" / "phase2k_det_paired_2026-07-09" / "results" / "det_paired.json"
SRC = ROOT / "experiments" / "phase2b_stochastic_2026-07-09" / "results" / "phase2b_v3_paired.json"

pytestmark = pytest.mark.skipif(not SRC.exists(), reason="phase2b source JSON absent")


def _load():
    if not OUT.exists():
        subprocess.run([sys.executable, str(SCRIPT)], check=True, capture_output=True)
    return json.loads(OUT.read_text())


def test_claim_loses_every_group_is_false():
    d = _load()
    assert d["claim_is_true"] is False
    # at least one group is won for at least one contrast on every encoder
    for enc, opps in d["groups_the_deterministic_decoder_wins"].items():
        assert max(opps.values()) > 0, f"{enc}: deterministic wins zero groups everywhere"


def test_win_fraction_matches_direct_recompute():
    """Independent recomputation from the raw per-group arrays -- the script's win
    fraction must equal a from-scratch count, or the sign convention is wrong."""
    payload = json.loads(SRC.read_text())
    d = _load()
    for enc in payload["per_group"]:
        v = payload["per_group"][enc]["variants"]
        det = np.asarray(v["deterministic"]["energy_distance"], float)
        for opp in ("matched_noise", "matched_noise_per_type", "stochastic_betanll"):
            o = np.asarray(v[opp]["energy_distance"], float)
            m = np.isfinite(det) & np.isfinite(o)
            expect = float(np.mean(det[m] < o[m]))
            got = d["results"][enc][opp]["det_win_fraction_groups"]
            assert abs(expect - got) < 1e-9, f"{enc}/{opp}: {expect} != {got}"


def test_deterministic_is_directionally_worse_everywhere():
    """Every median paired diff is positive (deterministic worse), and every contrast
    is significant at 0.05 under dataset-clustered inference -- the TRUE, defensible claim."""
    d = _load()
    for enc, opps in d["results"].items():
        for opp, r in opps.items():
            assert r["median_paired_diff_det_minus_opp"] > 0, f"{enc}/{opp} not worse"
            assert r["clustered_wilcoxon_p"] < 0.05, f"{enc}/{opp} n.s. clustered"


def test_weakest_contrast_is_not_overclaimable():
    """Guards the manuscript against reusing a blanket p<0.001: the weakest contrast
    (pca_512 vs matched_noise) is only ~0.03, so no sub-0.001 blanket is licensed."""
    d = _load()
    assert d["max_clustered_wilcoxon_p"] > 1e-3
