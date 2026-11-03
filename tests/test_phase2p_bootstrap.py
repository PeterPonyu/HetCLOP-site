"""Pins the dataset-level bootstrap CI for F4's primary claim (Tier 3, spec §42).

Closes the 'seeds vary decoder init only' limitation: a cluster bootstrap over the 50 source
datasets must show the Poisson-beats-NB majority is robust (95% CI excludes 0.5) in every seed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "phase2p_bootstrap_2026-07-10" / "run_bootstrap.py"
OUT = ROOT / "experiments" / "phase2p_bootstrap_2026-07-10" / "results" / "bootstrap_ci.json"
SRC = ROOT / "experiments" / "phase2e_expressed_rvar_2026-07-09" / "results" / "e5_expressed_rvar.json"

pytestmark = pytest.mark.skipif(not SRC.exists(), reason="e5 source JSON absent")


def _load():
    if not OUT.exists():
        subprocess.run([sys.executable, str(SCRIPT)], check=True, capture_output=True)
    return json.loads(OUT.read_text())


def test_headline_robust_to_dataset_resampling():
    d = _load()
    assert d["robust"] is True
    for s, v in d["per_seed"].items():
        lo, hi = v["poisson_win_ci95"]
        assert lo > 0.5, f"seed {s}: 95% CI lower bound {lo} does not exclude 0.5"
        assert v["poisson_win_point"] > 0.5


def test_pooled_ci_excludes_half():
    d = _load()
    lo, hi = d["pooled_over_seeds"]["poisson_win_ci95"]
    assert lo > 0.5 and hi > lo
    # the point estimate must sit inside its own CI (sanity on the percentile computation)
    pt = d["pooled_over_seeds"]["poisson_win_point"]
    assert lo <= pt <= hi
