"""Unit tests for Phase 2h leave-two-out enumeration machinery (T1).

Verifies the dataset-dropping, the Holm-over-encoders family helper, the
inventory enumeration, and (as an integration guard) that the persisted results
reproduce the sanity assert and leave-two-out invariants. Does not retrain.

"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
PHASE2H = HERE / "experiments" / "phase2h_reanalysis_2026-07-09"
sys.path.insert(0, str(PHASE2H))

from leave_two_out import (  # noqa: E402
    GATE_ENCODERS,
    MISNAMED_MOUSE_DATASETS,
    drop_datasets,
    holm_family_over_encoders,
    load_per_comparison,
)

P2F = HERE / "experiments/phase2f_f4_resolution_2026-07-09/results/phase2f.json"
RESULTS = PHASE2H / "results"


def test_drop_datasets_removes_only_targets():
    recs = [
        {"dataset": "GSE222002_TcellsHmCancer", "cell_type": "T", "a": 1, "b": 0, "diff": 1},
        {"dataset": "GSE168181_BreastHmCancer", "cell_type": "B", "a": 1, "b": 0, "diff": 1},
        {"dataset": "GSE_keep", "cell_type": "X", "a": 1, "b": 0, "diff": 1},
    ]
    out = drop_datasets(recs, MISNAMED_MOUSE_DATASETS)
    assert len(out) == 1 and out[0]["dataset"] == "GSE_keep"


def test_holm_family_over_two_encoders():
    fam = {"pca_512": 0.006, "randproj_512": 1e-20}
    hp = holm_family_over_encoders(fam)
    # larger raw (pca) gets *1; smaller (randproj) gets *2
    assert hp["pca_512"] == pytest.approx(0.006)
    assert hp["randproj_512"] == pytest.approx(2e-20)


def test_load_per_comparison_shape_and_target_presence():
    idx, stored = load_per_comparison(P2F)
    # 4 encoders x 3 seeds x 2 metrics = 24
    assert len(idx) == 24
    recs = idx[("pca_512", 42, "energy_distance")]
    assert len(recs) == 314
    datasets = {r["dataset"] for r in recs}
    assert len(datasets) == 50
    for t in MISNAMED_MOUSE_DATASETS:
        assert t in datasets


def test_leave_two_out_drops_exactly_two_datasets():
    idx, _ = load_per_comparison(P2F)
    recs = idx[("pca_512", 42, "energy_distance")]
    full = {r["dataset"] for r in recs}
    lto = {r["dataset"] for r in drop_datasets(recs, MISNAMED_MOUSE_DATASETS)}
    assert len(full) - len(lto) == 2


# ---- integration guards on persisted results (require run_phase2h.py first) ----
def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet; run run_phase2h.py")
    return json.load(open(p))


def test_sanity_assert_passed():
    t1 = _load("t1_leave_two_out.json")
    assert t1["sanity_assert"]["status"] == "PASS"
    assert t1["sanity_assert"]["n_checks"] == 120


def test_all_gate_legs_stay_significant_under_leave_two_out():
    t1 = _load("t1_leave_two_out.json")
    for name, leg in t1["per_leg_summary"].items():
        l = leg["leave2out48"]
        assert l["n_datasets"] == 48
        # every gate leg remains Holm-significant on all 3 seeds after dropping 2 datasets
        assert l["seeds_naive_holm_sig_0.05"] == 3, name


def test_inventory_size_and_f4_leg_survives():
    t2 = _load("t2_multiplicity.json")
    assert t2["inventory_size"] == 25
    assert t2["f4_carrying_leg"]["survives_paper_level"] is True
