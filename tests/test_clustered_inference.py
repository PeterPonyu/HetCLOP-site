"""Unit tests for Phase 2f dataset-clustered inference machinery."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
PHASE2F = HERE / "experiments" / "phase2f_f4_resolution_2026-07-09"
sys.path.insert(0, str(PHASE2F))

from clustered_inference import (  # noqa: E402
    clustered_dataset_medians,
    dataset_median_diffs,
    dataset_signflip_permutation,
    naive_paired,
    paired_group_diffs,
    within_dataset_signflip_permutation,
)


def _mk_map(entries, metric="energy_distance"):
    """entries: list of (dataset, cell_type, a_value, b_value). Returns two maps."""
    a_map, b_map = {}, {}
    for ds, ct, av, bv in entries:
        a_map[(ds, ct)] = {metric: av}
        b_map[(ds, ct)] = {metric: bv}
    return a_map, b_map


def test_paired_group_diffs_aligns_and_filters_nonfinite():
    a_map, b_map = _mk_map([
        ("d1", "t1", 1.0, 0.5),
        ("d1", "t2", 2.0, 3.0),
        ("d2", "t1", np.nan, 1.0),  # dropped: non-finite a
    ])
    recs = paired_group_diffs(a_map, b_map, "energy_distance")
    assert len(recs) == 2
    assert recs[0] == {"dataset": "d1", "cell_type": "t1", "a": 1.0, "b": 0.5, "diff": 0.5}
    assert recs[1]["diff"] == pytest.approx(-1.0)


def test_paired_group_diffs_only_shared_keys():
    a_map = {("d1", "t1"): {"m": 1.0}, ("d1", "t2"): {"m": 2.0}}
    b_map = {("d1", "t1"): {"m": 0.0}}
    recs = paired_group_diffs(a_map, b_map, "m")
    assert len(recs) == 1
    assert recs[0]["cell_type"] == "t1"


def test_naive_paired_reproduces_scipy_wilcoxon():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    a = rng.normal(size=40) + 0.3
    b = rng.normal(size=40)
    recs = [{"dataset": "d", "cell_type": str(i), "a": float(a[i]),
             "b": float(b[i]), "diff": float(a[i] - b[i])} for i in range(40)]
    r = naive_paired(recs)
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    assert r["p_value"] == pytest.approx(p)
    assert r["wilcoxon_stat"] == pytest.approx(stat)
    assert r["n_groups"] == 40


def test_dataset_median_diffs_collapses_by_dataset():
    recs = [
        {"dataset": "d1", "cell_type": "t1", "diff": 1.0},
        {"dataset": "d1", "cell_type": "t2", "diff": 3.0},
        {"dataset": "d2", "cell_type": "t1", "diff": -2.0},
    ]
    ds, med = dataset_median_diffs(recs)
    assert ds == ["d1", "d2"]
    assert med[0] == pytest.approx(2.0)  # median(1,3)
    assert med[1] == pytest.approx(-2.0)


def test_clustered_collapses_pseudoreplication():
    # 3 datasets, 100 groups each, all diffs tiny positive. Naive sees 300
    # "independent" wins -> tiny p. Clustered sees only 3 units -> cannot be
    # significant at 0.05 (sign test on 3 all-positive gives p=0.25).
    recs = []
    for d in range(3):
        for t in range(100):
            recs.append({"dataset": f"d{d}", "cell_type": f"t{t}", "diff": 0.01})
    for r in recs:  # provide a/b for naive
        r["a"], r["b"] = r["diff"], 0.0
    naive = naive_paired(recs)
    clust = clustered_dataset_medians(recs)
    assert naive["p_value"] < 1e-10          # pseudoreplicated: hugely significant
    assert clust["n_datasets"] == 3
    assert clust["sign_test_p"] == pytest.approx(0.25)  # 3/3 one-sided sign, two-sided
    assert clust["sign_test_p"] > 0.05       # clustered: not significant


def test_clustered_sign_test_direction():
    # 40 datasets NB-better (diff<0), 10 datasets NB-worse (diff>0).
    recs = []
    for d in range(40):
        recs.append({"dataset": f"neg{d}", "cell_type": "t", "diff": -1.0,
                     "a": -1.0, "b": 0.0})
    for d in range(10):
        recs.append({"dataset": f"pos{d}", "cell_type": "t", "diff": 1.0,
                     "a": 1.0, "b": 0.0})
    clust = clustered_dataset_medians(recs)
    assert clust["n_datasets"] == 50
    assert clust["sign_test_n_nb_better"] == 40
    assert clust["nb_win_fraction_datasets"] == pytest.approx(0.8)
    assert clust["sign_test_p"] < 0.05       # 40/50 is significant


def test_signflip_permutation_null_is_nonsignificant():
    # symmetric diffs around 0 -> permutation p should be large.
    rng = np.random.default_rng(1)
    recs = []
    for d in range(50):
        v = float(rng.normal(scale=1.0))
        recs.append({"dataset": f"d{d}", "cell_type": "t", "diff": v,
                     "a": v, "b": 0.0})
    perm = dataset_signflip_permutation(recs, n_perm=2000, seed=0)
    assert perm["p_value"] > 0.05
    assert perm["n_datasets"] == 50


def test_signflip_permutation_strong_effect_is_significant():
    # all 50 datasets strongly negative -> permutation p at the floor.
    recs = [{"dataset": f"d{d}", "cell_type": "t", "diff": -1.0,
             "a": -1.0, "b": 0.0} for d in range(50)]
    perm = dataset_signflip_permutation(recs, n_perm=2000, seed=0)
    assert perm["p_value"] < 0.01
    # signed-rank sum of 50 equal negative values = -(1+..+50) = -1275
    assert perm["observed_stat"] == pytest.approx(-1275.0)


def test_signflip_permutation_robust_to_magnitude_outlier():
    # 45 datasets consistently favour one side by a SMALL margin; ONE dataset has
    # a huge opposite-sign magnitude. A mean-based statistic would be dominated by
    # the outlier (p ~ 0.5); the rank statistic must still detect the consensus.
    recs = [{"dataset": f"d{d}", "cell_type": "t", "diff": 0.01} for d in range(45)]
    recs.append({"dataset": "outlier", "cell_type": "t", "diff": -50.0})
    perm = dataset_signflip_permutation(recs, n_perm=5000, seed=0)
    assert perm["p_value"] < 0.05  # 45/46 positive by rank -> significant
    clust = clustered_dataset_medians(recs)
    assert clust["sign_test_n_nb_better"] == 1  # only the outlier is NB-better


def test_signflip_permutation_p_is_bounded():
    recs = [{"dataset": f"d{d}", "cell_type": "t", "diff": float(d - 5),
             "a": 0.0, "b": 0.0} for d in range(11)]
    perm = dataset_signflip_permutation(recs, n_perm=500, seed=3)
    assert 0.0 < perm["p_value"] <= 1.0


def test_within_dataset_permutation_runs():
    rng = np.random.default_rng(2)
    recs = []
    for d in range(20):
        for t in range(5):
            v = float(rng.normal())
            recs.append({"dataset": f"d{d}", "cell_type": f"t{t}", "diff": v})
    perm = within_dataset_signflip_permutation(recs, n_perm=1000, seed=0)
    assert 0.0 < perm["p_value"] <= 1.0
