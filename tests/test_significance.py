import numpy as np
import pytest

from src.diagnostics.significance import permutation_null, bootstrap_ci
from src.diagnostics.tightness import tightness


def _make_well_separated_clusters():
    # 2 types x 20 cells, tight within each type, far apart between types —
    # deterministic grid offsets, no randomness needed for the data itself.
    offsets = np.array([[i * 0.01, 0.0] for i in range(20)])
    type_a = offsets + np.array([0.0, 0.0])
    type_b = offsets + np.array([10.0, 0.0])
    X = np.vstack([type_a, type_b])
    labels = np.array(["A"] * 20 + ["B"] * 20)
    return X, labels


def test_permutation_null_shape_and_direction():
    X, labels = _make_well_separated_clusters()
    observed = tightness(X, labels)
    null_vals = permutation_null(tightness, X, labels, n_perm=200, seed=0)
    assert null_vals.shape == (200,)
    # Real grouping is far tighter (within << between) than any shuffled
    # grouping (which mixes the two well-separated clusters), so the
    # observed statistic must sit at the extreme low tail of the null.
    assert observed < np.percentile(null_vals, 5)


def test_bootstrap_ci_brackets_point_estimate_and_is_ordered():
    X, labels = _make_well_separated_clusters()
    observed = tightness(X, labels)
    lo, hi = bootstrap_ci(tightness, X, labels, n_boot=200, seed=0)
    assert lo <= hi
    assert 0.0 <= lo <= observed * 3 + 1e-6
    assert hi >= observed * 0.3
