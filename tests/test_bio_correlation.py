import numpy as np
import pytest

from src.diagnostics.bio_correlation import bio_correlation_control


def _six_point_type(base):
    return base + np.array(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    )


def test_identity_embedding_is_near_perfectly_correlated():
    raw_x = np.vstack([_six_point_type(np.array([[0.0]])), _six_point_type(np.array([[100.0]]))])
    labels = ["A"] * 6 + ["B"] * 6
    embed_x = raw_x.copy()
    r_identity = bio_correlation_control(raw_x, embed_x, labels)
    assert r_identity == pytest.approx(1.0, abs=1e-6)


def test_scrambled_embedding_correlates_much_less_than_identity():
    raw_x = np.vstack([_six_point_type(np.array([[0.0]])), _six_point_type(np.array([[100.0]]))])
    labels = ["A"] * 6 + ["B"] * 6
    r_identity = bio_correlation_control(raw_x, raw_x.copy(), labels)
    # Scramble the within-type order so pairwise-distance structure no
    # longer matches the raw space, while keeping within-type variance
    # nonzero (required by the function's variance guard).
    scrambled = raw_x.copy()
    scrambled[0:6] = raw_x[0:6][[0, 3, 1, 4, 2, 5]]
    scrambled[6:12] = raw_x[6:12][[5, 2, 4, 0, 3, 1]]
    r_scrambled = bio_correlation_control(raw_x, scrambled, labels)
    assert r_scrambled < r_identity - 0.3


def test_raises_when_no_type_has_enough_cells():
    raw_x = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        bio_correlation_control(raw_x, raw_x.copy(), ["A", "A"], min_cells_per_type=4)


def test_zero_variance_type_is_excluded_not_nan():
    """Type with zero pairwise-distance variance (all cells identical) should be
    silently excluded by the guard, not cause nan to propagate through mean.
    """
    # Type A: 4 identical cells → all pairwise distances = 0 → zero variance
    # Should be skipped by the guard (line 25-26 in bio_correlation.py)
    type_a = np.array([[0.0], [0.0], [0.0], [0.0]])

    # Type B: 6 cells with normal variance (distinguishable values)
    type_b = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])

    raw_x = np.vstack([type_a, type_b])
    labels = ["A"] * 4 + ["B"] * 6

    # embed_x = raw_x for both → if guard works correctly, only type B is
    # included in per_type_r, giving r=1.0 (perfect correlation of distances)
    embed_x = raw_x.copy()

    r = bio_correlation_control(raw_x, embed_x, labels)
    # If type A were included as nan, np.mean([nan, 1.0]) = nan (not ~1.0)
    # Proof that type A was excluded by the guard, not silently nan
    assert r == pytest.approx(1.0, abs=1e-6)
