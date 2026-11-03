import math

import numpy as np
import pytest

from src.diagnostics.tightness import (
    between_type_distance,
    per_dataset_tightness_ratio,
    tightness,
    tightness_ratio,
    within_l2_ratio,
)


RAW_X = np.array([[0.0], [2.0], [10.0], [12.0]])
LABELS = ["A", "A", "B", "B"]


def test_within_l2_ratio_and_tightness_under_proportional_compression():
    # Encoder compresses every distance by exactly 10x (both within- and
    # between-type) -> tightness_ratio should stay ~1.0 even though the raw
    # magnitude compression (within_l2_ratio) is heavy. This is the exact
    # distinction the independent design review required: raw-magnitude
    # compression and structural (within/between) compression are different.
    encoder_x_proportional = np.array([[0.0], [0.2], [1.0], [1.2]])
    assert within_l2_ratio(encoder_x_proportional, RAW_X, LABELS) == pytest.approx(0.1, abs=1e-9)
    assert tightness_ratio(encoder_x_proportional, RAW_X, LABELS) == pytest.approx(1.0, abs=1e-9)


def test_tightness_ratio_below_one_under_disproportionate_compression():
    # Encoder compresses within-type distance far more than between-type
    # distance (the scGPT-like failure mode) -> tightness_ratio < 1.
    encoder_x_disproportionate = np.array([[0.0], [0.02], [1.0], [1.02]])
    assert tightness_ratio(encoder_x_disproportionate, RAW_X, LABELS) == pytest.approx(0.1, abs=1e-9)


def test_tightness_requires_at_least_two_types():
    with pytest.raises(ValueError):
        tightness(RAW_X, ["A", "A", "A", "A"])


def test_tightness_returns_nan_on_fully_degenerate_types():
    # Both types collapse to the exact same single point -> every within- AND
    # cross-type pairwise distance is 0.0 -> between_type_distance == 0.0,
    # which must yield nan (not ZeroDivisionError). (Pre-fix, this test used a
    # "centroids coincide but types still overlap with real spread" case --
    # that no longer zeros out between_type_distance now that it is defined
    # via cross-type pairwise distance rather than centroid distance: real
    # overlap between distinct, spread-out types now correctly yields a
    # nonzero, well-defined between_type_distance instead of nan.)
    X = np.array([[5.0], [5.0], [5.0], [5.0]])
    labels = ["A", "A", "B", "B"]
    assert math.isnan(tightness(X, labels))


def test_between_type_distance_not_invariant_to_shared_zero_mean_noise_dim():
    # Regression test for the dimensionality confound found 2026-07-05: the
    # old centroid-based between_type_distance ignored any extra dimension
    # whose per-type mean was unchanged (noise cancels out in a mean), while
    # within_type_distance (always pairwise) did not -- so appending such a
    # dimension mechanically inflated tightness_ratio with zero real
    # biological content. between_type_distance must now move too.
    X_base = np.array([[0.0], [2.0], [10.0], [12.0]])
    labels = ["A", "A", "B", "B"]
    # Zero mean per type (mean of [5, -5] is 0 for both A and B) -- a
    # centroid-based estimator is entirely blind to this dimension.
    noise_dim = np.array([[5.0], [-5.0], [5.0], [-5.0]])
    X_with_noise = np.hstack([X_base, noise_dim])

    base_between = between_type_distance(X_base, labels)
    with_noise_between = between_type_distance(X_with_noise, labels)
    assert with_noise_between > base_between


def test_per_dataset_tightness_ratio_matches_manual_median():
    # 2 datasets, each with its own 2-type separated structure, at different
    # absolute scales (so a global pool would mix scales; per-dataset must not).
    raw_x_d1 = np.array([[0.0], [1.0], [10.0], [11.0]])
    raw_x_d2 = np.array([[0.0], [2.0], [100.0], [102.0]])
    labels = ["A", "A", "B", "B"] * 2
    dataset_ids = ["d1"] * 4 + ["d2"] * 4
    # identity encoder: tightness_ratio == 1.0 in every dataset
    encoder_x = np.vstack([raw_x_d1, raw_x_d2])
    raw_X_by_dataset = {"d1": raw_x_d1, "d2": raw_x_d2}

    median, (q25, q75), values = per_dataset_tightness_ratio(
        encoder_x, raw_X_by_dataset, labels, dataset_ids
    )
    assert values.shape == (2,)
    assert median == pytest.approx(1.0, abs=1e-9)
    assert q25 == pytest.approx(1.0, abs=1e-9)
    assert q75 == pytest.approx(1.0, abs=1e-9)


def test_per_dataset_tightness_ratio_skips_datasets_with_too_few_types():
    raw_x_d1 = np.array([[0.0], [1.0], [10.0], [11.0]])
    raw_x_d2 = np.array([[0.0], [1.0]])
    labels = ["A", "A", "B", "B", "A", "A"]  # dataset d2 has only type A
    dataset_ids = ["d1"] * 4 + ["d2"] * 2
    encoder_x = np.vstack([raw_x_d1, raw_x_d2])
    raw_X_by_dataset = {"d1": raw_x_d1, "d2": raw_x_d2}
    median, (q25, q75), values = per_dataset_tightness_ratio(
        encoder_x, raw_X_by_dataset, labels, dataset_ids
    )
    assert values.shape == (1,)  # only d1 qualifies (>=2 types)


def test_per_dataset_tightness_ratio_raises_when_no_dataset_qualifies():
    raw_x = np.array([[0.0], [1.0], [2.0]])
    labels = ["A", "A", "A"]
    dataset_ids = ["d1", "d1", "d1"]
    raw_X_by_dataset = {"d1": raw_x}
    with pytest.raises(ValueError):
        per_dataset_tightness_ratio(raw_x.copy(), raw_X_by_dataset, labels, dataset_ids)


def test_per_dataset_tightness_ratio_skips_degenerate_nan_dataset():
    # d1: both types fully collapse to the same point -> tightness_ratio
    # returns nan (not raise), which must NOT be silently folded into the
    # median/IQR alongside a well-behaved dataset.
    raw_x_d1 = np.array([[5.0], [5.0], [5.0], [5.0]])
    labels_d1 = ["A", "A", "B", "B"]
    # d2: well-behaved, well-separated two-type structure.
    raw_x_d2 = np.array([[0.0], [1.0], [10.0], [11.0]])
    labels_d2 = ["A", "A", "B", "B"]

    labels = labels_d1 + labels_d2
    dataset_ids = ["d1"] * 4 + ["d2"] * 4
    encoder_x = np.vstack([raw_x_d1, raw_x_d2])  # identity encoder
    raw_X_by_dataset = {"d1": raw_x_d1, "d2": raw_x_d2}

    median, (q25, q75), values = per_dataset_tightness_ratio(
        encoder_x, raw_X_by_dataset, labels, dataset_ids
    )
    assert values.shape == (1,)  # only d2 qualifies; d1's nan must be excluded
    assert not math.isnan(median)
    assert not math.isnan(q25)
    assert not math.isnan(q75)
    assert median == pytest.approx(1.0, abs=1e-9)


def test_per_dataset_tightness_ratio_allows_different_gene_dimensionality_per_dataset():
    # d1's raw space has 1 gene, d2's has 3 -- real corpus datasets each
    # select their own top-8000 HVGs independently, so gene panels differ in
    # both identity AND count across datasets. A single shared raw_X matrix
    # sliced by dataset (the pre-fix design) could not express this at all --
    # this test proves the fix, not just re-verify math already covered above.
    raw_x_d1 = np.array([[0.0], [1.0], [10.0], [11.0]])
    raw_x_d2 = np.array(
        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [100.0, 50.0, 25.0], [102.0, 51.0, 25.5]]
    )
    labels = ["A", "A", "B", "B"] * 2
    dataset_ids = ["d1"] * 4 + ["d2"] * 4
    # encoder space is shared (e.g. a real embedder's fixed output dim) even
    # though the two datasets' raw gene spaces differ in dimensionality.
    encoder_x = np.array(
        [
            [0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1],
            [0.0, 0.0], [0.2, 0.2], [10.0, 10.0], [10.2, 10.2],
        ]
    )
    raw_X_by_dataset = {"d1": raw_x_d1, "d2": raw_x_d2}
    median, (q25, q75), values = per_dataset_tightness_ratio(
        encoder_x, raw_X_by_dataset, labels, dataset_ids
    )
    assert values.shape == (2,)
    assert not math.isnan(median)
