"""Encoder-comparison diagnostics.

Reports the STRUCTURAL (within/between-type) compression metric — tightness
ratio — as the primary quantity, per the corrected framing in
the design notes section 1.
`within_l2_ratio` (raw-magnitude compression) is provided for completeness and
cross-checking, but must never be used
as a stand-in for tightness_ratio: a magnitude compression can be uniform
across within- and between-type distances (tightness_ratio ~ 1, benign) or
disproportionate (tightness_ratio << 1, the actual failure mode).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def within_type_distance(X: np.ndarray, labels) -> float:
    labels = np.asarray(labels)
    per_type_means = []
    for t in np.unique(labels):
        mask = labels == t
        if mask.sum() < 2:
            continue
        per_type_means.append(pdist(X[mask], metric="euclidean").mean())
    if not per_type_means:
        raise ValueError("no cell type has >= 2 cells; within-type distance is undefined")
    return float(np.mean(per_type_means))


def between_type_distance(X: np.ndarray, labels) -> float:
    """Mean cross-type PAIRWISE distance (average over all cell-i-in-t1,
    cell-j-in-t2 pairs, then over all type pairs) -- NOT centroid-to-centroid
    distance.

    Centroid distance was the original definition, but it is asymmetric with
    within_type_distance in a way that biases tightness_ratio: a centroid is
    a MEAN over n cells, so any zero-signal noise dimension's contribution to
    it shrinks with sampling averaging (~1/n), while within_type_distance's
    raw pairwise distances get the FULL, unreduced noise contribution every
    time. Found 2026-07-05 via a real-data run where all 3 FMs (higher output
    dimensionality than PCA) beat PCA on tightness ratio: a synthetic check
    reproduced the exact effect -- appending pure-noise dimensions to an
    embedding mechanically inflates tightness_ratio with zero real
    biological content, asymptoting near sqrt(n_cells_per_type) as noise
    dimensions grow, independent of any real structure. Cross-type pairwise
    distance uses the same (unreduced, per-cell) estimator as
    within_type_distance, so a shared noise dimension inflates both
    numerator and denominator comparably instead of only the numerator --
    see test_between_type_distance_not_invariant_to_shared_zero_mean_noise_dim
    for the regression case this fixes."""
    labels = np.asarray(labels)
    types = np.unique(labels)
    if len(types) < 2:
        raise ValueError("need >= 2 cell types to compute a between-type distance")
    pair_means = [
        cdist(X[labels == t1], X[labels == t2], metric="euclidean").mean()
        for t1, t2 in combinations(types, 2)
    ]
    return float(np.mean(pair_means))


def within_l2_ratio(encoder_X: np.ndarray, raw_X: np.ndarray, labels) -> float:
    denom = within_type_distance(raw_X, labels)
    if denom == 0.0:
        return float("nan")
    return within_type_distance(encoder_X, labels) / denom


def tightness(X: np.ndarray, labels) -> float:
    denom = between_type_distance(X, labels)
    if denom == 0.0:
        return float("nan")
    return within_type_distance(X, labels) / denom


def tightness_ratio(encoder_X: np.ndarray, raw_X: np.ndarray, labels) -> float:
    denom = tightness(raw_X, labels)
    if denom == 0.0:
        return float("nan")
    return tightness(encoder_X, labels) / denom


def per_dataset_tightness_ratio(
    encoder_X: np.ndarray,
    raw_X_by_dataset: dict[str, np.ndarray],
    labels,
    dataset_ids,
    min_types_per_dataset: int = 2,
) -> tuple[float, tuple[float, float], np.ndarray]:
    """Tightness ratio computed PER DATASET then reduced to median + IQR
    across datasets -- the estimator of the original description
    ("PCA is fitted per-dataset", reporting
    median/IQR over eight datasets), NOT a single ratio pooled across all
    cell types at once.

    raw_X_by_dataset supplies each dataset's OWN native raw feature space
    (e.g. its own top-8000 HVG panel) -- an earlier version of this function
    took one shared raw_X matrix sliced by dataset, which silently assumed
    every dataset used the same gene panel. That assumption does not hold
    for the per-dataset HVG selection (confirmed empirically:
    two real datasets shared only 3,579 of 8,000 genes) -- found during a
    round-3 fix, see the design notes
    readiness-design.md section 3.

    INVARIANT this function relies on but cannot verify itself: for each
    dataset d, raw_X_by_dataset[d] must have its rows in the SAME ORDER as
    the `dataset_ids == d` slice of encoder_X/labels -- a row-count mismatch
    fails loud (numpy shape error), but a same-count-different-order
    mismatch will silently misalign rows and is NOT caught here (this
    function has no cell-identifier information to check against); callers
    (e.g. src.data.real_corpus.assemble_real_corpus) must guarantee this by
    construction."""
    labels = np.asarray(labels)
    dataset_ids = np.asarray(dataset_ids)
    per_dataset_values = []
    for d in np.unique(dataset_ids):
        mask = dataset_ids == d
        d_labels = labels[mask]
        if len(np.unique(d_labels)) < min_types_per_dataset:
            continue
        if d not in raw_X_by_dataset:
            raise ValueError(
                f"dataset '{d}' appears in dataset_ids but has no entry in raw_X_by_dataset"
            )
        try:
            value = tightness_ratio(encoder_X[mask], raw_X_by_dataset[d], d_labels)
        except ValueError:
            continue
        if np.isnan(value):
            # tightness_ratio/tightness return nan (rather than raising) on a
            # zero denominator (e.g. coincident type centroids within this
            # dataset) -- exclude it from the reduction rather than let a nan
            # silently corrupt the median/IQR across the other datasets.
            continue
        per_dataset_values.append(value)
    if not per_dataset_values:
        raise ValueError(
            "no dataset had >= min_types_per_dataset cell types with >= 2 cells each "
            "and a well-defined (non-nan) tightness ratio; per-dataset tightness ratio "
            "is undefined"
        )
    values = np.array(per_dataset_values)
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    return median, (float(q25), float(q75)), values
