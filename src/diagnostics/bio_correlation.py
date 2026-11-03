"""Biological-correlation control: does an encoder's within-type spread
correlate with real within-type raw-expression structure, or is it just
noise? Required per independent design review Q1 — the permutation null in
significance.py catches whether tightness reflects real TYPE structure, but
not whether the recovered variance is BIOLOGICAL rather than isotropic noise.
Uses a Mantel-style correlation of within-type pairwise-distance vectors.

The raw-vs-embedding correlation is partialled against per-cell library size
(reconstructed via expm1, since raw_X is assumed log1p-space per
src.data.real_corpus's convention) because raw within-type distance is
itself substantially driven by technical depth/dropout variance, not just
biology. PCA is a near-linear transform of log1p(raw counts), so it inherits
and preserves that technical variance almost by construction -- a high
bio-correlation for PCA is not strong evidence it is more "biological," only
that it is more linear. Flagged in a round-2 red-team review, then
elevated to blocking by the real Phase 0 run (2026-07-05): PCA's raw
bio-correlation (0.887) was far above every FM's (<=0.36) alongside FMs
also winning on tightness ratio -- a pattern this control could not
distinguish from "FMs discard technical noise while reorganizing around
real structure" without controlling for library size."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def bio_correlation_control(
    raw_X: np.ndarray, embed_X: np.ndarray, labels, min_cells_per_type: int = 4
) -> float:
    labels = np.asarray(labels)
    per_type_r = []
    for t in np.unique(labels):
        mask = labels == t
        if mask.sum() < min_cells_per_type:
            continue
        raw_d = pdist(raw_X[mask], metric="euclidean")
        embed_d = pdist(embed_X[mask], metric="euclidean")
        if np.std(raw_d) == 0 or np.std(embed_d) == 0:
            continue
        r_xy = float(pearsonr(raw_d, embed_d)[0])

        libsize = np.expm1(raw_X[mask]).sum(axis=1)
        libsize_d = pdist(libsize.reshape(-1, 1), metric="euclidean")
        if np.std(libsize_d) == 0:
            # No depth variation within this type to control for -- nothing
            # to partial out, fall back to the plain correlation.
            per_type_r.append(float(r_xy))
            continue
        r_xz = float(pearsonr(raw_d, libsize_d)[0])
        r_yz = float(pearsonr(embed_d, libsize_d)[0])
        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom == 0:
            per_type_r.append(float(r_xy))
            continue
        r_partial = (r_xy - r_xz * r_yz) / denom
        per_type_r.append(float(np.clip(r_partial, -1.0, 1.0)))
    if not per_type_r:
        raise ValueError(
            "no cell type had >= min_cells_per_type cells with nonzero pairwise-"
            "distance variance; bio-correlation control is undefined"
        )
    return float(np.mean(per_type_r))
