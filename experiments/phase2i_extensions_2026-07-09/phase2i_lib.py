"""Reusable helpers for Phase 2i extensions (G1-G5).

Only NEW machinery lives here; everything else is imported from the existing
phase2a/b/c/d/f/g modules and ``src`` so no metric or decoder is reimplemented:

  * ``pca_k_encoder`` / ``randproj_encoder`` -- byte-for-byte the encoder recipes
    used in phase2b/phase2f (PCA fit on the train split of the given feature
    matrix; fixed-seed Gaussian random projection), parameterised by dimension so
    a MONOTONE information gradient (G1: k in {2,5,10,25,50,512}) and the extra
    low-dim points for the monotonicity leg (G4: pca_50, randproj_50, pca_1250)
    can be constructed uniformly.
  * ``noise_corrupted_pca_encoder`` -- the second G1 gradient: additive Gaussian
    noise (in units of each gene's train-split std) is injected into the input
    BEFORE the PCA is fit, so more noise -> strictly less recoverable information.
  * ``adjacent_pairs`` -- consecutive (i, i+1) index pairs along an ordering, used
    by G1 to find the SMALLEST information step energy distance still resolves.

The exact n!-permutation Spearman p (spec defect #12: scipy reports p=0.0 for
|rho|=1) is reused directly from ``phase2g_lib.exact_spearman_perm_p`` -- not
re-implemented.

"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_k_encoder(X: np.ndarray, tr: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """PCA(k) fit on ``X[tr]`` and applied to all rows -- the exact recipe of
    phase2b/phase2f's ``pca_512`` with the number of components exposed. ``X`` is
    the (already sliced) feature matrix (full panel for Arm A, G_in for Arm B)."""
    k = int(min(k, X.shape[1], int(tr.sum())))
    return PCA(n_components=k, random_state=random_state).fit(
        X[tr]).transform(X).astype(np.float32)


def randproj_encoder(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Fixed-seed Gaussian random projection to ``k`` dims -- byte-for-byte the
    phase2b/phase2f ``randproj_512`` recipe (W ~ N(0, 1/k), columns of X @ W)."""
    k = int(k)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1 / np.sqrt(k), size=(X.shape[1], k)).astype(np.float32)
    return (X @ W).astype(np.float32)


def noise_corrupted_pca_encoder(X: np.ndarray, tr: np.ndarray, k: int,
                                noise_std: float, seed: int = 0,
                                random_state: int = 0) -> np.ndarray:
    """PCA(k) on an input CORRUPTED by additive Gaussian noise, sized per gene.

    ``X_corrupt = X + noise_std * sigma_g * eps`` with ``sigma_g`` the per-column
    std on the train split and ``eps ~ N(0, I)`` (fixed seed). The PCA is fit on
    the corrupted train rows and applied to all corrupted rows, so a larger
    ``noise_std`` monotonically destroys recoverable information. ``noise_std=0``
    reduces exactly to ``pca_k_encoder``."""
    sigma = X[tr].std(axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    Xc = (X + noise_std * sigma[None, :]
          * rng.standard_normal(X.shape).astype(np.float32)).astype(np.float32)
    k = int(min(k, Xc.shape[1], int(tr.sum())))
    return PCA(n_components=k, random_state=random_state).fit(
        Xc[tr]).transform(Xc).astype(np.float32)


def adjacent_pairs(order):
    """Consecutive (a, b) pairs along ``order`` (a list of names/indices)."""
    order = list(order)
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]
