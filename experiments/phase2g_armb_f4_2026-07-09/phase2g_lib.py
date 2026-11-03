"""Reusable, unit-tested helpers for Phase 2g (Arm-B replication of F4).

Arm A (all prior F3/F4 work): encoders see all 2500 panel genes and the decoder
predicts THOSE SAME genes. The paper disowns Arm A as "rewards linear encoders
through near-invertibility" -- the per-cell latent nearly reconstructs its own
input, so F4's mechanism claim ("the latent already absorbs the biological
cell-to-cell variation, so fitting marginal overdispersion double-counts it") is
demonstrated in the one setting where it is trivially true.

Arm B is the honest test. Encoders see ONLY G_in (1250 genes); the decoder
predicts G_out (1250 genes NO encoder ever saw). The latent cannot trivially
reconstruct the target.

Two things in this module carry load-bearing correctness and are therefore
unit-tested here (see tests/test_phase2g_armb.py):

  * ``gin_library_size`` -- the per-cell size-factor offset MUST be derived from
    G_in counts only. Using G_out's own depth would leak the prediction target
    and invalidate the whole arm. This function asserts G_in/G_out disjointness
    and that the returned offset is NOT the G_out depth.
  * ``exact_spearman_perm_p`` -- with only four encoders a perfect Spearman
    rho=+-1 makes scipy report p=0.0 (the t-statistic is infinite: a degeneracy,
    NOT a real p-value; this was defect #12 in the spec). The honest value is the
    exact permutation p over all 4! orderings.

"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.stats import spearmanr


def gin_library_size(raw: np.ndarray, gin_idx, gout_idx) -> np.ndarray:
    """Per-cell library-size offset for the Arm-B NB decoder, derived from G_in
    counts ONLY.

    ``lib_i = sum_{g in G_in} raw[i, g]``. The NB decoder recovers each cell's
    count mean as ``mu = exp(log_mu) * lib_i / mean(lib_train)``; if ``lib`` saw
    G_out the decoder would peek at its own prediction target's depth.

    Asserts (a) G_in and G_out are disjoint and (b) the returned offset does not
    equal the G_out depth -- the two failure modes that would leak the target.
    """
    gin_idx = np.asarray(gin_idx).ravel()
    gout_idx = np.asarray(gout_idx).ravel()
    assert np.intersect1d(gin_idx, gout_idx).size == 0, \
        "G_in and G_out gene indices must be disjoint (else the arm leaks)"
    lib = raw[:, gin_idx].sum(axis=1).astype(np.float32)
    lib_gout = raw[:, gout_idx].sum(axis=1).astype(np.float32)
    assert not np.allclose(lib, lib_gout), \
        "library-size offset must be G_in depth, not G_out depth (target leak)"
    assert (lib > 0).all(), "every cell must have positive G_in depth"
    return lib


def exact_spearman_perm_p(x, y) -> dict:
    """Exact permutation p-value for the Spearman correlation of two length-n
    vectors, over ALL n! relabellings of ``y`` against ``x``.

    For n=4 (the four encoders) this enumerates 24 orderings. A perfect monotone
    relationship (|rho|=1) yields exactly one ordering per sign, so p_one_sided =
    1/24 = 0.0417 and p_two_sided = 2/24 = 0.0833 -- the honest values the scipy
    parametric p (which returns 0.0 for |rho|=1, a degeneracy) cannot give.

    ``p_one_sided`` is the tail in the OBSERVED direction; ``p_two_sided`` counts
    |rho_perm| >= |rho_obs|. Observed permutation is included in both counts."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must be equal-length vectors of length >= 2")
    rho_obs = float(spearmanr(x, y).statistic)
    perms = list(permutations(range(y.size)))
    rhos = np.array([spearmanr(x, y[list(p)]).statistic for p in perms], dtype=np.float64)
    n = len(perms)
    tol = 1e-12
    ge_two = int(np.sum(np.abs(rhos) >= abs(rho_obs) - tol))
    if rho_obs >= 0:
        ge_one = int(np.sum(rhos >= rho_obs - tol))
    else:
        ge_one = int(np.sum(rhos <= rho_obs + tol))
    return {
        "rho": rho_obs,
        "n_perm": n,
        "p_two_sided": float(ge_two / n),
        "p_one_sided": float(ge_one / n),
    }


def assert_log1p_scale(decoded: np.ndarray, real: np.ndarray, name: str,
                       max_ratio: float = 4.0) -> dict:
    """Sanity guard against the recurring control-construction bug (spec §15):
    ``matched_noise`` / ``matched_noise_per_type`` MUST be built in log1p space.

    A control built in COUNT space (residual std of raw counts, magnitude in the
    hundreds given library sizes in the thousands) log1p'd afterward blows past
    the log1p range and annihilates itself (energy ~6-45 instead of ~1). Here we
    assert the decoded values live on the same bounded scale as the real log1p
    data: ``max(|decoded|) <= max_ratio * max(|real|)``. A count-space control
    fails this by an order of magnitude.

    Returns the observed maxima for logging."""
    dmax = float(np.nanmax(np.abs(decoded)))
    rmax = float(np.nanmax(np.abs(real)))
    ok = dmax <= max_ratio * rmax
    assert ok, (
        f"{name}: decoded max |{dmax:.2f}| exceeds {max_ratio}x real log1p max "
        f"|{rmax:.2f}| -- control looks built in COUNT space, not log1p space")
    return {"decoded_max_abs": dmax, "real_max_abs": rmax, "ratio": dmax / max(rmax, 1e-9)}


def build_armB_encoders(logX, gin_idx, tr, cache, pca_dim: int = 512, seed: int = 0):
    """Arm-B encoders, ALL restricted to G_in (1250 genes).

      * ``pca_512``      -- PCA(pca_dim) fit on ``logX[tr, gin_idx]``.
      * ``randproj_512`` -- fixed-seed Gaussian random projection of G_in.
      * ``scgpt_512``    -- cached ``enc_B_scgpt.npy`` (scGPT re-embedded on G_in).
      * ``geneformer``   -- cached ``enc_B_geneformer.npy`` (768-d, re-embedded on
        G_in).

    The two cached FM encoders are the whole point of Arm B: they were embedded on
    the 1250 G_in genes only, so they cannot see G_out. PCA/randproj are built here
    the same way (on the G_in slice). Returns a dict of available encoders."""
    from pathlib import Path

    from sklearn.decomposition import PCA

    cache = Path(cache)
    Xin = logX[:, np.asarray(gin_idx).ravel()]
    enc = {}
    enc["pca_512"] = PCA(n_components=pca_dim, random_state=0).fit(
        Xin[tr]).transform(Xin).astype(np.float32)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1 / np.sqrt(pca_dim), size=(Xin.shape[1], pca_dim)).astype(np.float32)
    enc["randproj_512"] = (Xin @ W).astype(np.float32)
    n = logX.shape[0]
    for key, fname in (("scgpt_512", "enc_B_scgpt.npy"),
                       ("geneformer", "enc_B_geneformer.npy")):
        p = cache / fname
        if p.exists():
            z = np.load(p)
            if z.shape[0] == n:
                enc[key] = z.astype(np.float32)
    return enc
