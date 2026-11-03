"""Shared human-symbol gene panel + one global PCA across all 50 datasets.

Phase 0 fit PCA *per dataset*, which is correct for a per-dataset tightness
diagnostic but gives every dataset mutually-incomparable axes (two same-species
datasets' independently-selected HVG panels overlap only ~9-40%). Phase 1.5's
aligner trains a SINGLE contrastive loss over all datasets pooled, so cells must
live in ONE shared feature space. This module builds that space:

  1. canonicalize_symbols  -- map every dataset's genes to a single human-symbol
     space (human: uppercase var_names as-is; mouse: map_mouse_symbols_to_human,
     dropping genes with no ortholog).
  2. select_shared_panel   -- rank genes by how many datasets they appear in
     (frequency across the 50 native HVG panels), tie-broken alphabetically,
     take the top ``target_size``.
  3. reproject_to_panel    -- reproject a dataset's raw counts onto the shared
     panel, zero-filling genes the dataset's own HVG selection didn't include
     (legitimate: reflects that dataset's HVG choice, not biological absence),
     then log1p (project convention).
  4. fit_global_pca        -- one PCA (n_components=50, matching Phase 0's PCA
     baseline dimensionality) over the pooled reprojected matrix.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.orthologs.mapping import map_mouse_symbols_to_human, normalize_symbol


def canonicalize_symbols(
    var_names: list[str], species: str, ortholog_table: dict[str, str]
) -> list[str | None]:
    """Return per-gene human symbol (or None to drop). Human genes are
    uppercased as-is; mouse genes routed through the MGI ortholog table."""
    if species == "human":
        return [normalize_symbol(str(s)) for s in var_names]
    if species == "mouse":
        return map_mouse_symbols_to_human([str(s) for s in var_names], ortholog_table)
    raise ValueError(f"unknown species: {species!r}")


def select_shared_panel(
    human_symbols_by_dataset: dict[str, list[str | None]],
    target_size: int = 2500,
) -> tuple[list[str], dict]:
    """Select the top ``target_size`` human symbols by dataset-frequency.

    A gene's "frequency" is the number of datasets whose (canonicalized) HVG
    panel contains it. Ties broken alphabetically for determinism. Returns the
    panel plus a coverage-stats dict for logging/auditing.
    """
    from collections import Counter

    freq: Counter[str] = Counter()
    for symbols in human_symbols_by_dataset.values():
        # Unique per dataset so a duplicated symbol can't inflate its frequency.
        for s in {x for x in symbols if x is not None}:
            freq[s] += 1

    n_datasets = len(human_symbols_by_dataset)
    # Deterministic order: higher frequency first, then alphabetical.
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    panel = [s for s, _ in ranked[:target_size]]

    # Coverage distribution over the whole canonical vocabulary.
    freq_values = np.array(list(freq.values()))
    panel_freqs = np.array([freq[s] for s in panel]) if panel else np.array([])
    stats = {
        "n_datasets": n_datasets,
        "n_unique_human_symbols": len(freq),
        "panel_size": len(panel),
        "panel_min_dataset_freq": int(panel_freqs.min()) if len(panel_freqs) else 0,
        "panel_max_dataset_freq": int(panel_freqs.max()) if len(panel_freqs) else 0,
        "panel_median_dataset_freq": float(np.median(panel_freqs)) if len(panel_freqs) else 0.0,
        "vocab_freq_histogram": {
            int(f): int((freq_values == f).sum()) for f in np.unique(freq_values)
        },
    }
    return panel, stats


def reproject_to_panel(
    counts: np.ndarray,
    dataset_human_symbols: list[str | None],
    panel: list[str],
) -> np.ndarray:
    """Reproject raw ``counts`` (n_cells x n_dataset_genes) onto ``panel``,
    zero-filling absent genes, then log1p. Returns (n_cells x len(panel))."""
    panel_index = {s: j for j, s in enumerate(panel)}
    out = np.zeros((counts.shape[0], len(panel)), dtype=np.float32)
    # Map each dataset column that lands in the panel.
    for col, sym in enumerate(dataset_human_symbols):
        if sym is None:
            continue
        j = panel_index.get(sym)
        if j is None:
            continue
        # If two dataset columns map to the same human symbol (rare, e.g. an
        # ortholog collision), accumulate so no signal is silently dropped.
        out[:, j] += counts[:, col]
    return np.log1p(out)


def fit_global_pca(
    pooled_matrix: np.ndarray, n_components: int = 50, random_state: int = 0
) -> tuple[np.ndarray, PCA]:
    """Fit ONE PCA over the pooled reprojected matrix. Returns (embeddings, pca)."""
    n_comp = min(n_components, pooled_matrix.shape[0] - 1, pooled_matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    emb = pca.fit_transform(pooled_matrix).astype(np.float32)
    if n_comp < n_components:
        emb = np.pad(emb, ((0, 0), (0, n_components - n_comp)))
    return emb, pca
