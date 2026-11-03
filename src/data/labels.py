"""Real per-cell cell-type label reconstruction from the corpus data-preparation files. Found necessary after a real-data pilot smoke test
showed that the real
per-cell cell-type label is not stored as a clean array anywhere current --
it lives in subcluster_metadata.json's per-cluster `cell_type` field. This
project's per-dataset tightness diagnostic only ever compares cell types
WITHIN one dataset at a time, so the cross-dataset 69-type
deduplication (whose primary input, caption_analysis.json, no longer exists
in the repo) is not needed -- see the design notes
phase0-readiness-design.md section 2."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np


def load_subcluster_metadata(dataset_name: str, corpus_source_root: Path) -> dict[str, dict]:
    path = corpus_source_root / "data" / "processed_h5ad" / "subcluster_metadata.json"
    with open(path) as f:
        all_metadata = json.load(f)
    if dataset_name not in all_metadata:
        raise KeyError(f"dataset '{dataset_name}' not found in {path}")
    return all_metadata[dataset_name]["clusters"]


def assert_base_hvg8000_alignment(
    dataset_name: str, hvg8000_adata: ad.AnnData, corpus_source_root: Path
) -> None:
    base_path = corpus_source_root / "data" / "processed_h5ad" / f"{dataset_name}_processed.h5ad"
    base_adata = ad.read_h5ad(base_path)
    base_names = list(base_adata.obs_names)
    hvg8000_names = list(hvg8000_adata.obs_names)
    if len(base_names) != len(hvg8000_names):
        raise ValueError(
            f"dataset '{dataset_name}': base processed_h5ad has {len(base_names)} cells "
            f"but processed_h5ad_hvg8000 has {len(hvg8000_names)} cells -- cell_indices "
            "from subcluster_metadata.json (positions into the base file) cannot be "
            "trusted against the hvg8000 file's row order"
        )
    if base_names != hvg8000_names:
        raise ValueError(
            f"dataset '{dataset_name}': base processed_h5ad and processed_h5ad_hvg8000 "
            "have the same cell count but a different obs_names order/identity -- "
            "cell_indices from subcluster_metadata.json cannot be trusted against the "
            "hvg8000 file's row order"
        )


def build_cell_type_labels(
    dataset_name: str, hvg8000_adata: ad.AnnData, corpus_source_root: Path
) -> np.ndarray:
    assert_base_hvg8000_alignment(dataset_name, hvg8000_adata, corpus_source_root)
    clusters = load_subcluster_metadata(dataset_name, corpus_source_root)
    labels = np.full(hvg8000_adata.n_obs, None, dtype=object)
    for cluster_info in clusters.values():
        if cluster_info["cell_type"] == "Unknown":
            continue
        labels[cluster_info["cell_indices"]] = cluster_info["cell_type"]
    return labels
