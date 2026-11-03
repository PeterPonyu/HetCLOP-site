import json

import anndata as ad
import numpy as np
import pytest

from src.data.labels import (
    assert_base_hvg8000_alignment,
    build_cell_type_labels,
    load_subcluster_metadata,
)


def _make_adata(obs_names):
    n = len(obs_names)
    a = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    a.obs_names = obs_names
    return a


def test_load_subcluster_metadata_returns_only_requested_dataset(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    base_dir.mkdir(parents=True)
    payload = {
        "datasetA": {"clusters": {"0": {"cell_type": "T cell", "cell_indices": [0, 1]}}},
        "datasetB": {"clusters": {"0": {"cell_type": "B cell", "cell_indices": [0]}}},
    }
    with open(base_dir / "subcluster_metadata.json", "w") as f:
        json.dump(payload, f)

    clusters = load_subcluster_metadata("datasetA", tmp_path)
    assert clusters == {"0": {"cell_type": "T cell", "cell_indices": [0, 1]}}


def test_load_subcluster_metadata_raises_for_unknown_dataset(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    base_dir.mkdir(parents=True)
    with open(base_dir / "subcluster_metadata.json", "w") as f:
        json.dump({"datasetA": {"clusters": {}}}, f)
    with pytest.raises(KeyError):
        load_subcluster_metadata("missing_dataset", tmp_path)


def test_assert_base_hvg8000_alignment_passes_when_identical(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    base_dir.mkdir(parents=True)
    obs_names = ["cellA", "cellB", "cellC"]
    _make_adata(obs_names).write(base_dir / "dsX_processed.h5ad")

    hvg8000_adata = _make_adata(obs_names)
    assert_base_hvg8000_alignment("dsX", hvg8000_adata, tmp_path)  # must not raise


def test_assert_base_hvg8000_alignment_raises_on_count_mismatch(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    base_dir.mkdir(parents=True)
    _make_adata(["cellA", "cellB", "cellC"]).write(base_dir / "dsX_processed.h5ad")

    hvg8000_adata = _make_adata(["cellA", "cellB"])  # fewer cells
    with pytest.raises(ValueError, match="cells"):
        assert_base_hvg8000_alignment("dsX", hvg8000_adata, tmp_path)


def test_assert_base_hvg8000_alignment_raises_on_reordered_identity(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    base_dir.mkdir(parents=True)
    _make_adata(["cellA", "cellB", "cellC"]).write(base_dir / "dsX_processed.h5ad")

    hvg8000_adata = _make_adata(["cellA", "cellC", "cellB"])  # same set, different order
    with pytest.raises(ValueError, match="order"):
        assert_base_hvg8000_alignment("dsX", hvg8000_adata, tmp_path)


def test_build_cell_type_labels_assigns_known_types_and_drops_unknown(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    base_dir.mkdir(parents=True)
    obs_names = ["cellA", "cellB", "cellC", "cellD"]
    _make_adata(obs_names).write(base_dir / "dsX_processed.h5ad")

    payload = {
        "dsX": {
            "clusters": {
                "0": {"cell_type": "T cell", "cell_indices": [0, 1]},
                "1": {"cell_type": "Unknown", "cell_indices": [2, 3]},
            }
        }
    }
    with open(base_dir / "subcluster_metadata.json", "w") as f:
        json.dump(payload, f)

    hvg8000_adata = _make_adata(obs_names)
    labels = build_cell_type_labels("dsX", hvg8000_adata, tmp_path)
    assert list(labels) == ["T cell", "T cell", None, None]
