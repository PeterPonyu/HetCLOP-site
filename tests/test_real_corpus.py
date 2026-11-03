import json

import anndata as ad
import numpy as np
import pytest

from src.data.real_corpus import assemble_real_corpus, detect_species, load_dataset


def test_detect_species_human_all_uppercase():
    a = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32))
    a.var_names = ["HES4", "ISG15", "TNFRSF18"]
    assert detect_species(a) == "human"


def test_detect_species_mouse_titlecase():
    a = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32))
    a.var_names = ["P2ry12", "Hexb", "Csf1r"]
    assert detect_species(a) == "mouse"


def test_detect_species_raises_on_no_alphabetic_symbols():
    a = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    a.var_names = ["123", "456"]
    with pytest.raises(ValueError):
        detect_species(a)


def test_load_dataset_filters_unknown_cells_and_swaps_raw_counts(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    hvg8000_dir = tmp_path / "data" / "processed_h5ad_hvg8000"
    base_dir.mkdir(parents=True)
    hvg8000_dir.mkdir(parents=True)

    obs_names = ["cellA", "cellB", "cellC", "cellD"]
    var_names = ["HES4", "ISG15"]  # human-looking symbols

    def make(counts):
        a = ad.AnnData(X=np.array(counts, dtype=np.float32))  # .X = normalized stand-in (unused)
        a.obs_names = obs_names
        a.var_names = var_names
        a.layers["counts"] = np.array(counts, dtype=np.float32)
        return a

    counts = [[1, 2], [3, 0], [0, 5], [2, 2]]
    make(counts).write(base_dir / "dsX_processed.h5ad")
    make(counts).write(hvg8000_dir / "dsX_processed.h5ad")

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

    result = load_dataset("dsX", tmp_path)
    assert result["adata"].n_obs == 2  # only the "T cell" cluster survives
    assert list(result["labels"]) == ["T cell", "T cell"]
    assert result["species"] == "human"
    np.testing.assert_array_equal(result["adata"].X, np.array(counts, dtype=np.float32)[:2])


def _write_dataset(base_dir, hvg8000_dir, name, obs_names, var_names, counts, base_obs_names=None):
    """Writes matching base and hvg8000 h5ad files for one dataset. Pass a
    different `base_obs_names` to simulate a base/hvg8000 alignment
    mismatch (different obs_names order/identity than the hvg8000 file)."""

    def make(names):
        a = ad.AnnData(X=np.array(counts, dtype=np.float32))  # .X = normalized stand-in (unused)
        a.obs_names = names
        a.var_names = var_names
        a.layers["counts"] = np.array(counts, dtype=np.float32)
        return a

    make(obs_names).write(hvg8000_dir / f"{name}_processed.h5ad")
    make(base_obs_names if base_obs_names is not None else obs_names).write(
        base_dir / f"{name}_processed.h5ad"
    )


def _two_type_clusters():
    return {
        "0": {"cell_type": "T cell", "cell_indices": [0, 1]},
        "1": {"cell_type": "B cell", "cell_indices": [2, 3]},
    }


def test_assemble_real_corpus_happy_path_two_datasets(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    hvg8000_dir = tmp_path / "data" / "processed_h5ad_hvg8000"
    base_dir.mkdir(parents=True)
    hvg8000_dir.mkdir(parents=True)

    var_names = ["HES4", "ISG15"]
    counts_a = [[1, 2], [3, 0], [0, 5], [2, 2]]
    counts_b = [[4, 1], [0, 6], [2, 2], [1, 1]]
    _write_dataset(base_dir, hvg8000_dir, "dsA", ["a0", "a1", "a2", "a3"], var_names, counts_a)
    _write_dataset(base_dir, hvg8000_dir, "dsB", ["b0", "b1", "b2", "b3"], var_names, counts_b)

    payload = {"dsA": {"clusters": _two_type_clusters()}, "dsB": {"clusters": _two_type_clusters()}}
    with open(base_dir / "subcluster_metadata.json", "w") as f:
        json.dump(payload, f)

    result = assemble_real_corpus(["dsA", "dsB"], tmp_path)

    assert result["skipped"] == []
    assert set(result["dataset_ids"]) == {"dsA", "dsB"}
    assert (result["dataset_ids"] == "dsA").sum() == 4
    assert (result["dataset_ids"] == "dsB").sum() == 4
    assert len(result["labels"]) == 8
    assert set(result["raw_X_by_dataset"].keys()) == {"dsA", "dsB"}
    assert set(result["species_by_dataset"].keys()) == {"dsA", "dsB"}
    assert set(result["adata_by_dataset"].keys()) == {"dsA", "dsB"}
    np.testing.assert_allclose(
        result["raw_X_by_dataset"]["dsA"], np.log1p(np.array(counts_a, dtype=np.float32))
    )
    np.testing.assert_allclose(
        result["raw_X_by_dataset"]["dsB"], np.log1p(np.array(counts_b, dtype=np.float32))
    )


def test_assemble_real_corpus_skips_dataset_missing_from_subcluster_metadata(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    hvg8000_dir = tmp_path / "data" / "processed_h5ad_hvg8000"
    base_dir.mkdir(parents=True)
    hvg8000_dir.mkdir(parents=True)

    var_names = ["HES4", "ISG15"]
    counts_good = [[1, 2], [3, 0], [0, 5], [2, 2]]
    counts_missing = [[4, 1], [0, 6], [2, 2], [1, 1]]
    _write_dataset(base_dir, hvg8000_dir, "dsGood", ["g0", "g1", "g2", "g3"], var_names, counts_good)
    # dsMissing has both h5ad files (base + hvg8000, aligned) but NO entry
    # in subcluster_metadata.json below.
    _write_dataset(
        base_dir, hvg8000_dir, "dsMissing", ["m0", "m1", "m2", "m3"], var_names, counts_missing
    )

    payload = {"dsGood": {"clusters": _two_type_clusters()}}
    with open(base_dir / "subcluster_metadata.json", "w") as f:
        json.dump(payload, f)

    result = assemble_real_corpus(["dsGood", "dsMissing"], tmp_path)

    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["dataset_name"] == "dsMissing"
    assert result["skipped"][0]["reason"] == "missing_subcluster_metadata"
    assert set(result["dataset_ids"]) == {"dsGood"}
    assert set(result["raw_X_by_dataset"].keys()) == {"dsGood"}


def test_assemble_real_corpus_skips_dataset_with_alignment_mismatch(tmp_path):
    base_dir = tmp_path / "data" / "processed_h5ad"
    hvg8000_dir = tmp_path / "data" / "processed_h5ad_hvg8000"
    base_dir.mkdir(parents=True)
    hvg8000_dir.mkdir(parents=True)

    var_names = ["HES4", "ISG15"]
    counts_good = [[1, 2], [3, 0], [0, 5], [2, 2]]
    counts_bad = [[4, 1], [0, 6], [2, 2], [1, 1]]
    _write_dataset(base_dir, hvg8000_dir, "dsGood", ["g0", "g1", "g2", "g3"], var_names, counts_good)
    # dsBad: base file's obs_names order/identity differs from the hvg8000
    # file's -- triggers assert_base_hvg8000_alignment's ValueError.
    _write_dataset(
        base_dir,
        hvg8000_dir,
        "dsBad",
        ["b0", "b1", "b2", "b3"],
        var_names,
        counts_bad,
        base_obs_names=["b3", "b2", "b1", "b0"],
    )

    payload = {
        "dsGood": {"clusters": _two_type_clusters()},
        "dsBad": {"clusters": _two_type_clusters()},
    }
    with open(base_dir / "subcluster_metadata.json", "w") as f:
        json.dump(payload, f)

    result = assemble_real_corpus(["dsGood", "dsBad"], tmp_path)

    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["dataset_name"] == "dsBad"
    assert result["skipped"][0]["reason"] == "alignment_mismatch"
    assert set(result["dataset_ids"]) == {"dsGood"}
    assert set(result["raw_X_by_dataset"].keys()) == {"dsGood"}
