"""Assembles real corpus dataset(s) into Phase-0-ready inputs: raw counts
(from layers['counts'], never .X), real cell-type labels (via
src.data.labels, dropping unlabeled/"Unknown" cells), and species detection
(via gene-symbol casing). `load_dataset` handles one dataset; `assemble_real_corpus`
composes a list of them into the `dataset_ids`/`raw_X_by_dataset` inputs
`src.diagnostics.tightness.per_dataset_tightness_ratio` and (eventually)
`compare_encoders` need. Actually CALLING embedders/`compare_encoders` for a
real multi-dataset run (which datasets, in what order, checkpointing) is
still deliberately left for a later, separate execution step -- this module
is the glue layer only, see
the design notes section 5."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from src.data.labels import build_cell_type_labels


def detect_species(adata: ad.AnnData) -> str:
    """Human gene panels are ~100% all-uppercase alphabetic symbols (e.g.
    HES4); mouse panels are ~0% (TitleCase/mixed, e.g. P2ry12) -- an
    unambiguous 0.0-vs-1.0 signal verified against 3 real corpus datasets
    during design (no middle ground observed)."""
    names = [str(n) for n in adata.var_names]
    # Real gene symbols often contain digits (e.g. "HES4", "P2ry12"), so
    # `n.isalpha()` (which requires the *entire* token to be alphabetic)
    # would wrongly exclude nearly all of them. Instead keep any name with
    # at least one alphabetic character, and rely on `n == n.upper()` for
    # the case check (digits are unaffected by .upper()).
    alpha_names = [n for n in names if any(c.isalpha() for c in n)]
    if not alpha_names:
        raise ValueError("cannot detect species: no alphabetic gene symbols in var_names")
    frac_upper = sum(1 for n in alpha_names if n == n.upper()) / len(alpha_names)
    return "human" if frac_upper > 0.5 else "mouse"


def load_dataset(dataset_name: str, corpus_source_root: Path) -> dict:
    hvg8000_path = (
        corpus_source_root / "data" / "processed_h5ad_hvg8000" / f"{dataset_name}_processed.h5ad"
    )
    adata = ad.read_h5ad(hvg8000_path)
    adata.X = adata.layers["counts"]

    labels = build_cell_type_labels(dataset_name, adata, corpus_source_root)
    keep_mask = labels != None  # noqa: E711 -- object-array elementwise compare, not `is not None`
    adata = adata[keep_mask].copy()
    labels = labels[keep_mask].astype(str)

    species = detect_species(adata)

    # Secondary cross-check against the Hm/Mm filename convention (warning
    # only -- the gene-symbol signal above is authoritative).
    if "Hm" in dataset_name and species != "human":
        print(
            f"WARNING: dataset '{dataset_name}' name suggests human (Hm) but "
            f"gene-symbol casing detected species='{species}'"
        )
    if "Mm" in dataset_name and species != "mouse":
        print(
            f"WARNING: dataset '{dataset_name}' name suggests mouse (Mm) but "
            f"gene-symbol casing detected species='{species}'"
        )

    return {"adata": adata, "labels": labels, "species": species}


def assemble_real_corpus(dataset_names: list[str], corpus_source_root: Path) -> dict:
    """Composes `load_dataset` across MULTIPLE datasets into exactly the
    inputs `src.diagnostics.tightness.per_dataset_tightness_ratio` needs:
    a concatenated `dataset_ids`/`labels` pair and a per-dataset
    `raw_X_by_dataset` dict of log1p-normalized raw expression (matching the
    convention already established in
    experiments/pilot_2026-07-05/run_pilot.py, where `tightness_ratio` is
    always called on `np.log1p(raw_X)`, never bare raw counts).

    Two failure modes are caught PER DATASET and turned into a clean, named
    skip (recorded in the returned `skipped` list) rather than crashing the
    whole batch:
      - `KeyError` from `load_subcluster_metadata` (dataset absent from
        subcluster_metadata.json) -- explicitly required to be a named skip
        by the design notes
        section 7.
      - `ValueError` from `assert_base_hvg8000_alignment` (base vs. hvg8000
        obs_names mismatch) -- section 7 describes this as the loader
        "may simply refuse to run on SOME datasets", which we read as a
        per-dataset refusal, not an all-or-nothing batch failure: crashing a
        whole multi-dataset run because one dataset has a stale cache would
        defeat the purpose of this function. This is still not silent --
        every skip is logged with its specific reason and returned in
        `skipped`, so nothing is silently swallowed.
    Any other exception is left to propagate (fail loud), since only these
    two failure modes are documented, expected-to-occur-on-real-data cases.
    """
    dataset_ids_parts = []
    labels_parts = []
    raw_X_by_dataset: dict[str, np.ndarray] = {}
    species_by_dataset: dict[str, str] = {}
    adata_by_dataset: dict[str, ad.AnnData] = {}
    skipped: list[dict] = []

    for name in dataset_names:
        try:
            result = load_dataset(name, corpus_source_root)
        except KeyError as e:
            print(f"SKIP dataset '{name}': missing from subcluster_metadata.json ({e})")
            skipped.append(
                {
                    "dataset_name": name,
                    "reason": "missing_subcluster_metadata",
                    "detail": str(e),
                }
            )
            continue
        except ValueError as e:
            print(f"SKIP dataset '{name}': base/hvg8000 obs_names alignment mismatch ({e})")
            skipped.append(
                {
                    "dataset_name": name,
                    "reason": "alignment_mismatch",
                    "detail": str(e),
                }
            )
            continue

        adata = result["adata"]
        labels = result["labels"]

        dataset_ids_parts.append(np.full(adata.n_obs, name, dtype=object))
        labels_parts.append(labels)

        raw_X = adata.X
        raw_X = raw_X.toarray() if hasattr(raw_X, "toarray") else np.asarray(raw_X)
        raw_X_by_dataset[name] = np.log1p(raw_X)

        species_by_dataset[name] = result["species"]
        adata_by_dataset[name] = adata

    dataset_ids = (
        np.concatenate(dataset_ids_parts) if dataset_ids_parts else np.array([], dtype=object)
    )
    labels = (
        np.concatenate(labels_parts) if labels_parts else np.array([], dtype=object)
    )

    return {
        "dataset_ids": dataset_ids,
        "labels": labels,
        "raw_X_by_dataset": raw_X_by_dataset,
        "species_by_dataset": species_by_dataset,
        "adata_by_dataset": adata_by_dataset,
        "skipped": skipped,
    }
