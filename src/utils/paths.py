"""Path constants for this repository.

Each constant can be overridden by an environment variable. Inputs that are not part of the archive (the public
GEO source matrices and foundation-model weights) are read from directories under data/ by default.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_path(env_key: str, default: Path) -> Path:
    val = os.environ.get(env_key)
    return Path(val).resolve() if val else default


# Processed per-dataset source files and their metadata (only needed to rebuild the corpus from GEO sources).
CORPUS_SOURCE_ROOT = _env_path("HETCLOP_CORPUS_SOURCE_ROOT", PROJECT_ROOT / "data" / "corpus_sources")
# Foundation-model assets (only needed to recompute the cached scGPT and Geneformer embeddings).
FM_ASSETS_ROOT = _env_path("HETCLOP_FM_ASSETS_ROOT", PROJECT_ROOT / "data" / "foundation_models")
MODELS_DIR = _env_path("HETCLOP_MODELS_DIR", PROJECT_ROOT / "data" / "models")

CACHE_DIR = _env_path("HETCLOP_CACHE_DIR", PROJECT_ROOT / "data" / "embeddings")
RESULTS_DIR = _env_path("HETCLOP_RESULTS_DIR", PROJECT_ROOT / "results")
ORTHOLOG_TABLE_PATH = _env_path(
    "HETCLOP_ORTHOLOG_TABLE", PROJECT_ROOT / "data" / "orthologs" / "human_mouse_orthologs.tsv"
)
