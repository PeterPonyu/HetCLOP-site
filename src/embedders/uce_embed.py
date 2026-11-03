"""UCE (Universal Cell Embedding) zero-shot cell embedder, adapted from
an earlier embedding script to thread `species` through instead of
hardcoding 'human'. UCE's downloaded protein-embeddings archive already
contains a mouse file (Mus_musculus.GRCm39...pt); the original wrapper never
used it. Drives the vendored snap-stanford/UCE eval_single_anndata.main
per-atlas (4-layer model), reads obsm['X_uce'] (1280-d).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import tarfile
import types

os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

# numba stubbed out (env numpy version cap); mirrors the original script.
_numba = types.ModuleType("numba")


def _identity(*args, **kwargs):
    return args[0] if (args and callable(args[0])) else (lambda f: f)


_numba.njit = _numba.jit = _identity
_numba.prange = range
_numba.vectorize = _numba.guvectorize = _identity
_numba.config = types.SimpleNamespace(NUMBA_NUM_THREADS=1)
_numba.set_num_threads = lambda *a, **k: None
for _sub in ["core", "types", "typed", "extending", "core.types"]:
    sys.modules[f"numba.{_sub}"] = types.ModuleType(f"numba.{_sub}")
_numba.types = sys.modules["numba.types"]
sys.modules["numba"] = _numba

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402

from src.embedders.validation import assert_raw_counts  # noqa: E402
from src.utils.paths import CACHE_DIR, MODELS_DIR, FM_ASSETS_ROOT  # noqa: E402

UCE_REPO = FM_ASSETS_ROOT / "scripts" / "uce_repo"
DOWNLOAD_DIR = MODELS_DIR / "UCE"
MODEL_FILES_DIR = UCE_REPO / "model_files"
os.makedirs(MODEL_FILES_DIR, exist_ok=True)

for _fn in ["4layer_model.torch", "all_tokens.torch", "species_chrom.csv", "species_offsets.pkl"]:
    dst, src = MODEL_FILES_DIR / _fn, DOWNLOAD_DIR / _fn
    if src.exists() and not dst.exists():
        os.symlink(src, dst)

if not (MODEL_FILES_DIR / "protein_embeddings").is_dir():
    with tarfile.open(DOWNLOAD_DIR / "protein_embeddings.tar.gz") as tar:
        tar.extractall(MODEL_FILES_DIR)
    if not (MODEL_FILES_DIR / "protein_embeddings").is_dir():
        for d in glob.glob(str(MODEL_FILES_DIR / "*protein*")):
            if os.path.isdir(d):
                os.symlink(d, MODEL_FILES_DIR / "protein_embeddings")
                break

os.chdir(UCE_REPO)
sys.path.insert(0, str(UCE_REPO))

GENEFORMER_DICT_DIR = MODELS_DIR / "Geneformer" / "geneformer"
_HUMAN_SYMBOLS = set(pickle.load(open(GENEFORMER_DICT_DIR / "gene_name_id_dict_gc104M.pkl", "rb")).keys())

from accelerate import Accelerator  # noqa: E402
from eval_single_anndata import main as _uce_main  # noqa: E402

_WORK_DIR = CACHE_DIR / "uce_work"
os.makedirs(_WORK_DIR, exist_ok=True)


def _args_for(adata_path: str, species: str) -> argparse.Namespace:
    return argparse.Namespace(
        adata_path=adata_path, dir=str(_WORK_DIR) + "/", species=species, filter=True, skip=True,
        model_loc=str(MODEL_FILES_DIR / "4layer_model.torch"), batch_size=25, pad_length=1536,
        pad_token_idx=0, chrom_token_left_idx=1, chrom_token_right_idx=2, cls_token_idx=3,
        CHROM_TOKEN_OFFSET=143574, sample_size=1024, CXG=True, nlayers=4, output_dim=1280,
        d_hid=5120, token_dim=5120, multi_gpu=False,
        spec_chrom_csv_path=str(MODEL_FILES_DIR / "species_chrom.csv"),
        token_file=str(MODEL_FILES_DIR / "all_tokens.torch"),
        protein_embeddings_dir=str(MODEL_FILES_DIR / "protein_embeddings") + "/",
        offset_pkl_path=str(MODEL_FILES_DIR / "species_offsets.pkl"),
    )


def run(src_h5ad: str, ctcol: str, bcol: str, name: str, species: str = "human") -> str:
    out_dir = CACHE_DIR / "fm_emb"
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f"uce_{name}.npz")
    if os.path.exists(out_path):
        return out_path

    adata = ad.read_h5ad(src_h5ad)
    assert_raw_counts(adata.X, context="uce_embed.run")
    if species == "human":
        overlap = len(set(str(s) for s in adata.var_names) & _HUMAN_SYMBOLS)
        if overlap < 2000:
            raise ValueError(f"human-symbol overlap {overlap} < 2000; refusing to embed {name}")
    # species == "mouse": UCE's own protein_embeddings archive already contains a
    # mouse file (Mus_musculus.*), keyed by mouse gene symbol; no overlap guard
    # equivalent to _HUMAN_SYMBOLS is applied here — UCE's own eval_single_anndata
    # pipeline performs its own species-specific vocabulary filtering internally.

    work_h5 = str(_WORK_DIR / f"{name}.h5ad")
    adata.write(work_h5)
    accelerator = Accelerator(project_dir=str(_WORK_DIR))
    _uce_main(_args_for(work_h5, species), accelerator)

    emb_path = str(_WORK_DIR / f"{name}_uce_adata.h5ad")
    embedded = ad.read_h5ad(emb_path)
    X = np.asarray(embedded.obsm["X_uce"], dtype=np.float32)
    np.savez(
        out_path, X=X, y=adata.obs[ctcol].astype(str).values, batch=adata.obs[bcol].astype(str).values
    )
    return out_path
