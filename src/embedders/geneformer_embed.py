"""Geneformer-V2-104M zero-shot cell embedder, adapted from
an earlier embedding script to add human-mouse ortholog
resolution. Faithful rank-value tokenization: norm-to-10k / gene-median,
rank desc, <cls>+genes+<eos>, mean-pool gene tokens. Bypasses the official
`geneformer` pip package, which does not install cleanly against modern torch.
"""
from __future__ import annotations

import os
import pickle

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
import sys

sys.modules["torchvision"] = None  # ABI clash w/ modern torch; transformers skips it if absent
from transformers.models.bert.modeling_bert import BertModel

from src.embedders.validation import assert_raw_counts
from src.orthologs.mapping import map_mouse_symbols_to_human, normalize_symbol
from src.utils.paths import MODELS_DIR

GENEFORMER_DIR = MODELS_DIR / "Geneformer"
MODEL_DIR = os.environ.get("GF_MODEL_DIR", str(GENEFORMER_DIR / "Geneformer-V2-104M"))
DICT_DIR = GENEFORMER_DIR / "geneformer"
INPUT_LEN = 2048

_tok = pickle.load(open(DICT_DIR / "token_dictionary_gc104M.pkl", "rb"))
_med = pickle.load(open(DICT_DIR / "gene_median_dictionary_gc104M.pkl", "rb"))
_n2i = pickle.load(open(DICT_DIR / "gene_name_id_dict_gc104M.pkl", "rb"))
CLS, EOS, PAD = _tok["<cls>"], _tok["<eos>"], _tok["<pad>"]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = BertModel.from_pretrained(MODEL_DIR, output_hidden_states=True, add_pooling_layer=False)
_model = _model.to(_device).half().eval() if _device == "cuda" else _model.eval()


def _resolve_gene_ids(var_names: list[str], species: str, ortholog_table: dict[str, str] | None):
    """Map anndata var_names to Geneformer token IDs. For species='mouse',
    resolve mouse symbols to human orthologs first (the original script only
    ever looked up human symbols, so all mouse genes silently failed)."""
    if species == "mouse":
        if ortholog_table is None:
            raise ValueError("species='mouse' requires an ortholog_table (see src.orthologs.mapping)")
        resolved = map_mouse_symbols_to_human(var_names, ortholog_table)
    else:
        resolved = [normalize_symbol(s) for s in var_names]
    ensembl_ids = [_n2i.get(s) if s is not None else None for s in resolved]
    keep_idx = [i for i, e in enumerate(ensembl_ids) if e is not None and e in _tok and e in _med]
    return keep_idx, [ensembl_ids[i] for i in keep_idx]


def embed(
    adata: ad.AnnData,
    ctcol: str,
    bcol: str,
    out: str,
    species: str = "human",
    ortholog_table: dict[str, str] | None = None,
) -> None:
    var_names = [str(s) for s in adata.var_names]
    keep_idx, ensembl_ids = _resolve_gene_ids(var_names, species, ortholog_table)
    if len(keep_idx) < 200:
        raise ValueError(
            f"only {len(keep_idx)} of {len(var_names)} genes mapped to Geneformer tokens "
            f"(species={species}); refusing to embed on near-zero gene overlap"
        )
    token_ids = np.array([_tok[e] for e in ensembl_ids])
    medians = np.array([_med[e] for e in ensembl_ids], dtype=np.float32)

    X = adata.X[:, keep_idx]
    assert_raw_counts(X, context="geneformer_embed.embed")
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    n = adata.n_obs
    hidden = int(_model.config.hidden_size)
    embeddings = np.zeros((n, hidden), dtype=np.float32)
    batch_size = 16

    for start in range(0, n, batch_size):
        rows, lengths = [], []
        for r in range(start, min(start + batch_size, n)):
            x = X[r].toarray().ravel()
            total = x.sum()
            if total <= 0:
                rows.append([CLS, EOS])
                lengths.append(2)
                continue
            normalized = (x / total) * 1e4
            value = normalized / medians
            nonzero = np.where(value > 0)[0]
            order = nonzero[np.argsort(-value[nonzero])][: INPUT_LEN - 2]
            ids = [CLS] + token_ids[order].tolist() + [EOS]
            rows.append(ids)
            lengths.append(len(ids))

        max_len = max(lengths)
        input_ids = np.full((len(rows), max_len), PAD, dtype=np.int64)
        attn_mask = np.zeros((len(rows), max_len), dtype=np.int64)
        for i, ids in enumerate(rows):
            input_ids[i, : len(ids)] = ids
            attn_mask[i, : len(ids)] = 1

        with torch.no_grad():
            out_t = _model(
                input_ids=torch.tensor(input_ids, device=_device),
                attention_mask=torch.tensor(attn_mask, device=_device),
            )
            hidden_states = out_t.hidden_states[-1].float()

        mask = torch.tensor(attn_mask, device=_device).float().unsqueeze(-1).clone()
        mask[:, 0, :] = 0  # drop <cls>
        for i, ln in enumerate(lengths):
            mask[i, ln - 1, :] = 0  # drop <eos>
        cell_emb = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings[start : start + len(rows)] = cell_emb.cpu().numpy()

    np.savez(
        out,
        X=embeddings,
        y=adata.obs[ctcol].astype(str).values,
        batch=adata.obs[bcol].astype(str).values,
    )
