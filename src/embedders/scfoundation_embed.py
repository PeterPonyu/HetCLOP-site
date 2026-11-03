"""scFoundation zero-shot cell embedder, adapted from
an earlier embedding script to add human-mouse ortholog
resolution (the original script's SYM2I lookup only ever matched human gene
symbols against the 19,264-gene panel, so mouse atlases were skipped
entirely: `if len(gi) < 2000: SKIP (human-only)`). Bypasses the uninstallable
`pretrainmodels` package; encoder-only path (12-layer transformer, decoder
dropped) faithful to biomap-research/scFoundation's get_embedding.py recipe.
"""
from __future__ import annotations

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

from src.embedders.validation import assert_raw_counts
from src.orthologs.mapping import map_mouse_symbols_to_human, normalize_symbol
from src.utils.paths import MODELS_DIR, PROJECT_ROOT, FM_ASSETS_ROOT

sys.path.insert(0, str(FM_ASSETS_ROOT / "scripts" / "scf_repo"))
import mae_autobin  # noqa: E402  (vendored module, path set above)
import transformer  # noqa: E402

CHECKPOINT = MODELS_DIR / "scFoundation-cell" / "model.pt"
GENE_INDEX_PATH = PROJECT_ROOT / "data" / "gene_panels" / "scfoundation_gene_index.tsv"
PAD_TOKEN_ID = 103

_gene_list = list(pd.read_csv(GENE_INDEX_PATH, header=0, delimiter="\t")["gene_name"])
_sym2i = {g: i for i, g in enumerate(_gene_list)}
N_GENES = len(_gene_list)
assert N_GENES == 19264, N_GENES

_device = "cuda" if torch.cuda.is_available() else "cpu"
_state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
if isinstance(_state_dict, dict) and "cell" in _state_dict:
    _state_dict = _state_dict["cell"]
_SEQ_LEN = _state_dict["pos_emb.weight"].shape[0] - 1

_model = mae_autobin.MaeAutobin(
    num_tokens=N_GENES, max_seq_len=_SEQ_LEN, embed_dim=768, decoder_embed_dim=512,
    bin_alpha=1.0, bin_num=100, pad_token_id=PAD_TOKEN_ID, mask_token_id=102,
)
_model.encoder = transformer.pytorchTransformerModule(max_seq_len=_SEQ_LEN, dim=768, depth=12, heads=12)
_model.decoder = torch.nn.Identity()
_missing, _ = _model.load_state_dict(_state_dict, strict=False)
_missing = [k for k in _missing if not k.startswith("decoder.")]
assert not _missing, f"unexpected missing keys: {_missing}"
_model = _model.to(_device).eval()
if _device == "cuda":
    _model = _model.half()

CAP = 2046


def _resolve_gene_columns(var_names: list[str], species: str, ortholog_table: dict[str, str] | None):
    if species == "mouse":
        if ortholog_table is None:
            raise ValueError("species='mouse' requires an ortholog_table (see src.orthologs.mapping)")
        resolved = map_mouse_symbols_to_human(var_names, ortholog_table)
    else:
        resolved = [normalize_symbol(s) for s in var_names]
    return [i for i, s in enumerate(resolved) if s is not None and s in _sym2i], resolved


@torch.no_grad()
def _gather(data: torch.Tensor, labels: torch.Tensor, pad_token_id: int):
    value_nums = labels.sum(1)
    max_num = int(max(value_nums))
    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device)
    data = torch.hstack([data, fake_data])
    fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float("inf"), device=labels.device)
    tmp = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels = labels + tmp
    labels = torch.hstack([labels, fake_label])
    idx = labels.topk(max_num).indices
    new_data = torch.gather(data, 1, idx)
    padding = new_data == pad_token_id
    return new_data, padding


@torch.no_grad()
def _forward(value_matrix: torch.Tensor, gene_ids: torch.Tensor) -> np.ndarray:
    gene_part = value_matrix[:, :N_GENES]
    keep_mask = torch.zeros_like(value_matrix, dtype=torch.bool)
    if N_GENES > CAP:
        top_val, top_idx = gene_part.topk(CAP, dim=1)
        keep_mask[:, :N_GENES].scatter_(1, top_idx, top_val > 0)
    else:
        keep_mask[:, :N_GENES] = gene_part > 0
    keep_mask[:, N_GENES:] = value_matrix[:, N_GENES:] > 0

    x, x_pad = _gather(value_matrix, keep_mask, PAD_TOKEN_ID)
    pos, _ = _gather(gene_ids.repeat(value_matrix.shape[0], 1), keep_mask, PAD_TOKEN_ID)
    xt = x.unsqueeze(2).half() if _device == "cuda" else x.unsqueeze(2).float()
    h = _model.token_emb(xt, output_weight=0) + _model.pos_emb(pos)
    g = _model.encoder(h, padding_mask=x_pad)
    e1 = g[:, -1, :]
    e2 = g[:, -2, :]
    e3, _ = torch.max(g[:, :-2, :], dim=1)
    e4 = torch.mean(g[:, :-2, :], dim=1)
    return torch.concat([e1, e2, e3, e4], dim=1).float().cpu().numpy()


def embed(
    adata: ad.AnnData,
    ctcol: str,
    bcol: str,
    out: str,
    species: str = "human",
    ortholog_table: dict[str, str] | None = None,
) -> None:
    var_names = [str(s) for s in adata.var_names]
    gene_idx, resolved = _resolve_gene_columns(var_names, species, ortholog_table)
    if len(gene_idx) < 2000:
        raise ValueError(
            f"only {len(gene_idx)} of {len(var_names)} genes mapped to the scFoundation "
            f"19,264-gene panel (species={species}); refusing to embed on near-zero overlap"
        )
    target_cols = np.array([_sym2i[resolved[i]] for i in gene_idx], dtype=np.int64)
    X = adata.X[:, gene_idx]
    assert_raw_counts(X, context="scfoundation_embed.embed")
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    n = adata.n_obs
    out_dim = 768 * 4
    embeddings = np.zeros((n, out_dim), dtype=np.float32)
    batch_size = 4
    gene_ids = torch.arange(_SEQ_LEN, device=_device).unsqueeze(0)

    for start in range(0, n, batch_size):
        rows = []
        for r in range(start, min(start + batch_size, n)):
            v = np.zeros(N_GENES, dtype=np.float64)
            xr = X[r].toarray().ravel()
            v[target_cols] = xr
            total = v.sum()
            if total <= 0:
                rows.append(None)
                continue
            log_norm = np.log1p(v / total * 1e4)
            log_total = np.log10(total)
            rows.append(np.concatenate([log_norm, [log_total, log_total]]))
        valid = [i for i, row in enumerate(rows) if row is not None]
        if not valid:
            continue
        mat = torch.tensor(np.stack([rows[i] for i in valid]), dtype=torch.float32, device=_device)
        merged = _forward(mat, gene_ids)
        for j, i in enumerate(valid):
            embeddings[start + i] = merged[j]

    np.savez(
        out,
        X=embeddings,
        y=adata.obs[ctcol].astype(str).values,
        batch=adata.obs[bcol].astype(str).values,
    )
