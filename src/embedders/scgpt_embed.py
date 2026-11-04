"""scGPT (whole-human) zero-shot cell encoder, ported to HetCLOP.

This is a faithful, self-contained port of a standalone scGPT cell
encoder, the single most decision-relevant baseline for
the HetCLOP Phase 2a decoder experiment. Only the ENCODE path is ported (we use
scGPT purely as an encoder here; decoding is done by a separately-trained,
encoder-agnostic MLP, not by scGPT's native generate()).

Weights are loaded READ-ONLY from models/scgpt_human/
(best_model.pt, vocab.json, args.json); the weights are never modified.

The port avoids the scgpt pip package / torchtext entirely by reimplementing the
minimal GeneEncoder + ContinuousValueEncoder + standard nn.TransformerEncoder
path and the [CLS]-token cell-embedding readout, as in the reference implementation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from src.utils.paths import CORPUS_SOURCE_ROOT

logger = logging.getLogger(__name__)

DEFAULT_SCGPT_DIR = CORPUS_SOURCE_ROOT / "models" / "scgpt_human"


class SimpleGeneVocab:
    """Minimal gene vocabulary: JSON dict of {gene_name: int_id} (no torchtext)."""

    def __init__(self, token2idx: dict):
        self._token2idx = token2idx
        self._idx2token = {v: k for k, v in token2idx.items()}
        self._default_idx = None

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SimpleGeneVocab":
        with open(path) as f:
            return cls(json.load(f))

    def __getitem__(self, token: str) -> int:
        if token in self._token2idx:
            return self._token2idx[token]
        if self._default_idx is not None:
            return self._default_idx
        raise KeyError(f"Token '{token}' not in vocabulary")

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def set_default_index(self, idx: int):
        self._default_idx = idx

    def append_token(self, token: str):
        if token not in self._token2idx:
            idx = max(self._token2idx.values()) + 1
            self._token2idx[token] = idx
            self._idx2token[idx] = token


class GeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.enc_norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class ExprDecoder(nn.Module):
    """Present only so pretrained decoder weights load cleanly; unused in encode."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        return dict(pred=self.fc(x).squeeze(-1))


class ClsDecoder(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.ReLU())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x):
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    def __init__(self, d_model, arch_style="inner product", **_):
        super().__init__()
        self.arch_style = arch_style
        self.gene2query = nn.Linear(d_model, d_model)
        self.query_activation = nn.Sigmoid()
        self.W = nn.Linear(d_model, d_model, bias=False)


class TransformerModel(nn.Module):
    """Minimal scGPT encoder (bowang-lab/scGPT), encode path only."""

    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, vocab=None,
                 dropout=0.5, pad_token="<pad>", cell_emb_style="cls"):
        super().__init__()
        self.d_model = d_model
        self.cell_emb_style = cell_emb_style
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token] if vocab else 0)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.decoder = ExprDecoder(d_model)
        self.cls_decoder = ClsDecoder(d_model, 1, nlayers=3)
        self.mvc_decoder = MVCDecoder(d_model)

    def _encode(self, src, values, src_key_padding_mask):
        src = self.encoder(src)
        values = self.value_encoder(values)
        total_embs = src + values
        return self.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)


def _load_pretrained(model, pretrained_params):
    if isinstance(pretrained_params, dict) and "model_state_dict" in pretrained_params:
        pretrained_params = pretrained_params["model_state_dict"]
    # flash-attn Wqkv -> standard nn.MultiheadAttention in_proj key rename
    pretrained_params = {k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()}
    model_dict = model.state_dict()
    matched = {k: v for k, v in pretrained_params.items()
               if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return len(matched), len(model_dict)


class CellDataCollator:
    def __init__(self, pad_token_id, pad_value, max_length, n_bins=51):
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.max_length = max_length
        self.n_bins = n_bins

    def __call__(self, batch):
        gene_list = [b["genes"] for b in batch]
        value_list = [b["expressions"] for b in batch]
        max_len = min(max(len(g) for g in gene_list), self.max_length)
        padded_genes, padded_values = [], []
        for genes, values in zip(gene_list, value_list):
            if len(genes) > max_len:
                idx = np.random.choice(len(genes) - 1, max_len - 1, replace=False) + 1
                idx = np.concatenate([[0], idx])
                genes = genes[idx]
                values = values[idx]
            pad_len = max_len - len(genes)
            if pad_len > 0:
                genes = torch.cat([genes, torch.full((pad_len,), self.pad_token_id, dtype=genes.dtype)])
                values = torch.cat([values, torch.full((pad_len,), self.pad_value, dtype=values.dtype)])
            padded_genes.append(genes)
            padded_values.append(values)
        result = {"gene": torch.stack(padded_genes), "expr": torch.stack(padded_values)}
        expr = result["expr"]
        mask = expr != self.pad_value
        if mask.any():
            non_zero = expr[mask][expr[mask] > 0]
            if non_zero.numel() > 0:
                bins = torch.quantile(
                    non_zero.float(), torch.linspace(0, 1, self.n_bins + 1).to(expr.device)
                )
                binned = torch.bucketize(expr, bins)
                binned[~mask] = self.pad_value
                result["expr"] = binned.long()
        return result


class ScGPTCellEncoder:
    def __init__(self, model_dir=DEFAULT_SCGPT_DIR, device="cuda", max_length=1200, batch_size=64):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        self.vocab = None
        self.model_configs = None

    def _load(self):
        if self.model is not None:
            return
        vocab_file = self.model_dir / "vocab.json"
        config_file = self.model_dir / "args.json"
        model_file = self.model_dir / "best_model.pt"
        for f in [vocab_file, config_file, model_file]:
            if not f.exists():
                raise FileNotFoundError(f"Missing {f.name} in {self.model_dir}")
        self.vocab = SimpleGeneVocab.from_file(vocab_file)
        with open(config_file) as f:
            self.model_configs = json.load(f)
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.model = TransformerModel(
            ntoken=len(self.vocab), d_model=self.model_configs["embsize"],
            nhead=self.model_configs["nheads"], d_hid=self.model_configs["d_hid"],
            nlayers=self.model_configs["nlayers"], vocab=self.vocab,
            dropout=self.model_configs.get("dropout", 0.0),
        )
        state_dict = torch.load(model_file, map_location=self.device)
        n_matched, n_total = _load_pretrained(self.model, state_dict)
        logger.info(f"scGPT loaded {n_matched}/{n_total} tensors from {model_file.name}")
        self.model.to(self.device).eval()

    @property
    def embed_dim(self) -> int:
        self._load()
        return self.model_configs["embsize"]

    @torch.no_grad()
    def encode(self, adata: ad.AnnData) -> np.ndarray:
        """Encode cells -> (N, embsize) via scGPT [CLS] readout. Raw counts are
        taken from adata.layers['counts'] if present, else adata.X. Gene names
        come from adata.var_names (expected to be UPPERCASE human symbols)."""
        self._load()
        genes = [str(g) for g in adata.var_names]
        gene_ids = np.array([self.vocab[g] if g in self.vocab else -1 for g in genes], dtype=int)
        valid_mask = gene_ids >= 0
        n_matched = int(valid_mask.sum())
        if n_matched < 100:
            gene_ids_u = np.array(
                [self.vocab[g.upper()] if g.upper() in self.vocab else -1 for g in genes], dtype=int
            )
            if (gene_ids_u >= 0).sum() > n_matched:
                gene_ids, valid_mask = gene_ids_u, gene_ids_u >= 0
                n_matched = int(valid_mask.sum())
        if n_matched < 100:
            raise ValueError(
                f"only {n_matched}/{len(genes)} genes matched scGPT vocab (min 100)"
            )
        gene_ids_f = gene_ids[valid_mask]

        if "counts" in adata.layers:
            cm = adata.layers["counts"]
        else:
            cm = adata.X
        cm = cm[:, valid_mask]
        cm = cm.toarray() if sp.issparse(cm) else np.asarray(cm)
        cm = cm.astype(np.float32)

        pad_token_id = self.vocab["<pad>"]
        cls_token_id = self.vocab["<cls>"]
        pad_value = self.model_configs.get("pad_value", -2)

        class CellDataset(Dataset):
            def __init__(self, counts, gids):
                self.counts, self.gids = counts, gids

            def __len__(self):
                return len(self.counts)

            def __getitem__(self, idx):
                row = self.counts[idx]
                nz = np.nonzero(row)[0]
                values = row[nz]
                gids = self.gids[nz]
                gids = np.insert(gids, 0, cls_token_id)
                values = np.insert(values, 0, pad_value)
                return {"genes": torch.from_numpy(gids).long(),
                        "expressions": torch.from_numpy(values).float()}

        dataset = CellDataset(cm, gene_ids_f)
        collator = CellDataCollator(pad_token_id, pad_value, self.max_length,
                                    n_bins=self.model_configs.get("n_bins", 51))
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            sampler=SequentialSampler(dataset), collate_fn=collator,
                            drop_last=False, num_workers=0, pin_memory=True)

        emb_out = np.zeros((len(dataset), self.model_configs["embsize"]), dtype=np.float32)
        count = 0
        use_cuda = self.device.type == "cuda"
        for data_dict in loader:
            input_gene_ids = data_dict["gene"].to(self.device)
            input_values = data_dict["expr"].float().to(self.device)
            mask = input_gene_ids.eq(pad_token_id)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                output = self.model._encode(input_gene_ids, input_values, src_key_padding_mask=mask)
            emb = output[:, 0, :].float().cpu().numpy()
            emb_out[count:count + len(emb)] = emb
            count += len(emb)
        return emb_out


def embed(adata: ad.AnnData, device="cuda", max_length=1200, batch_size=64) -> np.ndarray:
    """Convenience: encode an AnnData (raw counts, human-symbol var_names) to a
    (N, 512) scGPT cell-embedding array."""
    return ScGPTCellEncoder(device=device, max_length=max_length, batch_size=batch_size).encode(adata)
