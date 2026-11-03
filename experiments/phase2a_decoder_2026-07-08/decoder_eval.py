"""MLP decoder (encoder-agnostic) + evaluation helpers for Phase 2a.

The decoder architecture and training recipe mirror an encoder-agnostic
LinearDecoder / train_mlp_decoder.py exactly (embed_dim -> 1024 -> 2048 ->
num_genes; Linear+LayerNorm+GELU+Dropout(0.1); output_activation='none'; MSE
loss; AdamW lr=1e-3 wd=1e-4; cosine schedule; 50 epochs; batch 256; grad-clip
1.0), so the only thing that varies across encoders is the input embed_dim.
It is reimplemented here (not imported from another package) to honour the boundary
rule that all new code lives under HetCLOP/."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_genes: int, hidden_dims=(1024, 2048), dropout=0.1):
        super().__init__()
        layers = []
        dims = [embed_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                       nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_genes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_decoder(
    Z_train, Y_train, Z_val, Y_val, Z_test,
    hidden_dims=(1024, 2048), epochs=50, batch_size=256, lr=1e-3,
    weight_decay=1e-4, seed=42, device="cuda",
):
    """Train an MLP decoder Z->Y; early-stop on val MSE; return test predictions,
    val loss, and per-epoch history. All arrays are numpy float32."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embed_dim = Z_train.shape[1]
    num_genes = Y_train.shape[1]

    Xtr = torch.from_numpy(Z_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Xva = torch.from_numpy(Z_val).float().to(dev)
    Yva = torch.from_numpy(Y_val).float().to(dev)

    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=0)
    model = MLPDecoder(embed_dim, num_genes, hidden_dims=hidden_dims).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva), Yva).item()
        history.append({"epoch": ep + 1, "train": float(np.mean(losses)), "val": vloss})
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Z_test), 4096):
            xb = torch.from_numpy(Z_test[i:i + 4096]).float().to(dev)
            preds.append(model(xb).cpu().numpy())
    test_pred = np.concatenate(preds, axis=0).astype(np.float32)
    return test_pred, float(best_val), history, int(n_params)


def reconstruction_metrics(pred, target):
    """Mean per-gene Pearson r (genes with nonzero variance) and MSE."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2))
    pc = pred - pred.mean(0, keepdims=True)
    tc = target - target.mean(0, keepdims=True)
    pv = np.sqrt((pc ** 2).sum(0))
    tv = np.sqrt((tc ** 2).sum(0))
    valid = (pv > 1e-8) & (tv > 1e-8)
    r = np.zeros(pred.shape[1])
    r[valid] = (pc[:, valid] * tc[:, valid]).sum(0) / (pv[valid] * tv[valid])
    return {"mean_gene_pearson": float(r[valid].mean()) if valid.any() else float("nan"),
            "n_genes_scored": int(valid.sum()), "mse": mse}
