"""Phase 2j -- CFGen arm. Train CFGen's OWN decoder code on our corpus and run the
within-decoder Poisson-vs-NB dispersion control on its fitted mean.

Uses CFGen's actual ``EncoderModel`` class (vendored VERBATIM from
theislab/CFGen @ commit in cfgen_vendored/CFGEN_COMMIT.txt; see
cfgen_vendored/NOTICE.md) -- the exact autoencoder/decoder that defines p(x|z,l):
  * encode:  z = h(log1p(counts))                       (CFGen normalization log_gexp)
  * decode:  mu = softmax(decoder(z)) * size_factor     (size_factor = per-cell total counts)
  * likelihood: NegativeBinomial(mu, theta=exp(self.theta)), self.theta a GLOBAL
    gene-wise parameter (covariate_specific_theta=False, CFGen's default) -- the
    per-cell-latent-mean + separately-fitted global gene-wise dispersion arrangement
    F4 indicts.

This is CFGen's decoder code, trained with a MATCHED custom loop (CFGen defaults:
encoder dims [512,256,50]/default.yaml; AdamW lr=1e-3 wd=1e-5/configure_optimizers;
batch 256). The flow-matching stage is NOT run -- irrelevant to F4, which is a
within-decoder control on p(x|z,l) only. It is NOT the full CFGen pipeline and is
never labelled "CFGen" in the results/report as if it were the deployed end-to-end
model; it is "CFGen's decoder (real code, matched loop)".

The dispersion contrast holds mu fixed in both arms, so it is invariant to the
training-loop choice; no tuning is done.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cfgen_vendored"))  # vendored CFGen code (verbatim)
import phase2j_lib as L  # noqa: E402
from scvi.distributions import NegativeBinomial  # noqa: E402
from cfgen.models.base.encoder_model import EncoderModel  # noqa: E402  (CFGen's real class)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# CFGen defaults (configs_encoder/encoder/default.yaml + training_config/default.yaml)
ENC_DIMS = [512, 256, 50]
LR, WD, BATCH = 1e-3, 1e-5, 256


def build_model(n_genes, n_cat, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc_kwargs = {"rna": {"dims": list(ENC_DIMS), "batch_norm": True,
                          "dropout": False, "dropout_p": 0.0}}
    return EncoderModel(
        in_dim={"rna": n_genes}, encoder_kwargs=enc_kwargs,
        learning_rate=LR, weight_decay=WD,
        covariate_specific_theta=False,       # GLOBAL theta_g (CFGen default; F4-indicted)
        conditioning_covariate="dataset", n_cat=n_cat, is_binarized=False)


def _nb_nll_cfgen(model, x_norm, x_raw, size_factor):
    """One forward/NLL exactly as CFGen's EncoderModel._step (encoder_model.py
    L121-140): encode(X_norm) -> decode(z, size_factor) -> NB(mu, exp(theta))."""
    z = model.encode({"X_norm": {"rna": x_norm}})
    mu = model.decode(z, {"rna": size_factor})["rna"]
    px = NegativeBinomial(mu=mu, theta=torch.exp(model.theta))
    return -px.log_prob(x_raw).sum(1).mean()


def fit_and_extract(data, seed, max_epochs, patience, device):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logX, raw = data["logX"], data["raw"]
    tr, va, te = data["tr"], data["va"], data["te"]
    n_genes = raw.shape[1]
    n_cat = int(len(np.unique(data["dataset_ids"])))
    model = build_model(n_genes, n_cat, seed).to(dev)

    Xn_tr = torch.from_numpy(logX[tr]).float()
    Xr_tr = torch.from_numpy(raw[tr]).float()
    dl = DataLoader(TensorDataset(Xn_tr, Xr_tr), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=0)  # drop_last: BatchNorm needs >1
    Xn_va = torch.from_numpy(logX[va]).float().to(dev)
    Xr_va = torch.from_numpy(raw[va]).float().to(dev)
    sf_va = Xr_va.sum(1, keepdim=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        for xn, xr in dl:
            xn, xr = xn.to(dev), xr.to(dev)
            sf = xr.sum(1, keepdim=True)
            loss = _nb_nll_cfgen(model, xn, xr, sf)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(_nb_nll_cfgen(model, Xn_va, Xr_va, sf_va).item())
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    mus, thetas = [], []
    theta_g = torch.exp(model.theta).detach().cpu().numpy()
    with torch.no_grad():
        idx = np.where(te)[0]
        for i in range(0, len(idx), 4096):
            j = idx[i:i + 4096]
            xn = torch.from_numpy(logX[j]).float().to(dev)
            sf = torch.from_numpy(raw[j].sum(1, keepdims=True)).float().to(dev)
            z = model.encode({"X_norm": {"rna": xn}})
            mu = model.decode(z, {"rna": sf})["rna"]
            mus.append(mu.cpu().numpy())
    mu_te = np.concatenate(mus, axis=0).astype(np.float64)
    theta_te = np.broadcast_to(theta_g, mu_te.shape).copy()
    meta = {"n_epochs_run": ep + 1, "best_val_nb_nll": best_val,
            "theta_global_median": float(np.median(theta_g)),
            "encoder_dims": ENC_DIMS}
    return mu_te, theta_te, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--energy-redraws", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    L.log(f"loading corpus; CFGen(real decoder) seeds={seeds}")
    data = L.load_corpus()
    det = L.deterministic_baseline_log_pearson(data, seed=42, device=args.device)
    L.log(f"deterministic PCA-512 baseline log1p Pearson = {det:.4f}")

    out = {
        "model_family": "CFGen decoder (real EncoderModel code, vendored verbatim; "
                        "matched custom loop; NOT the full deployed pipeline; "
                        "flow stage not run -- see cfgen_vendored/NOTICE.md)",
        "arrangement": "mu = softmax(h_psi(z)) * size_factor; global gene-wise "
                       "theta_g = exp(self.theta); covariate_specific_theta=False",
        "deterministic_baseline_log_pearson": det, "seeds": seeds,
        "results": {},
    }
    for seed in seeds:
        name = f"cfgen_decoder_seed{seed}"
        L.log(f"=== training {name} (CFGen real EncoderModel) ===")
        mu, theta, meta = fit_and_extract(
            data, seed, args.max_epochs, args.patience, args.device)
        res = L.evaluate_dispersion_control(
            name, mu, theta, data, det, seed=seed, n_energy_redraws=args.energy_redraws)
        res["train_meta"] = meta
        out["results"][seed] = res
        (RESULTS / "cfgen.json").write_text(json.dumps(out, indent=2))
    L.log(f"=== CFGen arm done -> {RESULTS / 'cfgen.json'} ===")


if __name__ == "__main__":
    main()
