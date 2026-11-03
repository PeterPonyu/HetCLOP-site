"""Phase 2j -- scVI arm. Fit the REAL scvi-tools SCVI model on our corpus and run
the within-decoder Poisson-vs-NB dispersion control on its OWN fitted mean.

scVI is the deployed model (Lopez et al. 2018; ``scvi-tools`` 1.4.3, installed).
We use its OWN VAE encoder/decoder -- its latent is the conditioning variable,
exactly as the brief requires -- and fit it TWICE:
  * dispersion="gene"      -- global gene-wise theta (CFGen-like; F4-indicted).
  * dispersion="gene-cell" -- per-cell amortized theta (F4 predicts a WEAKER or
    vanishing Poisson-beats-NB effect here, since a per-cell dispersion can in
    principle absorb what a global one cannot). Testing that is the point.

Training: on our train split (split==0) with batch_key=dataset (the 50 datasets),
gene_likelihood="nb", scvi defaults + early stopping. Inference on the held-out
test split (split==2). The model's fitted mean mu_ng and dispersion theta are read
via ``get_likelihood_parameters`` (give_mean over latent draws). Then the SAME
within-decoder control as Phase 2c/2d/2f: draw NB(mu,theta) vs Poisson(mu), mu
fixed, evaluate on log1p, paired + dataset-clustered on the two valid metrics.

No tuning: scvi defaults are used as published; the dispersion contrast holds mu
fixed in both arms so it is invariant to how well mu is fit anyway.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scvi  # noqa: E402

import phase2j_lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def build_adata(data, mask):
    A = ad.AnnData(
        sp.csr_matrix(data["raw"][mask].astype(np.float32)),
        obs={"dataset": data["dataset_ids"][mask].astype(str)},
    )
    A.layers["counts"] = A.X.copy()
    return A


def fit_and_extract(data, dispersion, seed, n_param_samples, max_epochs, device):
    """Train SCVI (given dispersion mode) on the train split, return per-test-cell
    fitted mean mu (count space) and dispersion theta (broadcast to (n_te, G))."""
    scvi.settings.seed = seed
    A_tr = build_adata(data, data["tr"])
    scvi.model.SCVI.setup_anndata(A_tr, layer="counts", batch_key="dataset")
    model = scvi.model.SCVI(A_tr, gene_likelihood="nb", dispersion=dispersion)
    accel = "gpu" if device == "cuda" else "cpu"
    train_kwargs = dict(accelerator=accel, enable_progress_bar=False,
                        early_stopping=True)
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    model.train(**train_kwargs)

    A_te = build_adata(data, data["te"])
    params = model.get_likelihood_parameters(
        adata=A_te, n_samples=n_param_samples, give_mean=True)
    mu = np.asarray(params["mean"], dtype=np.float64)              # (n_te, G) count-space NB mean
    theta = np.asarray(params["dispersions"], dtype=np.float64)    # (n_te, G) inverse-dispersion
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, mu.shape).copy()
    n_epochs = int(len(model.history["elbo_train"])) if "elbo_train" in model.history else -1
    return mu, theta, {"n_epochs_trained": n_epochs, "dispersion": dispersion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", type=str, default="gene,gene-cell")
    ap.add_argument("--n-param-samples", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--energy-redraws", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    modes = args.modes.split(",")

    L.log(f"loading corpus; scVI seeds={seeds} modes={modes}")
    data = L.load_corpus()
    det = L.deterministic_baseline_log_pearson(data, seed=42, device=args.device)
    L.log(f"deterministic PCA-512 baseline log1p Pearson = {det:.4f}")

    out = {"model_family": "scVI (scvi-tools 1.4.3, real deployed model)",
           "deterministic_baseline_log_pearson": det,
           "seeds": seeds, "modes": modes,
           "n_param_samples": args.n_param_samples,
           "results": {m: {} for m in modes}}
    for mode in modes:
        for seed in seeds:
            name = f"scvi_{mode.replace('-', '_')}_seed{seed}"
            L.log(f"=== training {name} ===")
            mu, theta, meta = fit_and_extract(
                data, mode, seed, args.n_param_samples, args.max_epochs, args.device)
            res = L.evaluate_dispersion_control(
                name, mu, theta, data, det, seed=seed,
                n_energy_redraws=args.energy_redraws)
            res["train_meta"] = meta
            out["results"][mode][seed] = res
            (RESULTS / "scvi.json").write_text(json.dumps(out, indent=2))
    L.log(f"=== scVI arm done -> {RESULTS / 'scvi.json'} ===")


if __name__ == "__main__":
    main()
