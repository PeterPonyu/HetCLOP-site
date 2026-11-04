"""Phase 2v: END-TO-END generation test of the dispersion design rule.

The paper's biological-cost claim is mean-fixed (latents taken from real cells). This
experiment is the end-to-end arm: latents are SAMPLED from the
fitted scVI prior and decoded through the full generative path; labels are never used
by the generator (cell-type labels for evaluation are transferred by 1-NN in the
model's own latent space against real TRAIN cells).

Arms (same corpus, same splits, same fit protocol as phase2q/phase2j):
  (a) scVI `gene`       -- the deployed global-dispersion default; NB(mu(z), theta_g)
  (b) scVI `gene-cell`  -- conditional amortised dispersion;      NB(mu(z), theta_ng(z))
  (c) scVI `gene-cell` decoded dispersion-free:                   Poisson(mu(z))
Readouts (identical code path as the mean-fixed biological readout, run_biological.py):
marker recall@50 (Welch, real held-out markers), TSTR macro-F1, rare-type macro-F1
(rare = <50 real test cells, same 19 types). Real reference = held-out test cells.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2j_external_2026-07-09"))
sys.path.insert(0, str(ROOT / "experiments" / "phase2q_biological_2026-07-10"))
sys.path.insert(0, str(ROOT))

import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scvi  # noqa: E402

import phase2j_lib as L  # noqa: E402
from run_biological import MIN_CELLS, RARE_MAX, TOPK, marker_recall, marker_scores, tstr  # noqa: E402
from run_scvi import build_adata  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 1
N_PARAM_SAMPLES = 1
DEVICE = "cuda"
MAX_EPOCHS = 85  # the bio-phase fit converged (early-stopped) at 85; keep the cap

def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def fit_model(data, dispersion):
    scvi.settings.seed = SEED
    A_tr = build_adata(data, data["tr"])
    scvi.model.SCVI.setup_anndata(A_tr, layer="counts", batch_key="dataset")
    model = scvi.model.SCVI(A_tr, gene_likelihood="nb", dispersion=dispersion)
    model.train(accelerator="gpu" if DEVICE == "cuda" else "cpu",
                enable_progress_bar=False, early_stopping=True, max_epochs=MAX_EPOCHS)
    n_epochs = int(len(model.history["elbo_train"])) if "elbo_train" in model.history else -1
    return model, A_tr, n_epochs


def sample_generative(model, A_tr, data, n):
    """z ~ N(0,I); (dataset, library) resampled jointly from the train population;
    decode through the full generative path; return mu (n,G), theta (n,G), z (n,d)."""
    rng = np.random.default_rng(SEED)
    dev = next(model.module.parameters()).device
    n_lat = model.module.n_latent
    z = torch.randn(n, n_lat, device=dev)
    tr_idx = np.where(data["tr"])[0]
    pick = rng.integers(0, len(tr_idx), n)
    lib = torch.tensor(
        np.log(np.clip(data["raw"][tr_idx[pick]].sum(axis=1), 1.0, None)).astype(np.float32),
        device=dev).unsqueeze(-1)
    ds_codes = A_tr.obs["_scvi_batch"].to_numpy()
    batch_index = torch.tensor(ds_codes[pick].astype(np.int64), device=dev).unsqueeze(-1)
    with torch.no_grad():
        out = model.module.generative(z, lib, batch_index)
    px = out["px"]
    mu = px.mu.detach().cpu().numpy().astype(np.float64)
    theta = px.theta.detach().cpu().numpy().astype(np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, mu.shape).copy()
    return np.clip(mu, 0.0, None), np.clip(theta, 1e-8, None), z.detach().cpu().numpy()


def label_transfer(model, A_tr, data, z_syn):
    from sklearn.neighbors import NearestNeighbors
    z_tr = model.get_latent_representation(A_tr)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z_tr)
    idx = nn.kneighbors(z_syn, return_distance=False)[:, 0]
    return data["labels"][data["tr"]][idx]


def evaluate(name, syn_logX, lab_syn, real, lab_te, real_ms, types, rare_types):
    u_syn, c_syn = np.unique(lab_syn, return_counts=True)
    cover = {t: int(c_syn[list(u_syn).index(t)]) if t in set(u_syn) else 0 for t in types}
    scoreable = [t for t in types if cover[t] >= MIN_CELLS]
    syn_ms = marker_scores(syn_logX, lab_syn, scoreable)
    rec = marker_recall(real_ms, syn_ms, scoreable, TOPK)
    vals = np.array(list(rec.values())) if rec else np.array([float("nan")])
    macro, rare_macro, rare_full = tstr(syn_logX, lab_syn, real, lab_te, rare_types, seed=SEED)
    L.log(f"  [{name}] scoreable {len(scoreable)}/{len(types)} types; "
          f"marker recall mean={vals.mean():.4f} median={np.median(vals):.4f}; "
          f"TSTR macro={macro:.4f} rare={rare_macro:.4f} (unmasked {rare_full:.4f})")
    return {
        "n_scoreable_types": len(scoreable),
        "marker_recall_mean": float(vals.mean()), "marker_recall_median": float(np.median(vals)),
        "marker_recall_per_type": {t: float(rec[t]) for t in scoreable},
        "tstr_macro_f1": macro, "tstr_rare_f1": rare_macro, "tstr_rare_f1_unmasked": rare_full,
    }


def main() -> int:
    t0 = time.time()
    L.log("=== Phase 2v: end-to-end generation test (latents sampled from the prior) ===")
    data = L.load_corpus()
    real = data["logX"][data["te"]]
    lab_te = data["labels"][data["te"]]
    u, c = np.unique(lab_te, return_counts=True)
    types = [t for t, n in zip(u, c) if n >= MIN_CELLS]
    rare_types = {t for t, n in zip(u, c) if n < RARE_MAX}
    real_ms = marker_scores(real, lab_te, types)
    n_te = int(data["te"].sum())

    out = {"purpose": "end-to-end generation test of the dispersion design rule",
           "seed": SEED, "n_synthetic_per_arm": n_te,
           "n_types_real": len(types), "n_rare_types": len(rare_types), "arms": {}}
    for mode in ["gene", "gene-cell"]:
        L.log(f"fitting scVI dispersion={mode} ...")
        model, A_tr, n_epochs = fit_model(data, mode)
        L.log(f"  fit done ({n_epochs} epochs, {time.time() - t0:.0f}s); sampling {n_te} latents")
        mu, theta, z_syn = sample_generative(model, A_tr, data, n_te)
        lab_syn = label_transfer(model, A_tr, data, z_syn)

        draws = {f"e2e_nb_{mode}": _log1p(sample_nb(mu, theta, seed=SEED))}
        if mode == "gene-cell":
            draws["e2e_poisson"] = _log1p(sample_nb_fixed_theta(mu, float("inf"), seed=SEED))
        for name, X in draws.items():
            out["arms"][name] = evaluate(name, X, lab_syn, real, lab_te, real_ms, types, rare_types)
            out["arms"][name]["dispersion_mode"] = mode
            out["arms"][name]["n_epochs_trained"] = n_epochs
            out["arms"][name]["elapsed_s"] = round(time.time() - t0, 1)
        del model, A_tr
        torch.cuda.empty_cache()

    (RESULTS / out_file).write_text(json.dumps(out, indent=2))
    L.log(f"=== done in {time.time() - t0:.0f}s -> {RESULTS / out_file} ===")
    return 0


if __name__ == "__main__":
    import argparse  # noqa: E402
    p = argparse.ArgumentParser(description="Phase 2v: end-to-end generation test")
    p.add_argument("--seed", type=int, default=None, help="Random seed for scVI and draws")
    p.add_argument("--out", type=str, default="e2e_generation.json", help="Output JSON filename")
    args = p.parse_args()
    if args.seed is not None:
        SEED = args.seed
    out_file = args.out
    raise SystemExit(main())
