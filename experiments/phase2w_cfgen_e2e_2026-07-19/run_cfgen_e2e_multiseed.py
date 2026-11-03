"""Phase 2w: CFGen END-TO-END generation arm, 3 seeds.

A pre-submission review asked for
one more end-to-end arm with CFGen, the other named deployed default: the paper's claim
names "deployed generators such as CFGen and scVI" but only scVI had an end-to-end test
(phase2v). This script is the CFGen analogue of phase2v's run_e2e_multiseed.py:

  * Fit CFGen's real ``EncoderModel`` (vendored verbatim in
    phase2j_external_2026-07-09/cfgen_vendored @ CFGEN_COMMIT.txt) from scratch on the
    SAME corpus and SAME train/val/test split (phase2j_lib.load_corpus), with the SAME
    matched training loop as phase2j's run_cfgen.py (CFGen defaults: encoder dims
    [512,256,50], AdamW lr=1e-3 wd=1e-5, batch 256, val-NLL early stopping, patience 15,
    max 200 epochs). CFGen's decoder is a global-theta_g NB: mu = softmax(decoder(z)) *
    size_factor, theta_g = exp(self.theta), covariate_specific_theta=False.
  * LATENTS: CFGen's deployed latent generator is a conditional flow-matching model that
    is NOT part of the vendored subset (only the autoencoder is vendored), so end-to-end
    latents are sampled from a full-covariance Gaussian fit to the train cells' latent
    codes (the standard empirical-prior surrogate; ridge-stabilised). Size factors are
    resampled from the train population, mirroring phase2v's joint library resampling.
  * Labels for evaluation are transferred by 1-NN in CFGen's OWN latent space against
    real TRAIN cells; the generator never sees a label.
  * Arms per seed (n_synthetic = n_test, matching the scVI e2e arm):
      - e2e_cfgen_nb       : the deployed CFGen draw, NB(mu(z), theta_g)   [global theta_g]
      - e2e_cfgen_poisson  : dispersion-free limit on the same mean, Poisson(mu(z))
  * Readouts are the IDENTICAL code path as the scVI e2e arm: run_e2e.evaluate (marker
    recall Welch@50 vs real held-out markers, TSTR macro-F1, rare-type macro-F1 masked +
    unmasked; rare = <50 real test cells).

Per-seed JSONs e2e_cfgen_seed{1,2,3}.json plus aggregated e2e_cfgen_multiseed.json
(mean/std/median/min/max per metric) are written under results/.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PHASE2J = ROOT / "experiments" / "phase2j_external_2026-07-09"
sys.path.insert(0, str(PHASE2J / "cfgen_vendored"))  # before run_cfgen import (vendored pkg)
sys.path.insert(0, str(PHASE2J))
sys.path.insert(0, str(ROOT / "experiments" / "phase2q_biological_2026-07-10"))
sys.path.insert(0, str(ROOT / "experiments" / "phase2v_e2e_2026-07-19"))
sys.path.insert(0, str(ROOT))

import phase2j_lib as L  # noqa: E402
import run_e2e  # noqa: E402  (evaluate + _log1p; SEED patched per seed)
from run_biological import MIN_CELLS, RARE_MAX, marker_scores  # noqa: E402
from run_cfgen import BATCH, _nb_nll_cfgen, build_model  # noqa: E402  (phase2j fit protocol)
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [1, 2, 3]
DEVICE = "cuda"
MAX_EPOCHS = 200   # phase2j run_cfgen.py defaults
PATIENCE = 15


def fit_cfgen(data, seed):
    """Train CFGen's EncoderModel exactly as phase2j run_cfgen.fit_and_extract does
    (same loop, same early-stopping rule), but return the fitted model itself so we
    can encode train cells and decode sampled latents."""
    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logX, raw = data["logX"], data["raw"]
    tr, va = data["tr"], data["va"]
    n_genes = raw.shape[1]
    n_cat = int(len(np.unique(data["dataset_ids"])))
    model = build_model(n_genes, n_cat, seed).to(dev)  # sets torch/np seeds internally

    Xn_tr = torch.from_numpy(logX[tr]).float()
    Xr_tr = torch.from_numpy(raw[tr]).float()
    dl = DataLoader(TensorDataset(Xn_tr, Xr_tr), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=0)  # drop_last: BatchNorm needs >1
    Xn_va = torch.from_numpy(logX[va]).float().to(dev)
    Xr_va = torch.from_numpy(raw[va]).float().to(dev)
    sf_va = Xr_va.sum(1, keepdim=True)
    opt = torch.optim.AdamW(model.parameters(), lr=model.learning_rate,
                            weight_decay=model.weight_decay)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
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
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {"n_epochs_run": ep + 1, "best_val_nb_nll": best_val,
            "theta_global_median": float(np.median(torch.exp(model.theta).detach().cpu().numpy()))}
    return model, dev, meta


def encode_cells(model, Xn, dev, bs=8192):
    """z = CFGen encoder(log1p counts), batched."""
    zs = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], bs):
            xn = torch.from_numpy(Xn[i:i + bs]).float().to(dev)
            zs.append(model.encode({"X_norm": {"rna": xn}})["rna"].cpu().numpy())
    return np.concatenate(zs, axis=0)


def sample_latents(z_tr, n, seed):
    """Empirical-prior surrogate for CFGen's (non-vendored) flow stage: full-covariance
    Gaussian fit to the train latent codes, ridge-stabilised."""
    rng = np.random.default_rng(seed)
    mu_z = z_tr.mean(axis=0)
    cov = np.cov(z_tr.T)
    cov = cov + 1e-6 * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
    return rng.multivariate_normal(mu_z, cov, size=n).astype(np.float32)


def decode_latents(model, z, sf, dev, bs=8192):
    """mu = softmax(decoder(z)) * size_factor, batched; size_factor (n,1) raw counts."""
    mus = []
    with torch.no_grad():
        for i in range(0, z.shape[0], bs):
            zb = torch.from_numpy(z[i:i + bs]).float().to(dev)
            sfb = torch.from_numpy(sf[i:i + bs]).float().to(dev)
            mus.append(model.decode({"rna": zb}, {"rna": sfb})["rna"].cpu().numpy())
    return np.concatenate(mus, axis=0).astype(np.float64)


def run_one_seed(data, seed, real, lab_te, real_ms, types, rare_types):
    t0 = time.time()
    L.log(f"=== seed {seed} ===")
    run_e2e.SEED = seed  # evaluate() passes this to tstr()
    n_te = int(data["te"].sum())
    out = {"seed": seed, "n_synthetic_per_arm": n_te,
           "latent_model": "full-covariance Gaussian fit to train latent codes "
                           "(empirical prior; CFGen flow-matching stage not vendored)",
           "arms": {}}

    L.log(f"seed {seed} | fitting CFGen EncoderModel (phase2j matched loop) ...")
    model, dev, meta = fit_cfgen(data, seed)
    L.log(f"seed {seed} | fit done ({meta['n_epochs_run']} epochs, "
          f"best val NB-NLL {meta['best_val_nb_nll']:.1f}, "
          f"theta_g median {meta['theta_global_median']:.3f}, {time.time() - t0:.0f}s)")

    tr_idx = np.where(data["tr"])[0]
    z_tr = encode_cells(model, data["logX"][data["tr"]], dev)
    z_syn = sample_latents(z_tr, n_te, seed)
    rng = np.random.default_rng(seed + 500)
    pick = rng.integers(0, len(tr_idx), n_te)
    sf = data["raw"][tr_idx[pick]].sum(axis=1, keepdims=True).astype(np.float32)
    mu = np.clip(decode_latents(model, z_syn, sf, dev), 0.0, None)
    theta_g = torch.exp(model.theta).detach().cpu().numpy().astype(np.float64)
    theta = np.broadcast_to(theta_g, mu.shape).copy()

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(z_tr)
    lab_syn = data["labels"][data["tr"]][nn.kneighbors(z_syn, return_distance=False)[:, 0]]

    draws = {
        "e2e_cfgen_nb": run_e2e._log1p(sample_nb(mu, theta, seed=seed)),
        "e2e_cfgen_poisson": run_e2e._log1p(sample_nb_fixed_theta(mu, float("inf"), seed=seed)),
    }
    for name, X in draws.items():
        ev = run_e2e.evaluate(name, X, lab_syn, real, lab_te, real_ms, types, rare_types)
        ev["decoder"] = "cfgen_encoder_model_global_theta_g"
        ev["train_meta"] = meta
        ev["elapsed_s"] = round(time.time() - t0, 1)
        out["arms"][name] = ev
    out["elapsed_s"] = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()
    return out


def main() -> int:
    t0 = time.time()
    L.log("=== Phase 2w: CFGen end-to-end generation arm (3 seeds) ===")
    data = L.load_corpus()
    real = data["logX"][data["te"]]
    lab_te = data["labels"][data["te"]]
    u, c = np.unique(lab_te, return_counts=True)
    types = [t for t, n in zip(u, c) if n >= MIN_CELLS]
    rare_types = {t for t, n in zip(u, c) if n < RARE_MAX}
    real_ms = marker_scores(real, lab_te, types)
    L.log(f"{len(types)} scoreable types, {len(rare_types)} rare; "
          f"n_test={int(data['te'].sum())}")

    per_seed = []
    for seed in SEEDS:
        per_seed.append(run_one_seed(data, seed, real, lab_te, real_ms, types, rare_types))
        (RESULTS / f"e2e_cfgen_seed{seed}.json").write_text(
            json.dumps(per_seed[-1], indent=2))

    summary = {"purpose": "3-seed CFGen end-to-end generation arm (global theta_g NB "
                          "decoder; empirical-Gaussian latents; 1-NN label transfer)",
               "seeds": SEEDS,
               "n_synthetic_per_arm": per_seed[0]["n_synthetic_per_arm"],
               "n_types_real": len(types), "n_rare_types": len(rare_types),
               "arms": {}}
    for arm in ["e2e_cfgen_nb", "e2e_cfgen_poisson"]:
        recs = [s["arms"][arm] for s in per_seed if arm in s["arms"]]
        if not recs:
            continue
        metrics = ["marker_recall_mean", "marker_recall_median", "tstr_macro_f1",
                   "tstr_rare_f1", "tstr_rare_f1_unmasked"]
        agg = {}
        for m in metrics:
            vals = [r[m] for r in recs]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{m}_median"] = float(np.median(vals))
            agg[f"{m}_min"] = float(np.min(vals))
            agg[f"{m}_max"] = float(np.max(vals))
        agg["per_seed"] = {m: [r[m] for r in recs] for m in metrics}
        summary["arms"][arm] = agg

    summary["elapsed_total_s"] = round(time.time() - t0, 1)
    (RESULTS / "e2e_cfgen_multiseed.json").write_text(json.dumps(summary, indent=2))

    L.log(f"=== done in {time.time() - t0:.0f}s; summary -> "
          f"{RESULTS / 'e2e_cfgen_multiseed.json'} ===")
    for arm, a in summary["arms"].items():
        L.log(f"{arm}: marker_recall_mean={a['marker_recall_mean_mean']:.4f} ± "
              f"{a['marker_recall_mean_std']:.4f}; "
              f"TSTR={a['tstr_macro_f1_mean']:.4f} ± {a['tstr_macro_f1_std']:.4f}; "
              f"rare={a['tstr_rare_f1_mean']:.4f} ± {a['tstr_rare_f1_std']:.4f} "
              f"(unmasked {a['tstr_rare_f1_unmasked_mean']:.4f} ± "
              f"{a['tstr_rare_f1_unmasked_std']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
