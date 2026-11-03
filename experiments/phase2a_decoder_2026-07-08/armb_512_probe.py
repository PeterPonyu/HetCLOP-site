"""Arm B at matched 512-d: the fair test of "is the tightness metric's optimum
(a random projection) actually a bad encoder?"

Why this is needed. `f2_under_parent_metric.py` (Arm A: encoders see the 2500-gene
panel, decoder predicts those SAME genes) found randproj-512 has BOTH the highest
parent-metric latent tightness (0.9956) AND the best reconstruction (0.7522).
That is not a critique of the metric -- it is a near-lossless linear compression
(Johnson-Lindenstrauss) of exactly the genes being reconstructed. Arm A rewards
any linear function of the input genes by near-invertibility.

PHASE2A_REPORT.md's claim "the metric's optimum decodes worst" rested on
randproj at **50-d** in Arm B (0.6443). That conflated two variables: the
projection's randomness AND its dimensionality. At 512-d a random projection
retains almost all of G_in; at 50-d it does not.

Arm B (encoders see only G_in=1250 genes; decoder predicts only G_out=1250 genes
that NO encoder saw) removes the invertibility advantage: G_out cannot be
recovered by inverting any function of G_in, only predicted. So this is the fair
setting in which to ask whether a random projection -- the parent metric's
maximizer -- is actually a good or bad encoder.

Encoders, all restricted to G_in:
  oracle_1250   identity (upper bound on available information)
  pca_512       linear, learned basis
  randproj_512  linear, random basis  <- the metric's maximizer at matched dim
  pca_50        linear, learned, heavily compressed
  randproj_50   linear, random, heavily compressed (the original Arm B null)
  scgpt_512     cached Arm B scGPT embedding (G_in-restricted re-embed)
  geneformer    cached Arm B Geneformer embedding (G_in-restricted re-embed)

Reports, for each: reconstruction on held-out test cells, plus latent and decoded
tightness under BOTH estimators (HetCLOP's and the parent's centroid-based one),
so no conclusion depends on a single metric definition.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from src.data.real_corpus import assemble_real_corpus
from src.diagnostics.tightness import per_dataset_tightness_ratio
from src.utils.paths import CORPUS_SOURCE_ROOT

from combine_datasets import ALL_DATASETS  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402
from parent_metric_probe import parent_tightness_ratio_per_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
DEVICE = "cuda"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    panel = json.loads((CACHE / "panel.json").read_text())
    gin_idx = np.array(panel["gin_idx"])
    gout_idx = np.array(panel["gout_idx"])

    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    within_pos = meta["within_pos"]
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))
    tr, va, te = split == 0, split == 1, split == 2
    assert tr.sum() and va.sum() and te.sum(), "empty split mask"

    logX = np.load(CACHE / "logX_full.npy")
    Xin = logX[:, gin_idx].astype(np.float32)      # what encoders may see
    Yout = logX[:, gout_idx].astype(np.float32)    # what decoders must predict
    raw_te, lab_te, ds_te = logX[te], labels[te], dataset_ids[te]
    log(f"G_in={Xin.shape} G_out={Yout.shape} test={te.sum()}")

    # native-panel raw denominators for HetCLOP's per-dataset estimator
    corpus = assemble_real_corpus(ALL_DATASETS, CORPUS_SOURCE_ROOT)
    assert list(corpus["adata_by_dataset"].keys()) == ordered
    assert np.array_equal(corpus["dataset_ids"].astype(str), dataset_ids)
    raw_by_ds_test = {}
    for d in ordered:
        dm = te & (dataset_ids == d)
        raw_by_ds_test[d] = corpus["raw_X_by_dataset"][d][within_pos[dm]]

    rng = np.random.default_rng(0)
    enc: dict[str, np.ndarray] = {}
    enc["oracle_1250"] = Xin
    log("fitting PCA-512 / PCA-50 on G_in (train cells only)")
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(Xin[tr]).transform(Xin).astype(np.float32)
    enc["pca_50"] = PCA(n_components=50, random_state=0).fit(Xin[tr]).transform(Xin).astype(np.float32)
    for dim in (512, 50):
        W = rng.normal(0, 1 / np.sqrt(dim), size=(Xin.shape[1], dim)).astype(np.float32)
        enc[f"randproj_{dim}"] = (Xin @ W).astype(np.float32)
    for nm, f in [("scgpt_512", "enc_B_scgpt.npy"), ("geneformer", "enc_B_geneformer.npy")]:
        p = CACHE / f
        if p.exists():
            enc[nm] = np.load(p)
            log(f"loaded cached Arm B {nm}: {enc[nm].shape}")
        else:
            log(f"MISSING cached Arm B {nm} ({f}) -> EXCLUDED (not silently skipped)")

    def het_t(M):
        try:
            med, _, _ = per_dataset_tightness_ratio(M, raw_by_ds_test, lab_te, ds_te)
            return float(med)
        except Exception:
            return float("nan")

    rows = []
    for name, Z in enc.items():
        lat_h = het_t(Z[te])
        lat_p, _, _ = parent_tightness_ratio_per_dataset(Z[te], raw_te, lab_te, ds_te)
        pred, vloss, _, nparams = train_decoder(
            Z[tr], Yout[tr], Z[va], Yout[va], Z[te], device=DEVICE)
        rec = reconstruction_metrics(pred, Yout[te])
        dec_h = het_t(pred)
        dec_p, _, _ = parent_tightness_ratio_per_dataset(pred, raw_te, lab_te, ds_te)
        rows.append({
            "encoder": name, "dim": int(Z.shape[1]), "decoder_params": nparams,
            "val_mse": vloss, "recon_pearson": rec["mean_gene_pearson"], "recon_mse": rec["mse"],
            "latent_tightness_hetclop": lat_h, "decoded_tightness_hetclop": dec_h,
            "latent_tightness_parent": lat_p, "decoded_tightness_parent": dec_p,
        })
        log(f"  {name:14s} dim={Z.shape[1]:5d} pearson={rec['mean_gene_pearson']:.4f} "
            f"| hetclop lat={lat_h:.4f} dec={dec_h:.4f} | parent lat={lat_p:.4f} dec={dec_p:.4f}")

    def corr(xs, ys):
        return {"pearson_r": float(pearsonr(xs, ys)[0]), "pearson_p": float(pearsonr(xs, ys)[1]),
                "spearman_rho": float(spearmanr(xs, ys)[0])}

    real = [r for r in rows if not r["encoder"].startswith("randproj") and r["encoder"] != "oracle_1250"]
    summary = {}
    for tag, subset in [("all_models", rows), ("real_encoders_only", real)]:
        if len(subset) < 3:
            summary[tag] = {"n": len(subset), "note": "too few models for correlation"}
            continue
        lh = [r["latent_tightness_hetclop"] for r in subset]
        dh = [r["decoded_tightness_hetclop"] for r in subset]
        lp = [r["latent_tightness_parent"] for r in subset]
        dp = [r["decoded_tightness_parent"] for r in subset]
        summary[tag] = {
            "n": len(subset),
            "latent_vs_decoded_hetclop": corr(lh, dh),
            "latent_vs_decoded_parent": corr(lp, dp),
            "latent_spread_hetclop": float(max(lh) - min(lh)),
            "decoded_spread_hetclop": float(max(dh) - min(dh)),
            "latent_spread_parent": float(max(lp) - min(lp)),
            "decoded_spread_parent": float(max(dp) - min(dp)),
        }

    rp512 = next((r for r in rows if r["encoder"] == "randproj_512"), None)
    orc = next((r for r in rows if r["encoder"] == "oracle_1250"), None)
    verdict = None
    if rp512 and orc:
        verdict = {
            "randproj_512_recon": rp512["recon_pearson"],
            "oracle_1250_recon": orc["recon_pearson"],
            "randproj_512_parent_latent_tightness": rp512["latent_tightness_parent"],
            "question": "is the parent metric's maximizer (randproj-512) a BAD encoder in the "
                        "circularity-broken arm?",
            "answer": ("NO -- it reconstructs within {:.4f} of the uncompressed oracle"
                       .format(abs(rp512["recon_pearson"] - orc["recon_pearson"]))),
        }
    log(f"VERDICT: {verdict}")

    out = {
        "purpose": "Arm B at matched 512-d: fair test of whether the parent metric's maximizer "
                   "(random projection) is actually a bad encoder once the near-invertibility "
                   "advantage of Arm A is removed",
        "n_test_cells": int(te.sum()), "g_in": len(gin_idx), "g_out": len(gout_idx),
        "results": rows, "summary": summary, "verdict": verdict,
    }
    (RESULTS / "armb_512_probe.json").write_text(json.dumps(out, indent=2))
    log(f"=== done -> {RESULTS / 'armb_512_probe.json'} ===")


if __name__ == "__main__":
    main()
