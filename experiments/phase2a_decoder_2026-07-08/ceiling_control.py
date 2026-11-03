"""Is the ~0.67 Pearson ceiling intrinsic to the G_in->G_out task, or is it the
MLP decoder's own capacity ceiling?

The oracle control showed an uncompressed 1250-d input does no better than a
50-d PCA. But that is equally consistent with:
  (H2a) the task is saturated -- gene-gene predictability tops out near 0.67;
  (H2b) the MLP decoder caps EVERY input at ~0.67, oracle included.
Both predict oracle ~= encoders, so the oracle alone cannot separate them.

Two probes that do:
  1. RIDGE (closed-form linear least squares, G_in -> G_out). A completely
     different model class: no SGD, no early stopping, no capacity ceiling from
     depth/width. Swept over alpha. If ridge also lands ~0.67, the ceiling is
     not an artifact of the MLP's optimization or architecture.
  2. BIG MLP (wider + deeper + longer) on the oracle input. If more capacity
     buys nothing, the MLP is not the binding constraint either.

If BOTH cap near 0.67 -> H2a: the readout task itself is saturated, so Arm B's
reconstruction numbers cannot rank encoders (a limitation of the experiment,
not a property of the pipeline).

NOTE: this probe is about RECONSTRUCTION only. The decoded-TIGHTNESS collapse
(latent 0.97 -> decoded 0.76, even for the oracle) is a separate phenomenon and
is NOT explained away by a reconstruction ceiling -- a deterministic decoder
predicting E[Y|Z] necessarily shrinks within-type variance regardless of how
predictable Y is. Both are reported.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.data.real_corpus import assemble_real_corpus
from src.diagnostics.tightness import per_dataset_tightness_ratio
from src.utils.paths import CORPUS_SOURCE_ROOT

from combine_datasets import ALL_DATASETS  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

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
    dataset_ids = meta["dataset_ids"].astype(str)
    labels = meta["labels"].astype(str)
    within_pos = meta["within_pos"]
    split = meta["split"]  # int8: 0=train 1=val 2=test
    ordered = list(meta["ordered"].astype(str))

    logX_full = np.load(CACHE / "logX_full.npy")
    tr, va, te = split == 0, split == 1, split == 2
    assert tr.sum() and va.sum() and te.sum(), "empty split mask -- refusing to run"
    log(f"split: train={tr.sum()} val={va.sum()} test={te.sum()}")

    X = logX_full[:, gin_idx].astype(np.float64)
    Y = logX_full[:, gout_idx].astype(np.float64)
    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va], Y[va]
    Xte, Yte = X[te], Y[te]

    rows = []

    # ---- 1. Ridge, closed form, alpha swept on val ------------------------
    log("=== RIDGE (closed-form linear, G_in -> G_out) ===")
    mu_x, mu_y = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mu_x, Ytr - mu_y
    G = Xc.T @ Xc                      # (1250, 1250)
    B = Xc.T @ Yc                      # (1250, 1250)
    eye = np.eye(G.shape[0])
    best = None
    for alpha in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
        W = np.linalg.solve(G + alpha * eye, B)
        vp = (Xva - mu_x) @ W + mu_y
        vmse = float(np.mean((vp - Yva) ** 2))
        log(f"  alpha={alpha:<8g} val_mse={vmse:.6f}")
        if best is None or vmse < best[1]:
            best = (alpha, vmse, W)
    alpha, vmse, W = best
    ridge_pred = ((Xte - mu_x) @ W + mu_y).astype(np.float32)
    rec = reconstruction_metrics(ridge_pred, Yte.astype(np.float32))
    log(f"  RIDGE best alpha={alpha} val_mse={vmse:.6f} "
        f"test pearson={rec['mean_gene_pearson']:.4f} mse={rec['mse']:.4f}")
    rows.append({"model": "ridge_linear_gin_to_gout", "best_alpha": alpha,
                 "val_mse": vmse, "reconstruction": rec})

    # ---- 2. Big MLP on the oracle (uncompressed) input --------------------
    log("=== BIG MLP on oracle input (wider/deeper/longer) ===")
    Z = logX_full[:, gin_idx].astype(np.float32)
    Yf = logX_full[:, gout_idx].astype(np.float32)
    pred, vloss, hist, nparams = train_decoder(
        Z[tr], Yf[tr], Z[va], Yf[va], Z[te],
        hidden_dims=(2048, 4096, 2048), epochs=120, device=DEVICE)
    rec_big = reconstruction_metrics(pred, Yf[te])
    log(f"  BIG MLP params={nparams:,} val_mse={vloss:.6f} "
        f"test pearson={rec_big['mean_gene_pearson']:.4f} mse={rec_big['mse']:.4f}")

    # Decoded tightness of the big MLP -- does more capacity rescue heterogeneity?
    corpus = assemble_real_corpus(ALL_DATASETS, CORPUS_SOURCE_ROOT)
    assert list(corpus["adata_by_dataset"].keys()) == ordered
    assert np.array_equal(corpus["dataset_ids"].astype(str), dataset_ids)
    raw_by_ds = corpus["raw_X_by_dataset"]
    raw_by_ds_test = {}
    for d in ordered:
        dmask_te = te & (dataset_ids == d)
        raw_by_ds_test[d] = raw_by_ds[d][within_pos[dmask_te]]
    labels_te, ds_te = labels[te], dataset_ids[te]

    def _tight(M):
        try:
            med, iqr, vals = per_dataset_tightness_ratio(M, raw_by_ds_test, labels_te, ds_te)
            return {"median": med, "iqr": list(iqr), "n_datasets": len(vals)}
        except Exception as e:
            return {"error": str(e)[:150]}

    big_dec_t = _tight(pred)
    ridge_dec_t = _tight(ridge_pred)
    log(f"  BIG MLP decoded tightness = {big_dec_t.get('median','NA')}")
    log(f"  RIDGE   decoded tightness = {ridge_dec_t.get('median','NA')}")

    rows.append({"model": "big_mlp_oracle_input", "decoder_params": nparams,
                 "val_mse": vloss, "reconstruction": rec_big,
                 "decoded_tightness": big_dec_t,
                 "val_curve_last5": [float(h["val"]) for h in hist[-5:]] if hist else []})
    rows[0]["decoded_tightness"] = ridge_dec_t

    out = {
        "purpose": "separate task-saturation (H2a) from MLP-capacity ceiling (H2b); "
                   "and check whether decoder capacity rescues decoded heterogeneity",
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "results": rows,
    }
    (RESULTS / "ceiling_control.json").write_text(json.dumps(out, indent=2))
    log(f"=== done -> {RESULTS / 'ceiling_control.json'} ===")


if __name__ == "__main__":
    main()
