"""Oracle / ceiling control for Arm B, added after the main Phase 2a run.

The main run found EVERY encoder -- PCA(50), scGPT(512), Geneformer(768), and
even a random 50-d Gaussian projection -- decodes G_out at essentially the
same reconstruction quality (test mean-gene Pearson 0.644-0.676) and lands in
a narrow decoded-tightness band (0.754-0.809) despite latent tightness
spanning 0.789-0.971. Two very different stories fit that:

  (H1) "The decoder is a co-bottleneck": the MLP decoder discards whatever
       structure the encoder preserved, so encoder choice barely matters.
  (H2) "The task is saturated": predicting G_out from G_in is intrinsically
       hard (different genes, weak cross-gene predictability), so ~0.67 is
       the CEILING and every encoder already reaches it. Nothing is being
       destroyed -- there was never more signal to pass along.

These have opposite implications for HetCLOP. H1 says fix the decoder. H2 says
the experiment can't distinguish encoders because the readout task is
uninformative, and says nothing about the decoder at all.

The discriminating control is an ORACLE encoder: the identity map on all 1250
G_in genes (no compression, no encoder). It has strictly more information than
any encoder here -- every encoder is a function of exactly these genes.
  * If oracle Pearson ~= 0.67 too -> H2. All encoders sit at the task ceiling;
    the 50-d bottleneck costs nothing because there is nothing more to extract.
  * If oracle Pearson >> 0.67 -> H1. Real information exists in G_in that the
    compressed encoders (and/or the decoder) fail to carry into G_out.

Also fits a PCA-1250 (full-rank-ish, n_comp=1250) variant to separate "linear
compression to 50-d" from "compression at all", and reports the decoder's
val-loss history so an undertrained-decoder explanation can be ruled out.

Reuses the EXACT cached cell split, gene split, targets, and decoder/eval code
from run_phase2a.py -- no re-embedding, no GPU encoder work.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.decomposition import PCA

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
    # split is int8-coded 0=train / 1=val / 2=test (see run_phase2a.py:289) --
    # NOT strings. An earlier version of this script used .astype(str) and
    # silently produced all-empty masks; assert non-empty rather than trust it.
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))

    logX_full = np.load(CACHE / "logX_full.npy")
    tr, va, te = split == 0, split == 1, split == 2
    log(f"split: train={tr.sum()} val={va.sum()} test={te.sum()}")
    assert tr.sum() and va.sum() and te.sum(), "empty split mask -- refusing to run"

    labels_te, ds_te = labels[te], dataset_ids[te]
    target_gout = logX_full[:, gout_idx]

    corpus = assemble_real_corpus(ALL_DATASETS, CORPUS_SOURCE_ROOT)
    assert list(corpus["adata_by_dataset"].keys()) == ordered
    assert np.array_equal(corpus["dataset_ids"].astype(str), dataset_ids)
    raw_by_ds = corpus["raw_X_by_dataset"]
    raw_by_ds_test = {}
    for d in ordered:
        dmask_te = te & (dataset_ids == d)
        raw_by_ds_test[d] = raw_by_ds[d][within_pos[dmask_te]]

    def _tight(Z):
        try:
            med, iqr, vals = per_dataset_tightness_ratio(Z, raw_by_ds_test, labels_te, ds_te)
            return {"median": med, "iqr": list(iqr), "n_datasets": len(vals)}
        except Exception as e:
            return {"error": str(e)[:150]}

    def eval_encoder(Z, name):
        pred, vloss, hist, nparams = train_decoder(
            Z[tr], target_gout[tr], Z[va], target_gout[va], Z[te], device=DEVICE)
        rec = reconstruction_metrics(pred, target_gout[te])
        row = {
            "encoder": name, "embed_dim": int(Z.shape[1]), "decoder_params": nparams,
            "val_mse": vloss,
            "reconstruction": rec,
            "latent_tightness": _tight(Z[te]),
            "decoded_tightness": _tight(pred),
            # hist entries are {"epoch","train","val"} dicts; keep val curve so an
            # undertrained-decoder explanation can be checked (did val plateau?).
            "val_curve_first5": [float(h["val"]) for h in hist[:5]] if hist else [],
            "val_curve_last5": [float(h["val"]) for h in hist[-5:]] if hist else [],
        }
        log(f"  {name}: dim={Z.shape[1]} pearson={rec['mean_gene_pearson']:.4f} "
            f"mse={rec['mse']:.4f} latentT={row['latent_tightness'].get('median','NA')} "
            f"decodedT={row['decoded_tightness'].get('median','NA')}")
        return row

    rows = []

    # ORACLE: identity on all 1250 G_in genes. Strictly more information than
    # any encoder in the main run -- every one of them is a function of these.
    log("=== ORACLE control: identity on all 1250 G_in genes (no compression) ===")
    Z_oracle = logX_full[:, gin_idx].astype(np.float32)
    rows.append(eval_encoder(Z_oracle, "oracle_identity_gin_1250d"))

    # PCA at full rank-ish: separates "compression to 50-d" from "linear compression".
    log("=== PCA-1250 (near-full-rank linear) on G_in ===")
    p = PCA(n_components=min(1250, int(tr.sum()) - 1), random_state=0)
    p.fit(Z_oracle[tr])
    Z_pca1250 = p.transform(Z_oracle).astype(np.float32)
    log(f"  PCA-1250 explained variance: {p.explained_variance_ratio_.sum():.4f}")
    rows.append(eval_encoder(Z_pca1250, "pca_1250d"))

    # PCA-50 re-fit here as an in-script anchor (should reproduce the main run's
    # Arm B pca row; a mismatch would mean the cached split/target drifted).
    log("=== PCA-50 anchor (should reproduce main run Arm B pca) ===")
    p50 = PCA(n_components=50, random_state=0)
    p50.fit(Z_oracle[tr])
    Z_pca50 = p50.transform(Z_oracle).astype(np.float32)
    log(f"  PCA-50 explained variance: {p50.explained_variance_ratio_.sum():.4f}")
    rows.append(eval_encoder(Z_pca50, "pca_50d_anchor"))

    out = {
        "purpose": "discriminate H1 (decoder co-bottleneck) vs H2 (task saturated) for Arm B",
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "g_in": len(gin_idx), "g_out": len(gout_idx),
        "results": rows,
    }
    (RESULTS / "oracle_control.json").write_text(json.dumps(out, indent=2))
    log(f"=== done -> {RESULTS / 'oracle_control.json'} ===")


if __name__ == "__main__":
    main()
