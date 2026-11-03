"""F2 (encoder-side heterogeneity does not survive decoding) re-tested under the centroid-based estimator.

Recomputes both the latent and the decoded tightness with the centroid-based estimator and reports the
latent-versus-decoded rank correlation, in the Arm A geometry. The random projection has near-ideal latent
tightness under this estimator; if F2 holds, its decoded tightness is not correspondingly best.
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

from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402
from parent_metric_probe import parent_tightness_ratio_per_dataset  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
DEVICE = "cuda"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fm_cache(fm: str, ordered, cache_dirs):
    """Concatenate a cached FM's per-dataset embeddings in `ordered` order."""
    parts = []
    for name in ordered:
        p = None
        for d, pat in cache_dirs:
            cand = d / pat.format(fm=fm, ds=name)
            if cand.exists():
                p = cand
                break
        if p is None:
            return None
        parts.append(np.load(p)["X"])
    return np.vstack(parts).astype(np.float32)


def main() -> None:
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))
    tr, va, te = split == 0, split == 1, split == 2
    assert tr.sum() and va.sum() and te.sum()

    logX = np.load(CACHE / "logX_full.npy")
    raw_te, lab_te, ds_te = logX[te], labels[te], dataset_ids[te]
    target = logX  # Arm A: decode the same 2500 genes

    root = PROJECT_ROOT
    cache_dirs = [
        (root / "experiments/real_phase0_2026-07-05/results", "{fm}_{ds}.npz"),
        (root / "experiments/real_phase0_extended_2026-07-08/results", "{fm}_{ds}.npz"),
        (root / "data/embeddings/fm_emb", "{fm}_{fm}_{ds}.npz"),
    ]

    rng = np.random.default_rng(0)
    encoders: dict[str, np.ndarray] = {}

    encoders["scgpt_512"] = np.load(CACHE / "enc_A_scgpt.npy")

    log("PCA-512 / PCA-50 (fit on train)")
    encoders["pca_512"] = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)
    encoders["pca_50"] = PCA(n_components=50, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)

    log("random projections 512-d / 50-d")
    W512 = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    encoders["randproj_512"] = (logX @ W512).astype(np.float32)
    W50 = rng.normal(0, 1 / np.sqrt(50), size=(logX.shape[1], 50)).astype(np.float32)
    encoders["randproj_50"] = (logX @ W50).astype(np.float32)

    for fm in ["geneformer", "scfoundation", "uce"]:
        Z = _fm_cache(fm, ordered, cache_dirs)
        if Z is None or Z.shape[0] != logX.shape[0]:
            log(f"  {fm}: cached embeddings unavailable/mismatched -> EXCLUDED (not silently skipped)")
            continue
        encoders[fm] = Z
        log(f"  {fm}: loaded {Z.shape}")

    rows = []
    for name, Z in encoders.items():
        lat_med, lat_iqr, _ = parent_tightness_ratio_per_dataset(Z[te], raw_te, lab_te, ds_te)
        pred, vloss, hist, nparams = train_decoder(
            Z[tr], target[tr], Z[va], target[va], Z[te], device=DEVICE)
        rec = reconstruction_metrics(pred, target[te])
        dec_med, dec_iqr, _ = parent_tightness_ratio_per_dataset(pred, raw_te, lab_te, ds_te)
        rows.append({
            "encoder": name, "dim": int(Z.shape[1]),
            "parent_latent_tightness": lat_med, "parent_latent_iqr": list(lat_iqr),
            "parent_decoded_tightness": dec_med, "parent_decoded_iqr": list(dec_iqr),
            "recon_pearson": rec["mean_gene_pearson"], "val_mse": vloss,
        })
        log(f"  {name:16s} dim={Z.shape[1]:5d} latentT={lat_med:.4f} decodedT={dec_med:.4f} "
            f"pearson={rec['mean_gene_pearson']:.4f}")

    lat = np.array([r["parent_latent_tightness"] for r in rows])
    dec = np.array([r["parent_decoded_tightness"] for r in rows])
    rec = np.array([r["recon_pearson"] for r in rows])

    real = [i for i, r in enumerate(rows) if not r["encoder"].startswith("randproj")]

    summary = {
        "all_models": {
            "n": len(rows),
            "latent_vs_decoded_pearson": float(pearsonr(lat, dec)[0]),
            "latent_vs_decoded_spearman": float(spearmanr(lat, dec)[0]),
            "latent_vs_decoded_p": float(pearsonr(lat, dec)[1]),
            "latent_range": [float(lat.min()), float(lat.max())],
            "decoded_range": [float(dec.min()), float(dec.max())],
        },
        "real_encoders_only": {
            "n": len(real),
            "latent_vs_decoded_pearson": float(pearsonr(lat[real], dec[real])[0]),
            "latent_vs_decoded_spearman": float(spearmanr(lat[real], dec[real])[0]),
            "latent_vs_decoded_p": float(pearsonr(lat[real], dec[real])[1]),
            "latent_range": [float(lat[real].min()), float(lat[real].max())],
            "decoded_range": [float(dec[real].min()), float(dec[real].max())],
            "latent_spread": float(lat[real].max() - lat[real].min()),
            "decoded_spread": float(dec[real].max() - dec[real].min()),
        },
        "randproj_check": {
            "note": "randproj has near-ideal latent geometry under the parent metric; "
                    "does that buy decoded quality?",
            "rows": [r for r in rows if r["encoder"].startswith("randproj")],
        },
    }
    log(f"ALL   n={summary['all_models']['n']} latent-vs-decoded "
        f"r={summary['all_models']['latent_vs_decoded_pearson']:.3f} "
        f"rho={summary['all_models']['latent_vs_decoded_spearman']:.3f}")
    log(f"REAL  n={summary['real_encoders_only']['n']} latent-vs-decoded "
        f"r={summary['real_encoders_only']['latent_vs_decoded_pearson']:.3f} "
        f"rho={summary['real_encoders_only']['latent_vs_decoded_spearman']:.3f}")
    log(f"REAL  latent spread={summary['real_encoders_only']['latent_spread']:.4f} "
        f"decoded spread={summary['real_encoders_only']['decoded_spread']:.4f}")

    out = {
        "purpose": "re-test F2 under the exact centroid-based estimator, Arm A geometry",
        "estimator": "parent: within=mean per-cell L2 from centroid; between=mean inter-centroid L2; ratio=enc/raw",
        "n_test_cells": int(te.sum()),
        "results": rows,
        "summary": summary,
    }
    (RESULTS / "f2_under_parent_metric.json").write_text(json.dumps(out, indent=2))
    log(f"=== done -> {RESULTS / 'f2_under_parent_metric.json'} ===")


if __name__ == "__main__":
    main()
