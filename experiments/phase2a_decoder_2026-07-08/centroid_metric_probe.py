"""Tightness ratio under the centroid-based estimator on the analysed corpus.

within = mean over cell types of the mean distance of cells to their type centroid; between = mean distance
between type centroids; tightness = within / between; ratio = tightness(encoded) / tightness(raw). The optimum
(a ratio of one) is reached by any near-isometry, including a random projection that learns nothing. The
script scores raw, PCA-512, random projection-512 and scGPT-512 at matched dimension, adds 50-dimensional
variants to show the estimator's sensitivity to dimension, and re-tests F2 under this estimator.

CPU-only; reuses the cached Arm A embeddings and panel.
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

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# the original centroid-based estimator
# ---------------------------------------------------------------------------
def centroid_within_between(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, int]:
    types = np.unique(labels)
    intra_vals, centroids = [], []
    for t in types:
        M = X[labels == t]
        if M.shape[0] < 2:
            continue
        mu = M.mean(axis=0)
        centroids.append(mu)
        intra_vals.append(float(np.linalg.norm(M - mu, axis=1).mean()))
    if len(centroids) < 2:
        return (float(np.mean(intra_vals)) if intra_vals else 0.0, 0.0, len(centroids))
    C = np.stack(centroids)
    inter = [float(np.linalg.norm(C[i] - C[j]))
             for i in range(len(C)) for j in range(i + 1, len(C))]
    return float(np.mean(intra_vals)), float(np.mean(inter)) if inter else 0.0, len(C)


def centroid_tightness(X: np.ndarray, labels: np.ndarray) -> float:
    intra, inter, _ = centroid_within_between(X, labels)
    return intra / inter if inter > 0 else float("nan")


def centroid_tightness_ratio_per_dataset(enc_X, raw_X, labels, dataset_ids):
    """Per-dataset statistic, reduced to a median across datasets
    (manuscript:710 reports median/IQR over 8 datasets; we have 50)."""
    vals = []
    for d in np.unique(dataset_ids):
        m = dataset_ids == d
        if len(np.unique(labels[m])) < 2:
            continue
        t_raw = centroid_tightness(raw_X[m], labels[m])
        t_enc = centroid_tightness(enc_X[m], labels[m])
        if not np.isfinite(t_raw) or not np.isfinite(t_enc) or t_raw == 0:
            continue
        vals.append(t_enc / t_raw)
    v = np.array(vals)
    return float(np.median(v)), (float(np.percentile(v, 25)), float(np.percentile(v, 75))), len(v)


def main() -> None:
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    te = split == 2
    assert te.sum(), "empty test mask"

    logX = np.load(CACHE / "logX_full.npy")           # (N, 2500) raw space
    scgpt = np.load(CACHE / "enc_A_scgpt.npy")        # (N, 512) native scGPT
    log(f"raw={logX.shape} scgpt={scgpt.shape}; scoring on test cells (n={te.sum()})")

    raw_te, lab_te, ds_te = logX[te], labels[te], dataset_ids[te]
    tr = split == 0

    rng = np.random.default_rng(0)
    encoders: dict[str, np.ndarray] = {}

    # --- Matched 512-d, ---
    log("fitting PCA-512 (fitted globally on train, not per dataset)")
    p512 = PCA(n_components=512, random_state=0).fit(logX[tr])
    encoders["pca_512"] = p512.transform(logX).astype(np.float32)

    log("random Gaussian projection -> 512-d (the metric's predicted optimum)")
    W512 = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    encoders["randproj_512"] = (logX @ W512).astype(np.float32)

    encoders["scgpt_512_native"] = scgpt

    # --- 50-d variants: isolates the estimator's dimension sensitivity.
    # Reported separately from the matched-dimension comparison.
    log("50-d variants (dimension-sensitivity of the estimator, reported separately)")
    p50 = PCA(n_components=50, random_state=0).fit(logX[tr])
    encoders["pca_50"] = p50.transform(logX).astype(np.float32)
    W50 = rng.normal(0, 1 / np.sqrt(50), size=(logX.shape[1], 50)).astype(np.float32)
    encoders["randproj_50"] = (logX @ W50).astype(np.float32)

    # identity = the trivial "no distortion at all" reference; ratio must be 1.0
    encoders["identity_raw_2500"] = logX

    rows = []
    for name, Z in encoders.items():
        med, iqr, n = centroid_tightness_ratio_per_dataset(Z[te], raw_te, lab_te, ds_te)
        rows.append({"encoder": name, "dim": int(Z.shape[1]),
                     "centroid_tightness_ratio_median": med, "iqr": list(iqr), "n_datasets": n})
        log(f"  {name:22s} dim={Z.shape[1]:5d}  centroid tightness ratio = {med:.4f}  "
            f"IQR=({iqr[0]:.4f},{iqr[1]:.4f})")

    # --- Noise-dimension inflation of the centroid estimator, in isolation ----------
    log("noise-dimension sensitivity of the centroid estimator (matched-dimension comparisons are immune;"
        " this concerns cross-dimension comparisons such as encoder-vs-raw normalization)")
    base = encoders["pca_50"][te]
    infl = []
    for extra in [0, 50, 200, 800, 2000]:
        noise = rng.normal(0, 1.0, size=(base.shape[0], extra)).astype(np.float32) if extra else None
        Z = base if extra == 0 else np.hstack([base, noise])
        t = centroid_tightness(Z, lab_te)
        infl.append({"extra_noise_dims": extra, "centroid_tightness": float(t)})
        log(f"  +{extra:5d} pure-noise dims -> centroid tightness = {t:.4f}")

    # --- F2 under the centroid estimator: does latent rank-order predict decoded? ---
    # Reuse Arm A decoded predictions if they were saved; otherwise skip loudly.
    dec_path = CACHE / "armA_decoded.npz"
    f2 = {"status": "skipped: no cached Arm A decoded predictions "
                    "(run_phase2a.py did not persist them); F2 under the centroid "
                    "estimator not re-tested here"}
    if dec_path.exists():
        dec = np.load(dec_path)
        lat, decd, names = [], [], []
        for name in dec.files:
            if name not in encoders:
                continue
            lm, _, _ = centroid_tightness_ratio_per_dataset(encoders[name][te], raw_te, lab_te, ds_te)
            dm, _, _ = centroid_tightness_ratio_per_dataset(dec[name], raw_te, lab_te, ds_te)
            lat.append(lm); decd.append(dm); names.append(name)
        if len(lat) >= 3:
            f2 = {"n": len(lat), "encoders": names,
                  "pearson_r": float(pearsonr(lat, decd)[0]),
                  "spearman_rho": float(spearmanr(lat, decd)[0])}
    log(f"F2-under-centroid-estimator: {f2}")

    out = {
        "purpose": "score encoders under the centroid-based tightness estimator",
        "centroid_estimator": "within = mean per-cell L2 from centroid; "
                            "between = mean inter-centroid L2; ratio = enc/raw",
        "n_test_cells": int(te.sum()),
        "results": rows,
        "noise_dim_inflation": infl,
        "f2_under_centroid_estimator": f2,
    }
    (RESULTS / "centroid_metric_probe.json").write_text(json.dumps(out, indent=2))
    log(f"=== done -> {RESULTS / 'centroid_metric_probe.json'} ===")


if __name__ == "__main__":
    main()
