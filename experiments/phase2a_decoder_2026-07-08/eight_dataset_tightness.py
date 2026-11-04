"""Tightness ratio in an eight-dataset setting, with a random-projection baseline.

Uses eight benchmark datasets (3000-cell samples, counts per 10,000 with log1p, 2000 highly variable genes),
cell-type labels with unknown cells removed and at least 20 cells per type, PCA fitted per dataset at 512
dimensions and the centroid-based estimator (mean distance of cells to their type centroid over the mean
distance between centroids, encoded over raw). A random 512-dimensional Gaussian projection is added as a
baseline and run over N_SEEDS seeds; the median over datasets is reported per seed with its across-seed spread.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

from centroid_metric_probe import centroid_tightness  # centroid-based estimator
from src.utils.paths import CORPUS_SOURCE_ROOT  # noqa: E402

CORPUS_SOURCE = CORPUS_SOURCE_ROOT
H5AD_DIR = CORPUS_SOURCE / "data" / "processed_h5ad_cap3k_seed0"
SUBCLUSTER_META = CORPUS_SOURCE / "data" / "processed_h5ad" / "subcluster_metadata.json"
RESULTS = Path(__file__).resolve().parent / "results"

SELECTED_DATASETS = [
    "GSE167597_spineMm",
    "GSE262288_breastMetasisHmCancer",
    "GSE145929_ProgastinMmDev",
    "GSE226131_HSCMmAged",
    "GSE222002_TcellsHmCancer",
    "GSE155109_bcECHmCancer",
    "dentate",
    "setty",
]
EMBED_DIM = 512   # all encoders map to 512 dimensions
N_SEEDS = 10
MIN_LABELED = 20  # skip if known_mask.sum() < 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_benchmark_dataset(name: str, meta: dict):
    """Raw matrix (.X = CP10K+log1p, 2000 HVG) and per-cell cell_type."""
    path = H5AD_DIR / f"{name}_processed.h5ad"
    if not path.exists():
        return None, None, f"missing h5ad: {path.name}"
    adata = ad.read_h5ad(path)
    if name not in meta:
        return None, None, "absent from subcluster_metadata.json"
    clusters = meta[name]["clusters"]
    ct = np.full(adata.n_obs, "Unknown", dtype=object)
    for info in clusters.values():
        idx = np.asarray(info["cell_indices"])
        idx = idx[idx < adata.n_obs]          # guard: cap3k subsample may be smaller
        ct[idx] = info["cell_type"]
    known = ct != "Unknown"
    if known.sum() < MIN_LABELED:
        return None, None, f"too few labeled cells ({known.sum()})"
    if len(np.unique(ct[known])) < 2:
        return None, None, "fewer than 2 cell types"
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return X[known].astype(np.float32), ct[known].astype(str), None


def main() -> None:
    meta = json.loads(SUBCLUSTER_META.read_text())
    rng_master = np.random.default_rng(0)

    per_ds = {}
    skipped = []
    for name in SELECTED_DATASETS:
        X, labels, err = load_benchmark_dataset(name, meta)
        if err:
            log(f"  SKIP {name}: {err}")
            skipped.append({"dataset": name, "reason": err})
            continue
        n_comp = min(EMBED_DIM, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=0)
        Z_pca = pca.fit_transform(X)
        if n_comp < EMBED_DIM:
            Z_pca = np.pad(Z_pca, ((0, 0), (0, EMBED_DIM - n_comp)))

        t_raw = centroid_tightness(X, labels)
        t_pca = centroid_tightness(Z_pca, labels)

        rp_ratios = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(int(rng_master.integers(1 << 30)) + s)
            W = rng.normal(0, 1 / np.sqrt(EMBED_DIM),
                           size=(X.shape[1], EMBED_DIM)).astype(np.float32)
            t_rp = centroid_tightness(X @ W, labels)
            rp_ratios.append(t_rp / t_raw)

        per_ds[name] = {
            "n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]),
            "n_types": int(len(np.unique(labels))),
            "pca_ratio": float(t_pca / t_raw),
            "randproj_ratio_per_seed": [float(v) for v in rp_ratios],
            "randproj_ratio_mean": float(np.mean(rp_ratios)),
            "randproj_ratio_std": float(np.std(rp_ratios)),
        }
        log(f"  {name:32s} n={X.shape[0]:5d} types={len(np.unique(labels)):3d}  "
            f"PCA={t_pca/t_raw:.4f}  randproj={np.mean(rp_ratios):.4f}±{np.std(rp_ratios):.4f}")

    if not per_ds:
        raise SystemExit("no datasets loaded")

    pca_vals = np.array([v["pca_ratio"] for v in per_ds.values()])
    # Per seed: median across datasets, then spread across seeds.
    rp_medians = []
    for s in range(N_SEEDS):
        vals = np.array([v["randproj_ratio_per_seed"][s] for v in per_ds.values()])
        rp_medians.append(float(np.median(vals)))
    rp_medians = np.array(rp_medians)

    summary = {
        "n_datasets_used": len(per_ds),
        "skipped": skipped,
        "embed_dim": EMBED_DIM,
        "n_seeds": N_SEEDS,
        "pca_median_tightness_ratio": float(np.median(pca_vals)),
        "pca_iqr": [float(np.percentile(pca_vals, 25)), float(np.percentile(pca_vals, 75))],
        "randproj_median_tightness_ratio_mean_over_seeds": float(rp_medians.mean()),
        "randproj_median_tightness_ratio_std_over_seeds": float(rp_medians.std()),
        "randproj_median_min_over_seeds": float(rp_medians.min()),
        "randproj_median_max_over_seeds": float(rp_medians.max()),
    }
    log("")
    log(f"Tightness over {len(per_ds)}/8 benchmark datasets, {N_SEEDS} seeds:")
    log(f"  PCA-512 (per-dataset fit)  median tightness ratio = {summary['pca_median_tightness_ratio']:.4f} ")
    log(f"  RANDOM PROJECTION 512-d    median tightness ratio = "
        f"{summary['randproj_median_tightness_ratio_mean_over_seeds']:.4f} "
        f"± {summary['randproj_median_tightness_ratio_std_over_seeds']:.4f} "
        f"(range {summary['randproj_median_min_over_seeds']:.4f}-{summary['randproj_median_max_over_seeds']:.4f})")
    beats = summary["randproj_median_tightness_ratio_mean_over_seeds"] > summary["pca_median_tightness_ratio"]
    log(f"  => random projection {'BEATS' if beats else 'does NOT beat'} PCA under the centroid-based estimator")

    out = {"purpose": "score the eight-dataset encoder comparison and "
                      "insert a random-projection baseline",
           "per_dataset": per_ds, "summary": summary}
    (RESULTS / "eight_dataset_tightness.json").write_text(json.dumps(out, indent=2))
    log(f"=== done -> {RESULTS / 'eight_dataset_tightness.json'} ===")


if __name__ == "__main__":
    main()
