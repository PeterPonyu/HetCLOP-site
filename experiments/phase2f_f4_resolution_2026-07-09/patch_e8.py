"""Recompute E8 (energy-estimator stability) with the CORRECTED noise source and
patch it into results/phase2f.json.

The first pipeline run's E8 varied the energy SUBSAMPLE seed, which is inert on
this data (max test group 286 cells < 300 -> no subsampling ever happens), so all
10 draws were identical. The genuine single-draw noise is the STOCHASTIC COUNT
SAMPLE. This script refits the seed-44 NB decoder for pca_512 and scgpt_512, holds
(mu, theta) fixed, and redraws nb_stochastic / poisson_on_nb_mean counts 10 times,
recomputing the paired energy test each draw. It rewrites ONLY the
``E8_energy_stability`` field of phase2f.json; E6/E7 are untouched.

"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_phase2f import (  # noqa: E402  (adds sys.path, imports heavy deps)
    CACHE, E8_DRAWS, E8_ENCODERS, E8_SEED, SEEDS, aggregate_over_groups,
    build_all_encoders, e8_count_redraw_stability, fit_nb_mean_theta,
    per_group_map,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())
    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())
    lab_te, ds_te = labels[te], dataset_ids[te]

    enc = build_all_encoders(logX, tr, ordered)

    # max test group size (verify the subsample is inert)
    agg0 = aggregate_over_groups(logX[te], logX[te], lab_te, ds_te,
                                 min_cells=20, energy_subsample=300, top_k=200, seed=0)
    max_group = int(max(g["n_cells"] for g in per_group_map(agg0).values()))
    log(f"max test group = {max_group} cells; subsample_inert={max_group <= 300}")

    e8 = {"seed_decoder": E8_SEED, "n_draws": E8_DRAWS, "encoders": [],
          "max_test_group_cells": max_group,
          "subsample_inert": bool(max_group <= 300),
          "noise_source": "stochastic count sample (energy subsample is inert: "
                          "max test group %d cells <= 300)" % max_group,
          "per_encoder": {}}
    for name in E8_ENCODERS:
        if name not in enc:
            continue
        e8["encoders"].append(name)
        log(f"=== E8 refit {name} seed {E8_SEED} ===")
        mu_te, theta_te = fit_nb_mean_theta(
            enc[name], tr, va, te, nva, raw, lib, mean_lib, E8_SEED)
        summ = e8_count_redraw_stability(mu_te, theta_te, logX[te], lab_te, ds_te, E8_DRAWS)
        e8["per_encoder"][name] = summ
        log(f"  [{name}] naive_p in [{summ['naive_p_min']:.2e}, {summ['naive_p_max']:.2e}] "
            f"({summ['naive_p_range_orders_of_magnitude']:.1f} orders) "
            f"any>0.05={summ['naive_p_any_above_0.05']} "
            f"nb_win in [{summ['nb_win_fraction_min']:.3f},{summ['nb_win_fraction_max']:.3f}] "
            f"clustWil_p in [{summ['clustered_wilcoxon_p_min']:.3f},{summ['clustered_wilcoxon_p_max']:.3f}]")

    path = RESULTS / "phase2f.json"
    out = json.loads(path.read_text())
    out["E8_energy_stability"] = e8
    out["E8_note"] = ("E8 recomputed by patch_e8.py: the energy subsample is inert "
                      "(max group %d < 300), so single-draw noise = stochastic count "
                      "sample; 10 count redraws around the fixed seed-44 mu/theta." % max_group)
    path.write_text(json.dumps(out, indent=2))
    (RESULTS / "e8_stability.json").write_text(json.dumps(e8, indent=2))
    log(f"=== E8 patched in {time.time()-t0:.0f}s -> {path} ===")


if __name__ == "__main__":
    main()
