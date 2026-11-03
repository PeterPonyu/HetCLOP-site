"""Phase 2v-follow-up: 3-seed end-to-end generation robustness test.

Extends phase2v (single seed) to three independent random seeds. Each seed fits
scVI `gene` and `gene-cell` modes from scratch and samples latents from the fitted
prior. The Poisson baseline is the dispersion-free draw from the gene-cell model.
Aggregate statistics are written into a separate summary JSON for downstream
figures and manuscript macros.
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

import phase2j_lib as L  # noqa: E402
from run_e2e import (  # noqa: E402
    _log1p, evaluate, fit_model, label_transfer, sample_generative,
    sample_nb, sample_nb_fixed_theta,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [1, 2, 3]
DEVICE = "cuda"
MAX_EPOCHS = 85


def run_one_seed(data, seed, real, lab_te, real_ms, types, rare_types):
    t0_seed = time.time()
    L.log(f"=== seed {seed} ===")
    n_te = int(data["te"].sum())
    out = {"seed": seed, "n_synthetic_per_arm": n_te, "arms": {}}
    for mode in ["gene", "gene-cell"]:
        L.log(f"seed {seed} | fitting scVI dispersion={mode} ...")
        # Patch the module-level SEED for this fit/eval cycle.
        import run_e2e
        run_e2e.SEED = seed
        model, A_tr, n_epochs = fit_model(data, mode)
        L.log(f"seed {seed} | {mode} fit done ({n_epochs} epochs, {time.time() - t0_seed:.0f}s); sampling {n_te} latents")
        mu, theta, z_syn = sample_generative(model, A_tr, data, n_te)
        lab_syn = label_transfer(model, A_tr, data, z_syn)
        draws = {f"e2e_nb_{mode}": _log1p(sample_nb(mu, theta, seed=seed))}
        if mode == "gene-cell":
            draws["e2e_poisson"] = _log1p(sample_nb_fixed_theta(mu, float("inf"), seed=seed))
        for name, X in draws.items():
            ev = evaluate(name, X, lab_syn, real, lab_te, real_ms, types, rare_types)
            ev["dispersion_mode"] = mode
            ev["n_epochs_trained"] = n_epochs
            ev["elapsed_s"] = round(time.time() - t0_seed, 1)
            out["arms"][name] = ev
        del model, A_tr
        torch.cuda.empty_cache()
    out["elapsed_s"] = round(time.time() - t0_seed, 1)
    return out


def main() -> int:
    t0 = time.time()
    L.log("=== Phase 2v-follow-up: 3-seed end-to-end generation robustness ===")
    data = L.load_corpus()
    real = data["logX"][data["te"]]
    lab_te = data["labels"][data["te"]]
    u, c = np.unique(lab_te, return_counts=True)
    from run_biological import MIN_CELLS, RARE_MAX
    types = [t for t, n in zip(u, c) if n >= MIN_CELLS]
    rare_types = {t for t, n in zip(u, c) if n < RARE_MAX}
    from run_biological import marker_scores
    real_ms = marker_scores(real, lab_te, types)

    per_seed = []
    for seed in SEEDS:
        per_seed.append(run_one_seed(data, seed, real, lab_te, real_ms, types, rare_types))
        (RESULTS / f"e2e_generation_seed{seed}.json").write_text(
            json.dumps(per_seed[-1], indent=2))

    # Aggregate summary across seeds.
    summary = {"purpose": "3-seed end-to-end generation robustness", "seeds": SEEDS,
               "n_synthetic_per_arm": per_seed[0]["n_synthetic_per_arm"],
               "n_types_real": len(types), "n_rare_types": len(rare_types),
               "arms": {}}
    for arm in ["e2e_nb_gene", "e2e_nb_gene-cell", "e2e_poisson"]:
        recs = [s["arms"][arm] for s in per_seed if arm in s["arms"]]
        if not recs:
            continue
        metrics = ["marker_recall_mean", "marker_recall_median", "tstr_macro_f1", "tstr_rare_f1"]
        agg = {}
        for m in metrics:
            vals = [r[m] for r in recs]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{m}_median"] = float(np.median(vals))
            agg[f"{m}_min"] = float(np.min(vals))
            agg[f"{m}_max"] = float(np.max(vals))
        # keep per-seed values too
        agg["per_seed"] = {m: [r[m] for r in recs] for m in metrics}
        summary["arms"][arm] = agg

    summary["elapsed_total_s"] = round(time.time() - t0, 1)
    (RESULTS / "e2e_generation_multiseed.json").write_text(json.dumps(summary, indent=2))

    L.log(f"=== done in {time.time() - t0:.0f}s; summary -> {RESULTS / 'e2e_generation_multiseed.json'} ===")
    for arm, a in summary["arms"].items():
        L.log(f"{arm}: marker_recall_mean={a['marker_recall_mean_mean']:.4f} ± {a['marker_recall_mean_std']:.4f}; "
              f"TSTR={a['tstr_macro_f1_mean']:.4f} ± {a['tstr_macro_f1_std']:.4f}; "
              f"rare={a['tstr_rare_f1_mean']:.4f} ± {a['tstr_rare_f1_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
