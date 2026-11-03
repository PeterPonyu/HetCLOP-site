"""G3 -- theta sweep beyond one encoder.

The phase2d E1 theta sweep (impose a scalar NB dispersion theta over a grid incl.
inf=Poisson, holding NB's OWN fitted count mean FIXED, no retraining per theta)
was run on pca_512 ONLY. This repeats it on randproj_512 and scgpt_512 (Arm A, 3
seeds) to test whether the two load-bearing pca_512 findings reproduce:
  * the over-inflation DIRECTION (expression-restricted signed r_var > 1 at the
    fitted theta; decreasing as theta rises) and
  * the Poisson-like energy PLATEAU for theta >= 20 (finite-theta argmin within
    the count-redraw CI of theta->inf),
and the theta at which the expression-restricted signed r_var crosses 1.0 (on
pca_512 in Arm A it NEVER crossed -- even Poisson over-inflated ~7%).

Reuses the phase2d E1 helpers byte-for-byte (flip_theta, energy_ci_over_count_
draws, paired, per_group_map) and the shared NB machinery; only the encoder and
the orchestration loop differ. Boundary rule: all code under HetCLOP/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2D = HERE.parent / "phase2d_completion_2026-07-09"
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
for p in (PHASE2D, PHASE2B, PHASE2A, HERE):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

import e1_theta_sweep as e1  # noqa: E402  (reuse its helpers verbatim)
from run_phase2b_v2 import CACHE, DEVICE, GATE_TOL  # noqa: E402
from src.architecture.stochastic_decoder import (  # noqa: E402
    log_theta_stats, sample_nb, sample_nb_fixed_theta, train_nb_decoder,
)
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
ENCODERS = ["randproj_512", "scgpt_512"]
LOGTHETA_MIN, LOGTHETA_MAX = e1.LOGTHETA_MIN, e1.LOGTHETA_MAX
FIXED_THETAS = e1.FIXED_THETAS
K_ENERGY = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sweep_one(name, Z, tr, va, te, nva, logX, raw, lib, mean_lib,
              lab_te, ds_te, seed):
    """One encoder x seed: replicate e1.main's per-seed body (train det + NB, run
    the imposed-theta grid, direction flips, energy plateau, paired nb-vs-poisson).
    """
    Ztest_arg = np.vstack([Z[va], Z[te]])
    det_log_all, _, _, _ = train_decoder(
        Z[tr], logX[tr], Z[va], logX[va], Ztest_arg, seed=seed, device=DEVICE)
    det_log_te = det_log_all[nva:]
    det_r = reconstruction_metrics(det_log_te, logX[te])["mean_gene_pearson"]

    mu_te, theta_te, nb_val, _, nb_np = train_nb_decoder(
        Z[tr], raw[tr], lib[tr], Z[va], raw[va], lib[va], Z[te], lib[te],
        mean_libsize=mean_lib, seed=seed, device=DEVICE,
        logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
        warmup_epochs=15, logtheta_lr_scale=0.1)
    nb_mean_log_te = np.log1p(np.clip(mu_te, 0.0, None)).astype(np.float32)
    nb_r = reconstruction_metrics(nb_mean_log_te, logX[te])["mean_gene_pearson"]
    gap = det_r - nb_r
    gate_passed = mean_acceptance_gate(det_r, nb_r, GATE_TOL)
    lt = np.log(np.clip(theta_te, 1e-8, None))
    lt_stats = log_theta_stats(lt, LOGTHETA_MIN, LOGTHETA_MAX)
    fitted_median_theta = float(lt_stats["theta_median"])
    log(f"  [{name} s{seed}] det_r={det_r:.4f} nb_r={nb_r:.4f} gap={gap:+.4f} "
        f"gate={'PASS' if gate_passed else 'FAIL'} fitted_theta={fitted_median_theta:.3f}")

    grid = sorted(set(FIXED_THETAS) | {fitted_median_theta},
                  key=lambda x: (np.isinf(x), x))
    nb_counts = sample_nb(mu_te, theta_te, seed=seed)
    agg_nb = aggregate_over_groups(logX[te], np.log1p(nb_counts).astype(np.float32),
                                   lab_te, ds_te, min_cells=20, energy_subsample=300,
                                   top_k=200, seed=0)
    map_nb = e1.per_group_map(agg_nb)

    sweep, maps_fixed = [], {}
    for i, theta in enumerate(grid):
        counts = sample_nb_fixed_theta(mu_te, theta, seed=(seed * 1000 + i))
        agg = aggregate_over_groups(logX[te], np.log1p(counts).astype(np.float32),
                                    lab_te, ds_te, min_cells=20, energy_subsample=300,
                                    top_k=200, seed=0)
        e_mean, e_sd, e_seeds = e1.energy_ci_over_count_draws(
            logX[te], mu_te, theta, lab_te, ds_te, min_cells=20, subsample=300,
            k_draws=K_ENERGY, base_seed=(seed * 1000 + i))
        tkey = "inf" if np.isinf(theta) else round(float(theta), 4)
        maps_fixed[tkey] = e1.per_group_map(agg)
        sweep.append({
            "theta": ("inf" if np.isinf(theta) else float(theta)),
            "is_fitted_median": bool(abs(theta - fitted_median_theta) < 1e-9),
            "r_var_signed_median": agg["r_var_signed"]["median"],
            "r_var_abs_log2_median": agg["r_var_abs_log2"]["median"],
            "expr_signed_median": agg["r_var_signed_expressed"]["median"],
            "expr_meanlog2_median": agg["r_var_meanlog2_expressed"]["median"],
            "expr_frac_gt1_median": agg["r_var_frac_gt1_expressed"]["median"],
            "topk_signed_median": agg["r_var_signed_topk"]["median"],
            "topk_frac_gt1_median": agg["r_var_frac_gt1_topk"]["median"],
            "energy_distance_median": agg["energy_distance"]["median"],
            "energy_ci_mean": e_mean, "energy_ci_sd": e_sd, "energy_ci_seeds": e_seeds,
        })
    for rec in sweep:
        tstr = rec["theta"] if isinstance(rec["theta"], str) else f"{rec['theta']:.3f}"
        log(f"    theta={tstr:>8s} EXPR signed={rec['expr_signed_median']:.3f} "
            f"frac>1={rec['expr_frac_gt1_median']:.3f} "
            f"energy={rec['energy_ci_mean']:.4f}+/-{rec['energy_ci_sd']:.4f}"
            f"{'  <-fitted' if rec['is_fitted_median'] else ''}")

    theta_vals = [np.inf if r["theta"] == "inf" else r["theta"] for r in sweep]
    flips = {
        "expr_signed_median_vs_1": e1.flip_theta(
            theta_vals, [r["expr_signed_median"] for r in sweep], 1.0),
        "expr_frac_gt1_vs_0.5": e1.flip_theta(
            theta_vals, [r["expr_frac_gt1_median"] for r in sweep], 0.5),
        "topk_signed_median_vs_1": e1.flip_theta(
            theta_vals, [r["topk_signed_median"] for r in sweep], 1.0),
    }
    fit_rec = next(r for r in sweep if r["is_fitted_median"])
    direction_at_fitted = {
        "fitted_theta": fitted_median_theta,
        "expr_signed_median": fit_rec["expr_signed_median"],
        "expr_frac_gt1_median": fit_rec["expr_frac_gt1_median"],
        "over_inflated_expr": bool(fit_rec["expr_frac_gt1_median"] > 0.5),
    }

    # Poisson-like plateau for theta >= 20: finite-theta argmin vs inf within CI.
    energy_ci = {("inf" if r["theta"] == "inf" else r["theta"]):
                 (r["energy_ci_mean"], r["energy_ci_sd"]) for r in sweep}
    finite_items = [(t, m, s) for t, (m, s) in energy_ci.items() if t != "inf"]
    argmin_t, argmin_m, argmin_s = min(finite_items, key=lambda x: x[1])
    inf_m, inf_s = energy_ci["inf"]
    sep = abs(argmin_m - inf_m)
    pooled_sd = float(np.hypot(argmin_s, inf_s))
    # plateau region: theta >= 20 all within CI of inf?
    plateau_ok = all(
        abs(m - inf_m) <= float(np.hypot(s, inf_s))
        for t, (m, s) in energy_ci.items()
        if t != "inf" and float(t) >= 20.0)
    energy_optimum = {
        "finite_argmin_theta": argmin_t,
        "finite_argmin_energy_mean": argmin_m, "inf_energy_mean": inf_m,
        "argmin_vs_inf_separation": float(sep), "pooled_subsample_sd": pooled_sd,
        "distinguishable_from_poisson": bool(sep > pooled_sd),
        "plateau_theta_ge_20_within_ci_of_inf": bool(plateau_ok),
    }
    pw = {
        "energy_distance": e1.paired(map_nb, maps_fixed["inf"], "energy_distance"),
        "r_var_abs_log2": e1.paired(map_nb, maps_fixed["inf"], "r_var_abs_log2"),
    }
    return {
        "gate": {"det_log_pearson": float(det_r), "nb_mean_log_pearson": float(nb_r),
                 "gap_vs_det": float(gap), "gate_passed": gate_passed,
                 "fitted_median_theta": fitted_median_theta, "logtheta_stats": lt_stats,
                 "nb_val_nll": float(nb_val), "nb_params": int(nb_np)},
        "theta_sweep": sweep, "restricted_direction_flips": flips,
        "direction_at_fitted_theta": direction_at_fitted,
        "energy_optimum": energy_optimum,
        "paired_nb_stochastic_vs_poisson_on_nb_mean": pw,
    }


def main() -> None:
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())
    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())
    lab_te, ds_te = labels[te], dataset_ids[te]
    log(f"Arm A: logX={logX.shape} test={int(te.sum())}")

    enc = {}
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    enc["randproj_512"] = (logX @ W).astype(np.float32)
    pA = CACHE / "enc_A_scgpt.npy"
    if pA.exists():
        z = np.load(pA)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
    encoders = [e for e in ENCODERS if e in enc]
    log(f"encoders: {encoders}; seeds={SEEDS}")

    per_encoder = {}
    for name in encoders:
        Z = enc[name]
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = sweep_one(
                name, Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_te, ds_te, seed)
        # cross-seed direction summary
        fracs = [per_seed[str(s)]["direction_at_fitted_theta"]["expr_frac_gt1_median"]
                 for s in SEEDS]
        crosses = [per_seed[str(s)]["restricted_direction_flips"]["expr_signed_median_vs_1"]["theta"]
                   for s in SEEDS]
        plateaus = [per_seed[str(s)]["energy_optimum"]["plateau_theta_ge_20_within_ci_of_inf"]
                    for s in SEEDS]
        per_encoder[name] = {
            "per_seed": per_seed,
            "summary": {
                "mean_expr_frac_gt1_at_fitted": float(np.mean(fracs)),
                "over_inflated_at_fitted_all_seeds": bool(all(f > 0.5 for f in fracs)),
                "expr_signed_crosses_1_thetas": crosses,
                "expr_signed_never_crosses_1_all_seeds": bool(all(c is None for c in crosses)),
                "poisson_plateau_reproduces_all_seeds": bool(all(plateaus)),
            },
        }
        log(f"  [{name} SUMMARY] over_inflated_all_seeds="
            f"{per_encoder[name]['summary']['over_inflated_at_fitted_all_seeds']} "
            f"never_crosses_1={per_encoder[name]['summary']['expr_signed_never_crosses_1_all_seeds']} "
            f"plateau_all={per_encoder[name]['summary']['poisson_plateau_reproduces_all_seeds']}")

    out = {
        "purpose": "G3: repeat the phase2d E1 imposed-theta sweep (NB's own fitted "
                   "mean held fixed) on randproj_512 and scgpt_512 (Arm A, 3 seeds). "
                   "Tests reproduction of the over-inflation direction, the "
                   "Poisson-like energy plateau (theta>=20), and the theta at which "
                   "expression-restricted signed r_var crosses 1.0.",
        "arm": "A", "seeds": SEEDS, "encoders": encoders,
        "fixed_theta_grid": [("inf" if np.isinf(t) else t) for t in FIXED_THETAS],
        "energy_ci_count_draws": K_ENERGY,
        "reference_pca512_armA": ("expr signed r_var NEVER crossed 1.0; even Poisson "
                                  "(theta->inf) over-inflated ~7%; energy on a "
                                  "Poisson-like plateau for theta>=20."),
        "per_encoder": per_encoder,
    }
    (RESULTS / "g3_theta_sweep.json").write_text(json.dumps(out, indent=2))
    log(f"=== G3 done in {time.time()-t0:.0f}s -> {RESULTS / 'g3_theta_sweep.json'} ===")


if __name__ == "__main__":
    main()
