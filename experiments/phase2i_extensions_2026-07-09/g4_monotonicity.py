"""G4 -- more encoders for the monotonicity leg.

Finding: the signed ENERGY effect (nb_stochastic - poisson_on_nb_mean, per-seed
median paired diff, averaged over 3 seeds) is rank-monotone in the NB mean-fit gap
(det_log_pearson - nb_mean_log_pearson) with Spearman rho = -1.0. But it rests on
only n = 4 encoders, where the exact two-sided permutation p can NEVER reach 0.05
(min = 2/4! = 2/24 = 0.083). This adds encoders (pca_50, randproj_50, pca_1250)
to make the relationship testable, and reports rho, Pearson r, and the EXACT
permutation p over all n! orderings (never scipy's degenerate p=0.0 for |rho|=1).

Reuse: the four existing encoders' (gap, energy_effect, rvar_effect) are taken
verbatim from phase2f.json's E6 per-encoder summary; pca_512 is RECOMPUTED here as
an identity cross-check (must match phase2f to ~1e-6) proving the effect pipeline
is byte-for-byte the same before the three new encoders are appended. Arm A.

"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2F = HERE.parent / "phase2f_f4_resolution_2026-07-09"
PHASE2C = HERE.parent / "phase2c_robustness_2026-07-09"
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
PHASE2G = HERE.parent / "phase2g_armb_f4_2026-07-09"
for p in (PHASE2F, PHASE2C, PHASE2B, PHASE2A, PHASE2G, HERE):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from run_phase2b_v2 import CACHE  # noqa: E402
from run_nb_arm import build_nb_variants  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from clustered_inference import naive_paired, paired_group_diffs  # noqa: E402
from phase2g_lib import exact_spearman_perm_p  # noqa: E402
from phase2i_lib import pca_k_encoder, randproj_encoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
METRICS = ["energy_distance", "r_var_abs_log2_expr"]
CONTROL = "poisson_on_nb_mean"
NEW_ENCODERS = ["pca_50", "randproj_50", "pca_1250"]
PHASE2F_JSON = PHASE2F / "results" / "phase2f.json"
XCHECK_TOL = 1e-6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def compute_effects(name, Z, tr, va, te, nva, logX, raw, lib, mean_lib,
                    lab_va, lab_te, ds_te):
    """(mean_fit_gap, energy_effect, rvar_effect) over 3 seeds -- the EXACT phase2f
    E6 definitions: gap = det-nb; effect = mean over seeds of median paired diff
    (nb_stochastic - poisson_on_nb_mean), positive => Poisson better."""
    gaps, e_eff, r_eff, gates = [], [], [], []
    for seed in SEEDS:
        decoded_log, gate = build_nb_variants(
            Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
        amap = {v: per_group_map(aggregate_over_groups(
            logX[te], decoded_log[v], lab_te, ds_te, min_cells=20,
            energy_subsample=300, top_k=200, seed=0))
            for v in ("nb_stochastic", CONTROL)}
        gaps.append(gate["gap_vs_det"])
        gates.append(gate["gate_passed"])
        eff = {}
        for metric in METRICS:
            recs = paired_group_diffs(amap["nb_stochastic"], amap[CONTROL], metric)
            eff[metric] = naive_paired(recs)["median_paired_diff"]
        e_eff.append(eff["energy_distance"])
        r_eff.append(eff["r_var_abs_log2_expr"])
    return {
        "mean_fit_gap": float(np.mean(gaps)),
        "energy_effect": float(np.mean(e_eff)),
        "rvar_expr_effect": float(np.mean(r_eff)),
        "gate_pass_seeds": int(sum(gates)),
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
    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    log(f"Arm A: logX={logX.shape} test={int(te.sum())}")

    # ---- Existing 4 encoders: reuse phase2f E6 summary verbatim ----------------
    p2f = json.loads(PHASE2F_JSON.read_text())
    e6f = p2f["E6_four_encoder_dispersion"]["per_encoder_summary"]
    existing = {}
    for name, s in e6f.items():
        existing[name] = {
            "mean_fit_gap": s["mean_fit_gap"],
            "energy_effect": s["energy_distance"]["mean_median_paired_diff"],
            "rvar_expr_effect": s["r_var_abs_log2_expr"]["mean_median_paired_diff"],
            "gate_pass_seeds": s.get("gate_pass_seeds"),
            "source": "phase2f.json",
        }
    log(f"reused phase2f encoders: {list(existing)}")

    # ---- Identity cross-check: recompute pca_512 and match phase2f -------------
    encP = pca_k_encoder(logX, tr, 512)
    xchk = compute_effects("pca_512", encP, tr, va, te, nva, logX, raw, lib,
                           mean_lib, lab_va, lab_te, ds_te)
    ref = existing["pca_512"]
    d_gap = abs(xchk["mean_fit_gap"] - ref["mean_fit_gap"])
    d_e = abs(xchk["energy_effect"] - ref["energy_effect"])
    d_r = abs(xchk["rvar_expr_effect"] - ref["rvar_expr_effect"])
    cross_check = {"d_gap": d_gap, "d_energy_effect": d_e, "d_rvar_effect": d_r,
                   "tol": XCHECK_TOL,
                   "passed": bool(d_gap <= XCHECK_TOL and d_e <= XCHECK_TOL and d_r <= XCHECK_TOL)}
    log(f"  [XCHECK pca_512] d_gap={d_gap:.2e} d_e={d_e:.2e} d_r={d_r:.2e} "
        f"{'PASS' if cross_check['passed'] else 'FAIL'}")
    assert cross_check["passed"], (
        f"pca_512 effect recomputation does not match phase2f.json within "
        f"{XCHECK_TOL}: the effect pipeline changed -- refusing to append new "
        f"encoders to an inconsistent series. {cross_check}")

    # ---- New encoders ----------------------------------------------------------
    new_enc = {
        "pca_50": pca_k_encoder(logX, tr, 50),
        "randproj_50": randproj_encoder(logX, 50, seed=0),
        "pca_1250": pca_k_encoder(logX, tr, 1250),
    }
    new = {}
    for name in NEW_ENCODERS:
        log(f"=== compute effects {name} (dim={new_enc[name].shape[1]}) ===")
        new[name] = compute_effects(name, new_enc[name], tr, va, te, nva, logX, raw,
                                    lib, mean_lib, lab_va, lab_te, ds_te)
        new[name]["source"] = "phase2i_g4"
        log(f"  [{name}] gap={new[name]['mean_fit_gap']:+.4f} "
            f"e_eff={new[name]['energy_effect']:+.6f} "
            f"r_eff={new[name]['rvar_expr_effect']:+.6f} "
            f"gate_pass={new[name]['gate_pass_seeds']}/3")

    # ---- Combined series -------------------------------------------------------
    all_enc = {**existing, **new}
    names = list(all_enc.keys())
    gaps = np.array([all_enc[n]["mean_fit_gap"] for n in names])
    e_eff = np.array([all_enc[n]["energy_effect"] for n in names])
    r_eff = np.array([all_enc[n]["rvar_expr_effect"] for n in names])
    n = len(names)

    sp_e = spearmanr(gaps, e_eff)
    sp_r = spearmanr(gaps, r_eff)
    exact_e = exact_spearman_perm_p(gaps, e_eff)
    exact_r = exact_spearman_perm_p(gaps, r_eff)
    stats = {
        "n_encoders": n,
        "encoders": names,
        "mean_fit_gap": gaps.tolist(),
        "energy_effect_nb_minus_poisson": e_eff.tolist(),
        "rvar_expr_effect_nb_minus_poisson": r_eff.tolist(),
        "gap_vs_energy_effect": {
            "spearman_rho": float(sp_e.statistic),
            "spearman_p_scipy_DEGENERATE_for_rho1": float(sp_e.pvalue),
            "pearson_r": float(pearsonr(gaps, e_eff)[0]),
            "pearson_p": float(pearsonr(gaps, e_eff)[1]),
            "exact_permutation_p": exact_e,
            "reaches_significance_0.05_two_sided": bool(exact_e["p_two_sided"] < 0.05),
        },
        "gap_vs_rvar_expr_effect": {
            "spearman_rho": float(sp_r.statistic),
            "pearson_r": float(pearsonr(gaps, r_eff)[0]),
            "exact_permutation_p": exact_r,
            "reaches_significance_0.05_two_sided": bool(exact_r["p_two_sided"] < 0.05),
        },
        "min_possible_two_sided_perm_p": 2.0 / math.factorial(n) if n <= 10 else None,
    }
    log(f"  [GAP vs ENERGY] rho={sp_e.statistic:.3f} pearson_r={stats['gap_vs_energy_effect']['pearson_r']:.3f} "
        f"exact_perm p1={exact_e['p_one_sided']:.4f} p2={exact_e['p_two_sided']:.4f} "
        f"significant={stats['gap_vs_energy_effect']['reaches_significance_0.05_two_sided']}")

    out = {
        "purpose": "G4: extend the mean-fit-gap vs signed-energy-effect "
                   "monotonicity leg from n=4 to more encoders (pca_50, randproj_50, "
                   "pca_1250) so the exact n!-permutation p can reach 0.05. Reuses "
                   "phase2f's four-encoder effects verbatim; recomputes pca_512 as an "
                   "identity cross-check. Arm A, 3 seeds.",
        "arm": "A", "seeds": SEEDS, "control": CONTROL,
        "effect_definition": ("energy_effect = mean over 3 seeds of median paired "
                              "diff (nb_stochastic - poisson_on_nb_mean) on the "
                              "metric; positive => Poisson better. gap = "
                              "det_log_pearson - nb_mean_log_pearson (>0 => NB mean "
                              "underfits)."),
        "reused_from_phase2f": list(existing),
        "new_encoders": NEW_ENCODERS,
        "pca_512_identity_cross_check": {"recomputed": xchk, "phase2f": ref,
                                         "diffs": cross_check},
        "per_encoder": all_enc,
        "statistics": stats,
    }
    (RESULTS / "g4_monotonicity.json").write_text(json.dumps(out, indent=2))
    log(f"=== G4 done in {time.time()-t0:.0f}s -> {RESULTS / 'g4_monotonicity.json'} ===")


if __name__ == "__main__":
    main()
