"""Phase 2f -- resolve finding F4 ("learned NB dispersion is counterproductive").

An adversarial audit of the paper's F4 found three problems with the load-bearing
``nb_stochastic`` vs ``poisson_on_nb_mean`` (theta->inf) contrast:

  1. The ENERGY leg reverses sign on the gate-EXCLUDED encoders (scgpt_512,
     geneformer): NB WINS there, ~4x larger effect, opposite the pca_512 verdict.
     The acceptance gate cannot justify excluding those counterexamples for THIS
     contrast, because ``poisson_on_nb_mean`` holds NB's OWN fitted mean fixed in
     both arms -- a mean underfit cannot confound a pure dispersion contrast.
  2. Pseudoreplication: 314 (dataset x cell type) groups from only 50 datasets,
     tested as if independent. The PCA energy leg wins only 62.4% of groups.
  3. Cross-implementation instability: the same seed-44 PCA energy p swings ~7x
     between two draws (5.83e-3 vs 7.90e-4) -- near the sampling-noise floor.

Three experiments:

  E6 -- report the ``nb_stochastic`` vs ``poisson_on_nb_mean`` dispersion contrast
        on ALL FOUR encoders, gate-independent, 3 seeds, both valid metrics
        (energy_distance and EXPRESSION-RESTRICTED |log2 r_var|). Test the
        hypothesis that the signed energy effect is monotone in the NB mean-fit
        gap (det_log_pearson - nb_mean_log_pearson): 4 points, regress + order.
  E7 -- recompute significance HONOURING the 50-dataset nesting: collapse to 50
        dataset-level medians (Wilcoxon + sign test) and a cluster-level
        sign-flip permutation. Naive 314-group p reported side by side.
  E8 -- energy-estimator stability: repeat the entire paired energy test over 10
        independent <=300-cell subsample draws on pca_512 and scgpt_512.

SANITY (rule #3): recomputing the naive 314-group UNRESTRICTED tests must
reproduce nb_arm.json to ~1e-6 -- proving only the intended thing changed.

Reuses ``build_nb_variants`` byte-for-byte from phase2c ``run_nb_arm`` and the
geneformer/randproj encoder construction from phase2d ``e2_nb_generalization``.

"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2C = HERE.parent / "phase2c_robustness_2026-07-09"
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
for p in (PHASE2C, PHASE2B, PHASE2A, HERE):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from run_phase2b_v2 import CACHE, DEVICE, load_geneformer_armA  # noqa: E402
from run_nb_arm import LOGTHETA_MAX, LOGTHETA_MIN, build_nb_variants  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, train_nb_decoder  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402

from clustered_inference import (  # noqa: E402
    clustered_dataset_medians, dataset_signflip_permutation, naive_paired,
    paired_group_diffs, within_dataset_signflip_permutation,
)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
ENCODERS = ["pca_512", "randproj_512", "scgpt_512", "geneformer"]
# The two valid discriminators (spec §16 E4: unrestricted r_var is discredited;
# use the EXPRESSION-RESTRICTED analogue). Both are lower-is-better.
METRICS = ["energy_distance", "r_var_abs_log2_expr"]
# Unrestricted metrics used ONLY for the nb_arm.json reproduction sanity check.
NAIVE_REPRO_METRICS = ["energy_distance", "r_var_abs_log2"]
CONTROL = "poisson_on_nb_mean"
N_PERM = 10000
E8_SEED = 44          # matches the audit's seed-44 cross-implementation reference
E8_ENCODERS = ["pca_512", "scgpt_512"]
E8_DRAWS = 10
NB_ARM_JSON = PHASE2C / "results" / "nb_arm.json"
REPRO_TOL = 1e-6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def fit_nb_mean_theta(Z, tr, va, te, nva, raw, lib, mean_lib, seed):
    """Train the NB decoder EXACTLY as build_nb_variants does and return its
    fitted per-cell count mean ``mu_te`` and per-(cell,gene) inverse-dispersion
    ``theta_te`` on the TEST cells. Used by E8 to redraw counts around a fixed
    (mu, theta) so the energy-estimator's single-draw noise can be quantified."""
    Ztest_arg = np.vstack([Z[va], Z[te]])
    lib_arg = np.concatenate([lib[va], lib[te]]).astype(np.float32)
    mu_all, theta_all, _, _, _ = train_nb_decoder(
        Z[tr], raw[tr], lib[tr], Z[va], raw[va], lib[va], Ztest_arg, lib_arg,
        mean_libsize=mean_lib, seed=seed, device=DEVICE,
        logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
        warmup_epochs=15, logtheta_lr_scale=0.1)
    return mu_all[nva:], theta_all[nva:]


def e8_count_redraw_stability(mu_te, theta_te, logX_te, lab_te, ds_te, n_draws):
    """E8 core: with the NB fitted (mu, theta) held FIXED, redraw both
    nb_stochastic (NB counts) and poisson_on_nb_mean (Poisson(mu)) counts
    ``n_draws`` independent times and recompute the paired ENERGY test each time.

    This is the correct single-draw noise source: the <=300-cell energy subsample
    is INERT here (max test group is 286 cells < 300, so every group uses all its
    cells with no subsampling), so the only stochasticity in the energy leg is the
    count sample itself -- exactly what drives the ~7x cross-implementation p
    swing the audit flagged. Returns (draws, summary)."""
    mu_c = np.clip(mu_te, 0.0, None)
    draws = []
    for k in range(n_draws):
        nb_counts = sample_nb(mu_te, theta_te, seed=(1000 + k))
        pois_counts = np.random.default_rng((2000, k)).poisson(mu_c).astype(np.float32)
        dec = {"nb_stochastic": np.log1p(nb_counts),
               "poisson_on_nb_mean": np.log1p(pois_counts)}
        amap = {}
        for v, d in dec.items():
            agg = aggregate_over_groups(logX_te, d, lab_te, ds_te,
                                        min_cells=20, energy_subsample=300,
                                        top_k=200, seed=0)
            amap[v] = per_group_map(agg)
        recs = paired_group_diffs(amap["nb_stochastic"], amap["poisson_on_nb_mean"],
                                  "energy_distance")
        naive = naive_paired(recs)
        clust = clustered_dataset_medians(recs)
        draws.append({
            "draw": k,
            "naive_p": naive["p_value"],
            "nb_win_fraction": naive["nb_win_fraction"],
            "median_paired_diff": naive["median_paired_diff"],
            "clustered_wilcoxon_p": clust["wilcoxon_p"],
            "clustered_sign_p": clust["sign_test_p"]})
    ps = np.array([x["naive_p"] for x in draws])
    wins = np.array([x["nb_win_fraction"] for x in draws])
    clps = np.array([x["clustered_wilcoxon_p"] for x in draws])
    summary = {
        "n_draws": int(n_draws),
        "naive_p_min": float(ps.min()), "naive_p_max": float(ps.max()),
        "naive_p_median": float(np.median(ps)),
        "naive_p_range_orders_of_magnitude":
            float(np.log10(ps.max() / max(ps.min(), 1e-300))),
        "naive_p_any_above_0.05": bool((ps > 0.05).any()),
        "naive_p_all_below_0.05": bool((ps < 0.05).all()),
        "nb_win_fraction_min": float(wins.min()), "nb_win_fraction_max": float(wins.max()),
        "nb_win_fraction_median": float(np.median(wins)),
        "clustered_wilcoxon_p_min": float(clps.min()),
        "clustered_wilcoxon_p_max": float(clps.max()),
        "clustered_wilcoxon_p_median": float(np.median(clps)),
        "draws": draws,
    }
    return summary


def build_all_encoders(logX, tr, ordered):
    """pca_512 + randproj_512 (byte-for-byte phase2b/phase2d) + scgpt_512 (cache)
    + geneformer (Arm A cache). Returns dict of available encoders."""
    enc = {}
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    enc["randproj_512"] = (logX @ W).astype(np.float32)
    p = CACHE / "enc_A_scgpt.npy"
    if p.exists():
        z = np.load(p)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
            log(f"loaded scgpt_512 {z.shape}")
    gf = load_geneformer_armA(ordered)
    if gf is not None and gf.shape[0] == logX.shape[0]:
        enc["geneformer"] = gf.astype(np.float32)
        log(f"loaded geneformer Arm A {gf.shape}")
    else:
        log("geneformer Arm A MISSING -> excluded")
    return enc


def main() -> None:
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
    assert np.allclose(np.log1p(raw[:2000]), logX[:2000], atol=1e-5), \
        "logX must equal log1p(raw)"
    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())
    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    n_datasets = int(len(np.unique(ds_te)))
    log(f"raw={raw.shape} mean_lib={mean_lib:.1f} test={int(te.sum())} "
        f"n_datasets={n_datasets}")

    enc = build_all_encoders(logX, tr, ordered)
    encoders = [e for e in ENCODERS if e in enc]
    log(f"encoders available: {encoders}; seeds={SEEDS}")

    nb_arm = json.loads(NB_ARM_JSON.read_text())

    # ---- Build the two needed variants for every (encoder, seed) and aggregate.
    # aggmaps[(enc,seed)] = {variant: per_group_map}. (E8 refits mu/theta for its
    # two encoders separately, so no full decoded arrays are cached here.)
    aggmaps = {}
    gates = {}
    n_groups = None
    for name in encoders:
        Z = enc[name]
        for seed in SEEDS:
            log(f"=== build {name} seed {seed} ===")
            decoded_log, gate = build_nb_variants(
                Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
            keep = {v: decoded_log[v] for v in ("nb_stochastic", CONTROL)}
            gates[(name, seed)] = gate
            amap = {}
            for v, dec in keep.items():
                agg = aggregate_over_groups(
                    logX[te], dec, lab_te, ds_te,
                    min_cells=20, energy_subsample=300, top_k=200, seed=0)
                amap[v] = per_group_map(agg)
                if n_groups is None:
                    n_groups = agg["n_groups"]
            aggmaps[(name, seed)] = amap
            log(f"    gate={'PASS' if gate['gate_passed'] else 'FAIL'} "
                f"gap={gate['gap_vs_det']:+.4f} "
                f"det_r={gate['det_log_pearson']:.4f} nb_r={gate['nb_mean_log_pearson']:.4f}")

    # ================= SANITY: reproduce nb_arm.json (unrestricted, 314 groups) =
    sanity = {"tol": REPRO_TOL, "checks": [], "all_passed": True}
    for name in ("pca_512", "scgpt_512"):
        if name not in encoders:
            continue
        for seed in SEEDS:
            amap = aggmaps[(name, seed)]
            for metric in NAIVE_REPRO_METRICS:
                recs = paired_group_diffs(amap["nb_stochastic"], amap[CONTROL], metric)
                got = naive_paired(recs)
                fam = f"{metric}__nb_stochastic_vs_{CONTROL}"
                ref = next(r for r in nb_arm["paired_tests"][str(seed)][fam]["per_encoder"]
                           if r["encoder"] == name)
                dp = abs(got["p_value"] - ref["p_value"])
                dmed = abs(got["median_paired_diff"] - ref["median_paired_diff"])
                dwin = abs(got["nb_win_fraction"] - ref["learned_win_fraction"])
                # p-values can be tiny; compare relatively for p, absolutely for the rest
                p_ok = dp <= max(REPRO_TOL, REPRO_TOL * abs(ref["p_value"]))
                ok = bool(p_ok and dmed <= REPRO_TOL and dwin <= REPRO_TOL
                          and got["n_groups"] == ref["n_pairs"])
                sanity["checks"].append({
                    "encoder": name, "seed": seed, "metric": metric,
                    "d_p": dp, "d_median": dmed, "d_win": dwin,
                    "n_groups_match": got["n_groups"] == ref["n_pairs"], "passed": ok})
                sanity["all_passed"] &= ok
    failed = [c for c in sanity["checks"] if not c["passed"]]
    for c in sanity["checks"]:
        log(f"  [SANITY] {c['encoder']:12s} s{c['seed']} {c['metric']:20s} "
            f"dp={c['d_p']:.2e} dmed={c['d_median']:.2e} dwin={c['d_win']:.2e} "
            f"{'PASS' if c['passed'] else 'FAIL'}")
    assert not failed, (f"nb_arm.json reproduction FAILED for {failed}: something "
                        f"other than the intended machinery changed. Refusing to trust results.")
    log(f"  [SANITY] nb_arm.json reproduced to <= {REPRO_TOL} for all "
        f"{len(sanity['checks'])} unrestricted naive checks.")

    # ============================= E6: four-encoder dispersion table + hypothesis
    log("=== E6: four-encoder dispersion contrast + mean-fit-gap hypothesis ===")
    e6 = {"per_encoder_seed": [], "per_encoder_summary": {}, "hypothesis": {}}
    for name in encoders:
        for seed in SEEDS:
            amap = aggmaps[(name, seed)]
            g = gates[(name, seed)]
            row = {"encoder": name, "seed": seed,
                   "gate_passed": g["gate_passed"],
                   "det_log_pearson": g["det_log_pearson"],
                   "nb_mean_log_pearson": g["nb_mean_log_pearson"],
                   "mean_fit_gap": g["gap_vs_det"]}  # det - nb; >0 => NB mean underfits
            for metric in METRICS:
                recs = paired_group_diffs(amap["nb_stochastic"], amap[CONTROL], metric)
                np_ = naive_paired(recs)
                row[metric] = {
                    "median_paired_diff": np_["median_paired_diff"],  # nb - poisson
                    "nb_win_fraction": np_["nb_win_fraction"],
                    "naive_p": np_["p_value"], "n_groups": np_["n_groups"],
                }
            e6["per_encoder_seed"].append(row)

    # Per-encoder means across 3 seeds.
    for name in encoders:
        rows = [r for r in e6["per_encoder_seed"] if r["encoder"] == name]
        summ = {
            "gate_pass_seeds": sum(r["gate_passed"] for r in rows),
            "mean_fit_gap": float(np.mean([r["mean_fit_gap"] for r in rows])),
        }
        for metric in METRICS:
            summ[metric] = {
                "mean_median_paired_diff": float(np.mean([r[metric]["median_paired_diff"] for r in rows])),
                "mean_nb_win_fraction": float(np.mean([r[metric]["nb_win_fraction"] for r in rows])),
            }
        e6["per_encoder_summary"][name] = summ
        log(f"  [{name}] gate_pass={summ['gate_pass_seeds']}/3 "
            f"gap={summ['mean_fit_gap']:+.4f} "
            f"energy_medDiff(nb-pois)={summ['energy_distance']['mean_median_paired_diff']:+.5f} "
            f"nb_win={summ['energy_distance']['mean_nb_win_fraction']:.3f} | "
            f"rvar_expr_medDiff={summ['r_var_abs_log2_expr']['mean_median_paired_diff']:+.5f} "
            f"nb_win={summ['r_var_abs_log2_expr']['mean_nb_win_fraction']:.3f}")

    # Hypothesis: signed ENERGY effect (nb - poisson) monotone in mean-fit gap.
    names_ord = encoders
    gaps = np.array([e6["per_encoder_summary"][n]["mean_fit_gap"] for n in names_ord])
    e_eff = np.array([e6["per_encoder_summary"][n]["energy_distance"]["mean_median_paired_diff"]
                      for n in names_ord])
    r_eff = np.array([e6["per_encoder_summary"][n]["r_var_abs_log2_expr"]["mean_median_paired_diff"]
                      for n in names_ord])
    order_by_gap = list(np.argsort(gaps))
    hyp = {
        "encoders": names_ord,
        "mean_fit_gap": gaps.tolist(),
        "energy_effect_nb_minus_poisson": e_eff.tolist(),
        "rvar_expr_effect_nb_minus_poisson": r_eff.tolist(),
        "encoders_sorted_by_gap": [names_ord[i] for i in order_by_gap],
        "energy_effect_sorted_by_gap": [float(e_eff[i]) for i in order_by_gap],
        "energy_effect_monotone_decreasing_in_gap": bool(
            all(np.diff([e_eff[i] for i in order_by_gap]) <= 1e-9)),
    }
    if len(names_ord) >= 3:
        hyp["pearson_gap_vs_energy_effect"] = {
            "r": float(pearsonr(gaps, e_eff)[0]), "p": float(pearsonr(gaps, e_eff)[1])}
        hyp["spearman_gap_vs_energy_effect"] = {
            "rho": float(spearmanr(gaps, e_eff)[0]), "p": float(spearmanr(gaps, e_eff)[1])}
        hyp["spearman_gap_vs_rvar_expr_effect"] = {
            "rho": float(spearmanr(gaps, r_eff)[0]), "p": float(spearmanr(gaps, r_eff)[1])}
    e6["hypothesis"] = hyp
    log(f"  [HYP] gap order: {hyp['encoders_sorted_by_gap']}")
    log(f"  [HYP] energy effect (nb-pois) in that order: "
        f"{[round(x,5) for x in hyp['energy_effect_sorted_by_gap']]}")
    log(f"  [HYP] energy effect monotone-decreasing in gap: "
        f"{hyp['energy_effect_monotone_decreasing_in_gap']}; "
        f"spearman rho={hyp.get('spearman_gap_vs_energy_effect',{}).get('rho')}")

    # ============================= E7: dataset-clustered inference ==============
    log("=== E7: dataset-clustered inference (fixes pseudoreplication) ===")
    e7 = {"per_comparison": [], "n_datasets_expected": n_datasets, "n_perm": N_PERM}
    for name in encoders:
        for seed in SEEDS:
            amap = aggmaps[(name, seed)]
            for metric in METRICS:
                recs = paired_group_diffs(amap["nb_stochastic"], amap[CONTROL], metric)
                naive = naive_paired(recs)
                clust = clustered_dataset_medians(recs)
                perm = dataset_signflip_permutation(recs, n_perm=N_PERM, seed=0)
                perm_w = within_dataset_signflip_permutation(recs, n_perm=N_PERM, seed=0)
                comp = {
                    "encoder": name, "seed": seed, "metric": metric,
                    "gate_passed": gates[(name, seed)]["gate_passed"],
                    "naive_314group": {
                        "p": naive["p_value"],
                        "nb_win_fraction": naive["nb_win_fraction"],
                        "median_paired_diff": naive["median_paired_diff"],
                        "significant_0.05": bool(naive["p_value"] < 0.05),
                        "favors": "NB" if naive["median_paired_diff"] < 0 else "Poisson"},
                    "clustered_50dataset": {
                        "n_datasets": clust["n_datasets"],
                        "wilcoxon_p": clust["wilcoxon_p"],
                        "sign_test_p": clust["sign_test_p"],
                        "sign_test_n_nb_better": clust["sign_test_n_nb_better"],
                        "nb_win_fraction_datasets": clust["nb_win_fraction_datasets"],
                        "median_of_dataset_medians": clust["median_of_dataset_medians"],
                        "wilcoxon_significant_0.05": bool(clust["wilcoxon_p"] < 0.05),
                        "sign_significant_0.05": bool(clust["sign_test_p"] < 0.05),
                        "favors": "NB" if clust["median_of_dataset_medians"] < 0 else "Poisson"},
                    "permutation_dataset_signflip": {
                        "p": perm["p_value"], "significant_0.05": bool(perm["p_value"] < 0.05)},
                    "permutation_within_dataset_signflip": {
                        "p": perm_w["p_value"], "significant_0.05": bool(perm_w["p_value"] < 0.05)},
                    # persisted arrays
                    "per_group": recs,
                    "dataset_medians": clust["dataset_medians"],
                    "datasets": clust["datasets"],
                }
                e7["per_comparison"].append(comp)
                log(f"  [{name:12s} s{seed} {metric:20s}] "
                    f"naive_p={naive['p_value']:.2e} "
                    f"clustWil_p={clust['wilcoxon_p']:.3f} "
                    f"sign_p={clust['sign_test_p']:.3f} "
                    f"perm_p={perm['p_value']:.4f} "
                    f"nb_win_ds={clust['nb_win_fraction_datasets']:.2f} "
                    f"favors={comp['clustered_50dataset']['favors']} "
                    f"{'[gate FAIL]' if not comp['gate_passed'] else ''}")

    # ============================= E8: energy-estimator stability ================
    # The <=300-cell energy subsample is INERT on this data (max test group 286
    # cells < 300), so redrawing the subsample seed is a no-op. The genuine
    # single-draw noise in the energy leg is the STOCHASTIC COUNT SAMPLE. E8 holds
    # each seed-44 decoder's fitted (mu, theta) FIXED and redraws nb_stochastic /
    # poisson_on_nb_mean counts 10 times, recomputing the paired energy test each
    # draw -- directly quantifying the ~7x cross-implementation p swing.
    log("=== E8: energy-estimator stability (10 count redraws, fixed seed-44 mu/theta) ===")
    grp_sizes = [g["n_cells"] for g in
                 [v for name in encoders for v in
                  aggmaps[(name, SEEDS[0])]["nb_stochastic"].values()]]
    max_group = int(max(grp_sizes)) if grp_sizes else 0
    e8 = {"seed_decoder": E8_SEED, "n_draws": E8_DRAWS, "encoders": [],
          "max_test_group_cells": max_group,
          "subsample_inert": bool(max_group <= 300),
          "noise_source": "stochastic count sample (energy subsample is inert: "
                          "max test group %d cells <= 300)" % max_group,
          "per_encoder": {}}
    for name in E8_ENCODERS:
        if name not in encoders:
            continue
        e8["encoders"].append(name)
        mu_te, theta_te = fit_nb_mean_theta(
            enc[name], tr, va, te, nva, raw, lib, mean_lib, E8_SEED)
        summ = e8_count_redraw_stability(
            mu_te, theta_te, logX[te], lab_te, ds_te, E8_DRAWS)
        e8["per_encoder"][name] = summ
        log(f"  [{name}] naive_p in [{summ['naive_p_min']:.2e}, {summ['naive_p_max']:.2e}] "
            f"({summ['naive_p_range_orders_of_magnitude']:.1f} orders) "
            f"any>0.05={summ['naive_p_any_above_0.05']} "
            f"nb_win in [{summ['nb_win_fraction_min']:.3f},{summ['nb_win_fraction_max']:.3f}] "
            f"clustWil_p in [{summ['clustered_wilcoxon_p_min']:.3f},{summ['clustered_wilcoxon_p_max']:.3f}]")

    out = {
        "purpose": "Phase 2f: resolve F4 (learned NB dispersion counterproductive). "
                   "Gate-independent 4-encoder dispersion contrast (E6), "
                   "dataset-clustered inference fixing pseudoreplication (E7), "
                   "energy-estimator stability (E8).",
        "seeds": SEEDS, "encoders": encoders, "n_groups": n_groups,
        "n_datasets": n_datasets, "control": CONTROL, "metrics": METRICS,
        "metric_note": ("r_var_abs_log2_expr is the EXPRESSION-RESTRICTED |log2 r_var| "
                        "(spec §16 E4; unrestricted r_var is discredited). "
                        "energy_distance and r_var_abs_log2_expr are the only valid "
                        "discriminators. Both lower-is-better; median_paired_diff = "
                        "nb_stochastic - poisson_on_nb_mean (negative => NB better)."),
        "gate_note": ("The acceptance gate is one-sided and governs mean/procedure "
                      "conclusions, but NOT this dispersion contrast: poisson_on_nb_mean "
                      "holds NB's OWN fitted mean fixed in both arms, so a mean underfit "
                      "cannot confound it. Gate status is reported alongside every row."),
        "nb_arm_reproduction_sanity": sanity,
        "E6_four_encoder_dispersion": e6,
        "E7_clustered_inference": e7,
        "E8_energy_stability": e8,
    }
    (RESULTS / "phase2f.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'phase2f.json'} ===")


if __name__ == "__main__":
    main()
