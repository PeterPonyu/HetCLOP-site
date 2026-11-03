"""Phase 2g -- ARM-B replication of finding F4 (the honest test).

F4 (paper, restated in spec §20): holding an NB decoder's OWN fitted mean fixed,
a dispersion-free Poisson draw (``poisson_on_nb_mean``, theta->inf) beats the
learned-theta NB on EXPRESSION-RESTRICTED ``|log2 r_var|``, 3/3 seeds,
dataset-clustered p<0.001, on the two gate-passing encoders. Its mechanism claim:
"the per-cell latent already absorbs the biological cell-to-cell variation, so
fitting marginal overdispersion on top double-counts it and over-inflates
within-type variance."

Every prior F3/F4 experiment ran in ARM A: encoders see all 2500 panel genes and
the decoder predicts those same genes. There the latent nearly reconstructs its
own input (deterministic recon Pearson ~0.74), so a referee can say F4's mechanism
was shown only where it is trivially true. ARM B is the honest test: encoders see
ONLY G_in (1250 genes); the decoder predicts G_out (1250 genes no encoder saw).

CRITICAL anti-leak: the library-size offset is derived from G_in counts only
(``gin_library_size``); using G_out's own depth would leak the prediction target.

Reuses byte-for-byte:
  * ``build_nb_variants`` (phase2c run_nb_arm) -- the exact NB decoder + gate +
    log1p-space controls, fed Arm-B targets (G_out) and the G_in offset.
  * ``aggregate_over_groups`` (src.evaluation.distributional) -- energy_distance
    and the EXPRESSION-RESTRICTED |log2 r_var| + signed direction stats.
  * ``clustered_inference`` (phase2f) -- 50-dataset clustered Wilcoxon/sign +
    Wilcoxon-signed-rank sign-flip permutation.
  * ``mean_acceptance_gate`` (src.evaluation.acceptance) -- one-sided underfit gate.

Experiments (mirroring phase2f so the two arms are directly comparable):
  ARM-A SANITY -- feed build_nb_variants Arm-A inputs (full 2500-gene targets,
    full-panel offset) for pca_512+scgpt_512 x 3 seeds and reproduce nb_arm.json's
    unrestricted naive tests to 1e-6. Proves ONLY the geometry changed.
  E6' -- the dispersion contrast (nb_stochastic vs poisson_on_nb_mean) on ALL FOUR
    encoders, gate-independent, 3 seeds, on energy_distance AND expression-
    restricted |log2 r_var|; plus the signed expressed r_var direction; plus the
    mean-fit-gap/energy monotonicity (exact 4!-permutation Spearman p).
  E7' -- dataset-clustered inference (50 datasets), naive reported alongside.
  E8' -- energy-estimator stability: >=10 count redraws at fixed seed-44 mu/theta.

"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2F = HERE.parent / "phase2f_f4_resolution_2026-07-09"
PHASE2C = HERE.parent / "phase2c_robustness_2026-07-09"
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
for p in (PHASE2F, PHASE2C, PHASE2B, PHASE2A, HERE):
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
from phase2g_lib import (  # noqa: E402
    assert_log1p_scale, build_armB_encoders, exact_spearman_perm_p, gin_library_size,
)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
ENCODERS = ["pca_512", "randproj_512", "scgpt_512", "geneformer"]
METRICS = ["energy_distance", "r_var_abs_log2_expr"]      # the two valid discriminators
CONTROL = "poisson_on_nb_mean"
# the five spec variants to build/aggregate (dispersion contrast needs the first two)
VARIANTS = ["nb_stochastic", "poisson_on_nb_mean", "deterministic",
            "matched_noise", "matched_noise_per_type"]
NAIVE_REPRO_METRICS = ["energy_distance", "r_var_abs_log2"]  # unrestricted, for Arm-A repro
N_PERM = 10000
E8_SEED, E8_ENCODERS, E8_DRAWS = 44, ["pca_512", "scgpt_512"], 10
NB_ARM_JSON = PHASE2C / "results" / "nb_arm.json"
REPRO_TOL = 1e-6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def median_over_groups(amap, key):
    vals = np.array([g[key] for g in amap.values() if np.isfinite(g[key])], dtype=np.float64)
    return float(np.median(vals)) if vals.size else float("nan")


def fit_nb_mean_theta(Z, tr, va, te, nva, raw_tgt, lib, mean_lib, seed):
    """Refit the Arm-B NB decoder (target=raw G_out, offset=G_in depth) and return
    the fitted per-cell count mean ``mu_te`` and per-(cell,gene) theta on TEST
    cells, for E8's fixed-(mu,theta) count redraws."""
    Ztest_arg = np.vstack([Z[va], Z[te]])
    lib_arg = np.concatenate([lib[va], lib[te]]).astype(np.float32)
    mu_all, theta_all, _, _, _ = train_nb_decoder(
        Z[tr], raw_tgt[tr], lib[tr], Z[va], raw_tgt[va], lib[va], Ztest_arg, lib_arg,
        mean_libsize=mean_lib, seed=seed, device=DEVICE,
        logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
        warmup_epochs=15, logtheta_lr_scale=0.1)
    return mu_all[nva:], theta_all[nva:]


def e8_count_redraw_stability(mu_te, theta_te, real_log_te, lab_te, ds_te, n_draws):
    """E8: hold NB's fitted (mu, theta) FIXED and redraw nb_stochastic (NB counts)
    and poisson_on_nb_mean (Poisson(mu)) ``n_draws`` times, recomputing the paired
    energy test each draw. The <=300-cell energy subsample is inert (max test
    group < 300), so the only stochasticity is the count sample -- the true source
    of the ~7x cross-implementation p swing the audit flagged."""
    mu_c = np.clip(mu_te, 0.0, None)
    draws = []
    for k in range(n_draws):
        nb_counts = sample_nb(mu_te, theta_te, seed=(1000 + k))
        pois_counts = np.random.default_rng((2000, k)).poisson(mu_c).astype(np.float32)
        dec = {"nb_stochastic": np.log1p(nb_counts),
               "poisson_on_nb_mean": np.log1p(pois_counts)}
        amap = {v: per_group_map(aggregate_over_groups(
            real_log_te, d, lab_te, ds_te, min_cells=20, energy_subsample=300,
            top_k=200, seed=0)) for v, d in dec.items()}
        recs = paired_group_diffs(amap["nb_stochastic"], amap["poisson_on_nb_mean"],
                                  "energy_distance")
        naive = naive_paired(recs)
        clust = clustered_dataset_medians(recs)
        draws.append({"draw": k, "naive_p": naive["p_value"],
                      "nb_win_fraction": naive["nb_win_fraction"],
                      "median_paired_diff": naive["median_paired_diff"],
                      "clustered_wilcoxon_p": clust["wilcoxon_p"],
                      "clustered_sign_p": clust["sign_test_p"]})
    ps = np.array([x["naive_p"] for x in draws])
    wins = np.array([x["nb_win_fraction"] for x in draws])
    clps = np.array([x["clustered_wilcoxon_p"] for x in draws])
    return {
        "n_draws": int(n_draws),
        "naive_p_min": float(ps.min()), "naive_p_max": float(ps.max()),
        "naive_p_median": float(np.median(ps)),
        "naive_p_range_orders_of_magnitude": float(np.log10(ps.max() / max(ps.min(), 1e-300))),
        "naive_p_any_above_0.05": bool((ps > 0.05).any()),
        "naive_p_all_below_0.05": bool((ps < 0.05).all()),
        "nb_win_fraction_min": float(wins.min()), "nb_win_fraction_max": float(wins.max()),
        "nb_win_fraction_median": float(np.median(wins)),
        "clustered_wilcoxon_p_min": float(clps.min()),
        "clustered_wilcoxon_p_max": float(clps.max()),
        "clustered_wilcoxon_p_median": float(np.median(clps)),
        "draws": draws,
    }


def armA_reproduction_sanity(logX, raw, tr, va, te, nva, lab_va, lab_te, ds_te,
                             scgpt_A):
    """Feed build_nb_variants ARM-A inputs (full 2500-gene targets, full-panel
    library offset) and reproduce nb_arm.json's unrestricted naive tests to
    REPRO_TOL. This proves the reused machinery is byte-for-byte intact, so any
    Arm-B difference is due ONLY to the changed geometry (G_in->G_out)."""
    lib_full = raw.sum(axis=1).astype(np.float32)
    mean_lib_full = float(lib_full[tr].mean())
    encA = {"pca_512": PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)}
    if scgpt_A is not None:
        encA["scgpt_512"] = scgpt_A
    nb_arm = json.loads(NB_ARM_JSON.read_text())
    sanity = {"tol": REPRO_TOL, "checks": [], "all_passed": True}
    for name, Z in encA.items():
        for seed in SEEDS:
            decoded_log, _ = build_nb_variants(
                Z, tr, va, te, nva, logX, raw, lib_full, mean_lib_full,
                lab_va, lab_te, seed)
            amap = {v: per_group_map(aggregate_over_groups(
                logX[te], decoded_log[v], lab_te, ds_te, min_cells=20,
                energy_subsample=300, top_k=200, seed=0))
                for v in ("nb_stochastic", CONTROL)}
            for metric in NAIVE_REPRO_METRICS:
                recs = paired_group_diffs(amap["nb_stochastic"], amap[CONTROL], metric)
                got = naive_paired(recs)
                fam = f"{metric}__nb_stochastic_vs_{CONTROL}"
                ref = next(r for r in nb_arm["paired_tests"][str(seed)][fam]["per_encoder"]
                           if r["encoder"] == name)
                dp = abs(got["p_value"] - ref["p_value"])
                dmed = abs(got["median_paired_diff"] - ref["median_paired_diff"])
                dwin = abs(got["nb_win_fraction"] - ref["learned_win_fraction"])
                p_ok = dp <= max(REPRO_TOL, REPRO_TOL * abs(ref["p_value"]))
                ok = bool(p_ok and dmed <= REPRO_TOL and dwin <= REPRO_TOL
                          and got["n_groups"] == ref["n_pairs"])
                sanity["checks"].append({
                    "encoder": name, "seed": seed, "metric": metric,
                    "d_p": dp, "d_median": dmed, "d_win": dwin, "passed": ok})
                sanity["all_passed"] &= ok
                log(f"  [ARM-A SANITY] {name:10s} s{seed} {metric:18s} "
                    f"dp={dp:.2e} dmed={dmed:.2e} dwin={dwin:.2e} "
                    f"{'PASS' if ok else 'FAIL'}")
    failed = [c for c in sanity["checks"] if not c["passed"]]
    assert not failed, (f"ARM-A reproduction FAILED for {failed}: the reused "
                        f"machinery changed something other than the geometry. "
                        f"Refusing to trust Arm-B results.")
    return sanity


def main() -> None:
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())

    panel = json.loads((CACHE / "panel.json").read_text())
    gin_idx = np.asarray(panel["gin_idx"], dtype=int)
    gout_idx = np.asarray(panel["gout_idx"], dtype=int)

    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    assert np.allclose(np.log1p(raw[:2000]), logX[:2000], atol=1e-5), \
        "logX must equal log1p(raw)"

    # ---- ARM-B target: G_out counts. real reference = log1p(raw[:, gout]).
    raw_gout = raw[:, gout_idx].astype(np.float32)
    logX_gout = logX[:, gout_idx].astype(np.float32)
    assert np.allclose(np.log1p(raw_gout[:2000]), logX_gout[:2000], atol=1e-5), \
        "logX_gout must equal log1p(raw_gout)"

    # ---- CRITICAL: the library-size offset is derived from G_in counts ONLY.
    # (asserts G_in/G_out disjoint AND offset != G_out depth -- no target leak.)
    lib = gin_library_size(raw, gin_idx, gout_idx)
    mean_lib = float(lib[tr].mean())
    lib_gout_depth = raw_gout.sum(axis=1)
    leak_corr = float(np.corrcoef(lib, lib_gout_depth)[0, 1])
    log(f"ARM B: target=G_out counts ({raw_gout.shape[1]} genes NO encoder saw). "
        f"offset=G_in depth ONLY (mean_lib={mean_lib:.1f}); "
        f"corr(G_in depth, G_out depth)={leak_corr:.3f} (informative but not the "
        f"target; the decoder never sees G_out counts).")

    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    real_log_te = logX_gout[te]
    n_datasets = int(len(np.unique(ds_te)))
    log(f"raw_gout={raw_gout.shape} test={int(te.sum())} n_datasets={n_datasets}")

    # ---- ARM-A reproduction sanity (proves only the geometry changed) ---------
    scgpt_A = None
    pA = CACHE / "enc_A_scgpt.npy"
    if pA.exists():
        zA = np.load(pA)
        if zA.shape[0] == logX.shape[0]:
            scgpt_A = zA.astype(np.float32)
    log("=== ARM-A reproduction sanity (feed Arm-A inputs to the reused machinery) ===")
    sanity = armA_reproduction_sanity(logX, raw, tr, va, te, nva,
                                      lab_va, lab_te, ds_te, scgpt_A)
    log(f"  [ARM-A SANITY] reproduced nb_arm.json to <= {REPRO_TOL} on all "
        f"{len(sanity['checks'])} unrestricted naive checks -> machinery intact.")

    # ---- Arm-B encoders (all G_in-restricted) ---------------------------------
    enc = build_armB_encoders(logX, gin_idx, tr, CACHE, pca_dim=512, seed=0)
    encoders = [e for e in ENCODERS if e in enc]
    for name in encoders:
        log(f"  Arm-B encoder {name}: dim={enc[name].shape[1]}")
    log(f"encoders available: {encoders}; seeds={SEEDS}")

    # ---- Build the variants for every (encoder, seed) and aggregate -----------
    aggmaps, gates = {}, {}
    n_groups = None
    log1p_scale_checks = {}
    for name in encoders:
        Z = enc[name]
        for seed in SEEDS:
            log(f"=== build {name} seed {seed} (Arm B: Z=G_in -> raw G_out) ===")
            decoded_log, gate = build_nb_variants(
                Z, tr, va, te, nva, logX_gout, raw_gout, lib, mean_lib,
                lab_va, lab_te, seed)
            # sanity: matched_noise / matched_noise_per_type must be log1p-scale.
            for cname in ("matched_noise", "matched_noise_per_type"):
                chk = assert_log1p_scale(decoded_log[cname], real_log_te, cname)
                log1p_scale_checks.setdefault((name, seed), {})[cname] = chk
            amap = {}
            for v in VARIANTS:
                agg = aggregate_over_groups(
                    real_log_te, decoded_log[v], lab_te, ds_te,
                    min_cells=20, energy_subsample=300, top_k=200, seed=0)
                amap[v] = per_group_map(agg)
                if n_groups is None:
                    n_groups = agg["n_groups"]
            aggmaps[(name, seed)] = amap
            gates[(name, seed)] = gate
            log(f"    gate={'PASS' if gate['gate_passed'] else 'FAIL'} "
                f"gap={gate['gap_vs_det']:+.4f} det_r={gate['det_log_pearson']:.4f} "
                f"nb_r={gate['nb_mean_log_pearson']:.4f}")

    # ============================= E6': four-encoder dispersion + gate table ====
    log("=== E6': four-encoder dispersion contrast (Arm B) ===")
    e6 = {"per_encoder_seed": [], "per_encoder_summary": {}, "hypothesis": {},
          "gate_table": []}
    for name in encoders:
        for seed in SEEDS:
            amap = aggmaps[(name, seed)]
            g = gates[(name, seed)]
            e6["gate_table"].append({
                "encoder": name, "seed": seed, "gate_passed": g["gate_passed"],
                "det_log_pearson": g["det_log_pearson"],
                "nb_mean_log_pearson": g["nb_mean_log_pearson"],
                "mean_fit_gap": g["gap_vs_det"]})
            row = {"encoder": name, "seed": seed, "gate_passed": g["gate_passed"],
                   "det_log_pearson": g["det_log_pearson"],
                   "nb_mean_log_pearson": g["nb_mean_log_pearson"],
                   "mean_fit_gap": g["gap_vs_det"]}
            for metric in METRICS:
                recs = paired_group_diffs(amap["nb_stochastic"], amap[CONTROL], metric)
                np_ = naive_paired(recs)
                row[metric] = {
                    "median_paired_diff": np_["median_paired_diff"],  # nb - poisson
                    "nb_win_fraction": np_["nb_win_fraction"],
                    "naive_p": np_["p_value"], "n_groups": np_["n_groups"]}
            # signed expressed r_var DIRECTION (over- vs under-inflation), per variant
            row["signed_direction"] = {}
            for v in ("nb_stochastic", CONTROL):
                row["signed_direction"][v] = {
                    "r_var_signed_expr_median": median_over_groups(amap[v], "r_var_signed_expr"),
                    "r_var_frac_gt1_expr_median": median_over_groups(amap[v], "r_var_frac_gt1_expr"),
                    "r_var_meanlog2_expr_median": median_over_groups(amap[v], "r_var_meanlog2_expr"),
                }
            e6["per_encoder_seed"].append(row)

    for name in encoders:
        rows = [r for r in e6["per_encoder_seed"] if r["encoder"] == name]
        summ = {"gate_pass_seeds": sum(r["gate_passed"] for r in rows),
                "mean_fit_gap": float(np.mean([r["mean_fit_gap"] for r in rows])),
                "nb_mean_log_pearson": float(np.mean([r["nb_mean_log_pearson"] for r in rows])),
                "det_log_pearson": float(np.mean([r["det_log_pearson"] for r in rows]))}
        for metric in METRICS:
            summ[metric] = {
                "mean_median_paired_diff": float(np.mean([r[metric]["median_paired_diff"] for r in rows])),
                "mean_nb_win_fraction": float(np.mean([r[metric]["nb_win_fraction"] for r in rows]))}
        summ["signed_direction"] = {
            v: {k: float(np.mean([r["signed_direction"][v][k] for r in rows]))
                for k in ("r_var_signed_expr_median", "r_var_frac_gt1_expr_median",
                          "r_var_meanlog2_expr_median")}
            for v in ("nb_stochastic", CONTROL)}
        e6["per_encoder_summary"][name] = summ
        log(f"  [{name:12s}] gate={summ['gate_pass_seeds']}/3 gap={summ['mean_fit_gap']:+.4f} "
            f"| energy medDiff(nb-pois)={summ['energy_distance']['mean_median_paired_diff']:+.5f} "
            f"nb_win={summ['energy_distance']['mean_nb_win_fraction']:.3f} "
            f"| rvar_expr medDiff={summ['r_var_abs_log2_expr']['mean_median_paired_diff']:+.5f} "
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
        "energy_effect_monotone_decreasing_in_gap":
            bool(all(np.diff([e_eff[i] for i in order_by_gap]) <= 1e-9)),
    }
    if len(names_ord) >= 3:
        # scipy parametric p (REPORTED WITH A CAVEAT: it is 0.0 / degenerate for
        # |rho|=1) plus the honest EXACT permutation p over all n! orderings.
        hyp["pearson_gap_vs_energy_effect"] = {
            "r": float(pearsonr(gaps, e_eff)[0]), "p_parametric": float(pearsonr(gaps, e_eff)[1])}
        hyp["spearman_gap_vs_energy_effect_scipy"] = {
            "rho": float(spearmanr(gaps, e_eff).statistic),
            "p_scipy_DEGENERATE_for_rho1": float(spearmanr(gaps, e_eff).pvalue)}
        hyp["spearman_gap_vs_energy_effect_exact_perm"] = exact_spearman_perm_p(gaps, e_eff)
        hyp["spearman_gap_vs_rvar_expr_effect_exact_perm"] = exact_spearman_perm_p(gaps, r_eff)
    e6["hypothesis"] = hyp
    ep = hyp.get("spearman_gap_vs_energy_effect_exact_perm", {})
    log(f"  [HYP] gap order {hyp['encoders_sorted_by_gap']}; "
        f"energy effect {[round(x,5) for x in hyp['energy_effect_sorted_by_gap']]}; "
        f"spearman rho={ep.get('rho')} exact_perm p1={ep.get('p_one_sided')} "
        f"p2={ep.get('p_two_sided')}")

    # ============================= E7': dataset-clustered inference =============
    log("=== E7': dataset-clustered inference (50 datasets) ===")
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
                        "p": naive["p_value"], "nb_win_fraction": naive["nb_win_fraction"],
                        "median_paired_diff": naive["median_paired_diff"],
                        "significant_0.05": bool(naive["p_value"] < 0.05),
                        "favors": "NB" if naive["median_paired_diff"] < 0 else "Poisson"},
                    "clustered_50dataset": {
                        "n_datasets": clust["n_datasets"], "wilcoxon_p": clust["wilcoxon_p"],
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
                    "per_group": recs,
                    "dataset_medians": clust["dataset_medians"],
                    "datasets": clust["datasets"],
                }
                e7["per_comparison"].append(comp)
                log(f"  [{name:12s} s{seed} {metric:20s}] naive_p={naive['p_value']:.2e} "
                    f"clustWil_p={clust['wilcoxon_p']:.3f} sign_p={clust['sign_test_p']:.3f} "
                    f"perm_p={perm['p_value']:.4f} nb_win_ds={clust['nb_win_fraction_datasets']:.2f} "
                    f"favors={comp['clustered_50dataset']['favors']} "
                    f"{'[gate FAIL]' if not comp['gate_passed'] else ''}")

    # ============================= E8': energy-estimator stability ==============
    log("=== E8': energy-estimator stability (10 count redraws, fixed seed-44 mu/theta) ===")
    grp_sizes = [g["n_cells"] for name in encoders
                 for g in aggmaps[(name, SEEDS[0])]["nb_stochastic"].values()]
    max_group = int(max(grp_sizes)) if grp_sizes else 0
    e8 = {"seed_decoder": E8_SEED, "n_draws": E8_DRAWS, "encoders": [],
          "max_test_group_cells": max_group, "subsample_inert": bool(max_group <= 300),
          "per_encoder": {}}
    for name in E8_ENCODERS:
        if name not in encoders:
            continue
        e8["encoders"].append(name)
        mu_te, theta_te = fit_nb_mean_theta(
            enc[name], tr, va, te, nva, raw_gout, lib, mean_lib, E8_SEED)
        summ = e8_count_redraw_stability(mu_te, theta_te, real_log_te, lab_te, ds_te, E8_DRAWS)
        e8["per_encoder"][name] = summ
        log(f"  [{name}] naive_p in [{summ['naive_p_min']:.2e}, {summ['naive_p_max']:.2e}] "
            f"({summ['naive_p_range_orders_of_magnitude']:.1f} orders) "
            f"any>0.05={summ['naive_p_any_above_0.05']} "
            f"clustWil_p in [{summ['clustered_wilcoxon_p_min']:.3f},{summ['clustered_wilcoxon_p_max']:.3f}]")

    l1 = {f"{k[0]}__s{k[1]}": v for k, v in log1p_scale_checks.items()}
    out = {
        "purpose": "Phase 2g: ARM-B replication of F4. Encoders see ONLY G_in (1250 "
                   "genes); decoder predicts G_out (1250 unseen genes). Library-size "
                   "offset from G_in depth ONLY (no target leak). Gate-independent "
                   "4-encoder dispersion contrast (E6'), dataset-clustered inference "
                   "(E7'), energy-estimator stability (E8').",
        "arm": "B", "seeds": SEEDS, "encoders": encoders, "n_groups": n_groups,
        "n_datasets": n_datasets, "control": CONTROL, "metrics": METRICS,
        "offset_provenance": {
            "source": "G_in counts only (raw[:, gin_idx].sum(1)), normalised by "
                      "train-split mean",
            "mean_train_libsize_gin": mean_lib,
            "corr_gin_depth_vs_gout_depth": leak_corr,
            "assertion": "gin_library_size() asserts G_in/G_out disjoint AND offset "
                         "!= G_out depth; the decoder never sees G_out counts.",
        },
        "gate_note": ("One-sided gate governs mean/procedure conclusions but NOT the "
                      "dispersion contrast: poisson_on_nb_mean holds NB's OWN fitted "
                      "mean fixed in both arms. Reported for all four encoders."),
        "metric_note": ("r_var_abs_log2_expr = EXPRESSION-RESTRICTED |log2 r_var| "
                        "(mean count >= 0.5). Unrestricted r_var is discredited. Both "
                        "metrics lower-is-better; median_paired_diff = nb_stochastic - "
                        "poisson_on_nb_mean (negative => NB better)."),
        "armA_reproduction_sanity": sanity,
        "log1p_scale_control_checks": l1,
        "E6_four_encoder_dispersion": e6,
        "E7_clustered_inference": e7,
        "E8_energy_stability": e8,
    }
    (RESULTS / "phase2g.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'phase2g.json'} ===")


if __name__ == "__main__":
    main()
