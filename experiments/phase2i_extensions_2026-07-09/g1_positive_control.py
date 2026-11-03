"""G1 -- THE POSITIVE CONTROL: prove the pipeline CAN separate encoders when a
real information difference exists.

The paper's F2 shows six encoders span only 0.0045 Pearson in decoded output, and
F4/F2 conclude "encoder identity is practically negligible for decoded output." A
referee's obvious objection: the metric is simply INSENSITIVE. This experiment
kills that objection by constructing encoders with a KNOWN, MONOTONE information
gradient and showing reconstruction Pearson AND energy distance separate them.

Gradient 1 (primary): PCA on G_in truncated to k in {2,5,10,25,50,512} components
  -- strictly more information as k grows.
Gradient 2 (corroborating): PCA-512 on a G_in input CORRUPTED by additive
  Gaussian noise (std in {1.0, 2.0} x per-gene train std) -- strictly less
  information as noise grows.

Geometry: ARM B (encoders see ONLY G_in; the decoder predicts G_out, which no
encoder saw; library-size offset from G_in depth ONLY -- gin_library_size). This
is the honest geometry, identical to phase2g.

Decoder + metrics: identical to every prior arm -- ``build_nb_variants`` (the NB
decoder + deterministic baseline + gate + log1p controls) and
``aggregate_over_groups`` (energy_distance, expression-restricted |log2 r_var|),
3 seeds. Reconstruction Pearson = the deterministic decoder's log1p Pearson on
G_out (gate['det_log_pearson']); energy distance = median over groups of the NB
stochastic decode vs real G_out.

Claims established:
  * ORDERING RECOVERED: reconstruction Pearson is monotone-increasing and energy
    distance monotone-decreasing in information (Spearman + EXACT n!-permutation p).
  * SEPARATION: energy distance separates PCA-2 from PCA-512 at dataset-clustered
    significance (Wilcoxon on 50 dataset medians + cluster sign-flip permutation),
    with a MULTI-DRAW count-redraw p-range (standing rule: no single-draw energy p).
  * SMALLEST RESOLVED STEP: the smallest adjacent reconstruction-Pearson gap on
    the PCA ladder that energy still resolves at clustered p<0.05 -- shown to be
    much smaller than the 0.0045 FM-vs-PCA gap, so "encoders indistinguishable" is
    a property of the ENCODERS, not of the metric.

"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2G = HERE.parent / "phase2g_armb_f4_2026-07-09"
PHASE2F = HERE.parent / "phase2f_f4_resolution_2026-07-09"
PHASE2C = HERE.parent / "phase2c_robustness_2026-07-09"
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
for p in (PHASE2G, PHASE2F, PHASE2C, PHASE2B, PHASE2A, HERE):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from run_phase2b_v2 import CACHE, DEVICE  # noqa: E402
from run_nb_arm import LOGTHETA_MAX, LOGTHETA_MIN, build_nb_variants  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, train_nb_decoder  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from clustered_inference import (  # noqa: E402
    clustered_dataset_medians, dataset_signflip_permutation, naive_paired,
    paired_group_diffs,
)
from phase2g_lib import exact_spearman_perm_p, gin_library_size  # noqa: E402
from phase2i_lib import noise_corrupted_pca_encoder, pca_k_encoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
PCA_KS = [2, 5, 10, 25, 50, 512]
NOISE_STDS = [1.0, 2.0]
N_PERM = 10000
REDRAW_K = 8              # count redraws for the headline multi-draw energy p-range
FM_VS_PCA_GAP = 0.0045   # F2 decoded-Pearson span of six encoders (paper)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def energy_median(amap):
    vals = [g["energy_distance"] for g in amap.values() if np.isfinite(g["energy_distance"])]
    return float(np.median(vals)) if vals else float("nan")


def rvar_expr_median(amap):
    vals = [g["r_var_abs_log2_expr"] for g in amap.values()
            if np.isfinite(g["r_var_abs_log2_expr"])]
    return float(np.median(vals)) if vals else float("nan")


def pair_separation(a_map, b_map):
    """Energy-distance separation between two encoders' NB decodes over the shared
    groups (a = lower-info encoder, b = higher-info). Positive median diff => a is
    worse (higher energy) => the metric SEES a as further from real. Reports naive
    314-group, clustered 50-dataset, and cluster sign-flip permutation."""
    recs = paired_group_diffs(a_map, b_map, "energy_distance")
    naive = naive_paired(recs)
    clust = clustered_dataset_medians(recs)
    perm = dataset_signflip_permutation(recs, n_perm=N_PERM, seed=0)
    return {
        "n_groups": naive["n_groups"],
        "naive_p": naive["p_value"],
        "median_paired_diff": naive["median_paired_diff"],  # a - b (energy)
        "a_worse_win_fraction": float(np.mean([r["diff"] > 0 for r in recs])),
        "clustered_wilcoxon_p": clust["wilcoxon_p"],
        "clustered_sign_p": clust["sign_test_p"],
        "clustered_median_of_medians": clust["median_of_dataset_medians"],
        "n_datasets": clust["n_datasets"],
        "perm_p": perm["p_value"],
    }


def main() -> None:
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())

    panel = json.loads((CACHE / "panel.json").read_text())
    gin_idx = np.asarray(panel["gin_idx"], dtype=int)
    gout_idx = np.asarray(panel["gout_idx"], dtype=int)

    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    Xin = logX[:, gin_idx].astype(np.float32)
    raw_gout = raw[:, gout_idx].astype(np.float32)
    logX_gout = logX[:, gout_idx].astype(np.float32)
    lib = gin_library_size(raw, gin_idx, gout_idx)      # G_in depth ONLY (no leak)
    mean_lib = float(lib[tr].mean())
    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    real_log_te = logX_gout[te]
    n_datasets = int(len(np.unique(ds_te)))
    log(f"ARM B: Xin={Xin.shape} target G_out={raw_gout.shape} mean_lib_gin={mean_lib:.1f} "
        f"test={int(te.sum())} n_datasets={n_datasets}")

    # ---- Encoder ladder (built once; deterministic given seeds in the recipe) --
    enc = {}
    for k in PCA_KS:
        enc[f"pca_{k}"] = pca_k_encoder(Xin, tr, k)
    for s in NOISE_STDS:
        enc[f"pca512_noise{s}"] = noise_corrupted_pca_encoder(Xin, tr, 512, s, seed=0)
    names = list(enc.keys())
    for nm in names:
        log(f"  encoder {nm}: dim={enc[nm].shape[1]}")

    # ---- Train decoder for every (encoder, seed); collect recon + NB decode -----
    recon = {}      # (name, seed) -> gate dict
    nbmap = {}      # (name, seed) -> per_group_map of nb_stochastic
    mn_energy = {}  # (name, seed) -> matched_noise energy median
    for nm in names:
        Z = enc[nm]
        for seed in SEEDS:
            log(f"=== build {nm} seed {seed} (Arm B G_in->G_out) ===")
            decoded_log, gate = build_nb_variants(
                Z, tr, va, te, nva, logX_gout, raw_gout, lib, mean_lib,
                lab_va, lab_te, seed)
            nb_agg = aggregate_over_groups(real_log_te, decoded_log["nb_stochastic"],
                                           lab_te, ds_te, min_cells=20,
                                           energy_subsample=300, top_k=200, seed=0)
            mn_agg = aggregate_over_groups(real_log_te, decoded_log["matched_noise"],
                                           lab_te, ds_te, min_cells=20,
                                           energy_subsample=300, top_k=200, seed=0)
            recon[(nm, seed)] = gate
            nbmap[(nm, seed)] = per_group_map(nb_agg)
            mn_energy[(nm, seed)] = mn_agg["energy_distance"]["median"]
            log(f"    det_log_r={gate['det_log_pearson']:.4f} nb_mean_log_r="
                f"{gate['nb_mean_log_pearson']:.4f} gate="
                f"{'PASS' if gate['gate_passed'] else 'FAIL'} "
                f"nb_energy={nb_agg['energy_distance']['median']:.4f} "
                f"rvar_expr={rvar_expr_median(nbmap[(nm, seed)]):.4f}")

    # ---- Per-encoder summary (mean over seeds) ---------------------------------
    per_encoder = {}
    for nm in names:
        det_r = np.mean([recon[(nm, s)]["det_log_pearson"] for s in SEEDS])
        nb_r = np.mean([recon[(nm, s)]["nb_mean_log_pearson"] for s in SEEDS])
        nb_e = np.mean([energy_median(nbmap[(nm, s)]) for s in SEEDS])
        nb_rv = np.mean([rvar_expr_median(nbmap[(nm, s)]) for s in SEEDS])
        mn_e = np.mean([mn_energy[(nm, s)] for s in SEEDS])
        per_encoder[nm] = {
            "dim": int(enc[nm].shape[1]),
            "det_recon_pearson_mean": float(det_r),
            "nb_mean_recon_pearson_mean": float(nb_r),
            "nb_energy_median_mean": float(nb_e),
            "nb_rvar_abs_log2_expr_median_mean": float(nb_rv),
            "matched_noise_energy_median_mean": float(mn_e),
        }
        log(f"  [{nm:16s}] det_r={det_r:.4f} nb_r={nb_r:.4f} "
            f"nb_energy={nb_e:.4f} mn_energy={mn_e:.4f} nb_rvar_expr={nb_rv:.4f}")

    # ---- ORDERING RECOVERED on the PCA ladder (monotone in k) ------------------
    pca_names = [f"pca_{k}" for k in PCA_KS]
    ks = np.array(PCA_KS, dtype=np.float64)
    det_ladder = np.array([per_encoder[n]["det_recon_pearson_mean"] for n in pca_names])
    nb_e_ladder = np.array([per_encoder[n]["nb_energy_median_mean"] for n in pca_names])
    ordering = {
        "pca_ks": PCA_KS,
        "det_recon_pearson": det_ladder.tolist(),
        "nb_energy_median": nb_e_ladder.tolist(),
        # information (log k) vs reconstruction: expect strong POSITIVE monotone
        "spearman_logk_vs_recon": exact_spearman_perm_p(np.log(ks), det_ladder),
        "pearson_logk_vs_recon_r": float(pearsonr(np.log(ks), det_ladder)[0]),
        # information (log k) vs energy: expect strong NEGATIVE monotone
        "spearman_logk_vs_energy": exact_spearman_perm_p(np.log(ks), nb_e_ladder),
        "pearson_logk_vs_energy_r": float(pearsonr(np.log(ks), nb_e_ladder)[0]),
        "recon_monotone_increasing": bool(np.all(np.diff(det_ladder) >= -1e-6)),
        "energy_monotone_decreasing": bool(np.all(np.diff(nb_e_ladder) <= 1e-6)),
    }
    log(f"  [ORDERING] recon spearman rho={ordering['spearman_logk_vs_recon']['rho']:.3f} "
        f"perm_p2={ordering['spearman_logk_vs_recon']['p_two_sided']:.4f}; "
        f"energy spearman rho={ordering['spearman_logk_vs_energy']['rho']:.3f} "
        f"perm_p2={ordering['spearman_logk_vs_energy']['p_two_sided']:.4f}")

    # ---- Noise gradient corroboration ------------------------------------------
    noise_names = ["pca_512"] + [f"pca512_noise{s}" for s in NOISE_STDS]
    noise_levels = [0.0] + NOISE_STDS
    noise_grad = {
        "noise_std": noise_levels,
        "det_recon_pearson": [per_encoder[n]["det_recon_pearson_mean"] for n in noise_names],
        "nb_energy_median": [per_encoder[n]["nb_energy_median_mean"] for n in noise_names],
    }
    log(f"  [NOISE] recon {['%.4f' % v for v in noise_grad['det_recon_pearson']]} "
        f"energy {['%.4f' % v for v in noise_grad['nb_energy_median']]}")

    # ---- SEPARATION: adjacent-pair clustered tests along the PCA ladder --------
    # Sort encoders by det recon (ascending) so 'a' is always the lower-info side.
    order = [n for _, n in sorted(zip(det_ladder, pca_names))]
    log(f"  [SEP] recon-sorted PCA ladder: {order}")
    adjacent = []
    for a, b in zip(order[:-1], order[1:]):
        gap = per_encoder[b]["det_recon_pearson_mean"] - per_encoder[a]["det_recon_pearson_mean"]
        per_seed = [pair_separation(nbmap[(a, s)], nbmap[(b, s)]) for s in SEEDS]
        worst_clust = max(r["clustered_wilcoxon_p"] for r in per_seed)
        worst_perm = max(r["perm_p"] for r in per_seed)
        adjacent.append({
            "a": a, "b": b, "recon_gap": float(gap),
            "per_seed": per_seed,
            "worst_clustered_wilcoxon_p": float(worst_clust),
            "worst_perm_p": float(worst_perm),
            "resolved_clustered_0.05": bool(worst_clust < 0.05),
        })
        log(f"    {a:8s}->{b:8s} recon_gap={gap:+.4f} "
            f"worst_clustWil_p={worst_clust:.4f} worst_perm_p={worst_perm:.4f} "
            f"{'RESOLVED' if worst_clust < 0.05 else 'ns'}")

    resolved = [a for a in adjacent if a["resolved_clustered_0.05"]]
    smallest = min(resolved, key=lambda a: a["recon_gap"]) if resolved else None
    smallest_step = {
        "smallest_resolved_recon_gap": (smallest["recon_gap"] if smallest else None),
        "smallest_resolved_pair": ([smallest["a"], smallest["b"]] if smallest else None),
        "fm_vs_pca_gap_reference": FM_VS_PCA_GAP,
        "smaller_than_fm_vs_pca_gap": (
            bool(smallest["recon_gap"] < FM_VS_PCA_GAP) if smallest else None),
        "n_adjacent_resolved": len(resolved),
        "n_adjacent_total": len(adjacent),
    }
    log(f"  [SMALLEST STEP] {smallest_step}")

    # ---- HEADLINE PCA-2 vs PCA-512 with a MULTI-DRAW energy p-range -------------
    # Standing rule: any energy claim must report a multi-draw p-range, not a
    # single count draw. Hold each encoder's fitted (mu, theta) FIXED (seed 42) and
    # redraw NB counts REDRAW_K times, recomputing the clustered/naive p each draw.
    log("  [HEADLINE] PCA-2 vs PCA-512, multi-draw energy p-range (seed-42 mu/theta)")
    lib_arg = np.concatenate([lib[va], lib[te]]).astype(np.float32)

    def fit_mu_theta(nm):
        Z = enc[nm]
        Ztest_arg = np.vstack([Z[va], Z[te]])
        mu_all, th_all, _, _, _ = train_nb_decoder(
            Z[tr], raw_gout[tr], lib[tr], Z[va], raw_gout[va], lib[va], Ztest_arg, lib_arg,
            mean_libsize=mean_lib, seed=42, device=DEVICE,
            logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
            warmup_epochs=15, logtheta_lr_scale=0.1)
        return mu_all[nva:], th_all[nva:]

    mu2, th2 = fit_mu_theta("pca_2")
    mu512, th512 = fit_mu_theta("pca_512")
    draws = []
    for kdraw in range(REDRAW_K):
        c2 = sample_nb(mu2, th2, seed=1000 + kdraw)
        c512 = sample_nb(mu512, th512, seed=2000 + kdraw)
        m2 = per_group_map(aggregate_over_groups(
            real_log_te, np.log1p(c2), lab_te, ds_te, min_cells=20,
            energy_subsample=300, top_k=200, seed=0))
        m512 = per_group_map(aggregate_over_groups(
            real_log_te, np.log1p(c512), lab_te, ds_te, min_cells=20,
            energy_subsample=300, top_k=200, seed=0))
        sep = pair_separation(m2, m512)  # a=pca_2 (worse), b=pca_512
        draws.append(sep)
    naive_ps = np.array([d["naive_p"] for d in draws])
    clust_ps = np.array([d["clustered_wilcoxon_p"] for d in draws])
    perm_ps = np.array([d["perm_p"] for d in draws])
    headline = {
        "pair": ["pca_2", "pca_512"],
        "n_redraws": REDRAW_K,
        "naive_p_range": [float(naive_ps.min()), float(naive_ps.max())],
        "clustered_wilcoxon_p_range": [float(clust_ps.min()), float(clust_ps.max())],
        "clustered_wilcoxon_p_median": float(np.median(clust_ps)),
        "perm_p_range": [float(perm_ps.min()), float(perm_ps.max())],
        "median_energy_diff_pca2_minus_pca512": float(np.median(
            [d["median_paired_diff"] for d in draws])),
        "pca2_worse_win_fraction_median": float(np.median(
            [d["a_worse_win_fraction"] for d in draws])),
        "separates_all_draws_clustered_0.05": bool((clust_ps < 0.05).all()),
        "recon_gap_pca2_to_pca512": float(
            per_encoder["pca_512"]["det_recon_pearson_mean"]
            - per_encoder["pca_2"]["det_recon_pearson_mean"]),
    }
    log(f"  [HEADLINE] clustered_wilcoxon_p range "
        f"[{clust_ps.min():.2e},{clust_ps.max():.2e}] "
        f"median_energy_diff={headline['median_energy_diff_pca2_minus_pca512']:+.4f} "
        f"separates_all={headline['separates_all_draws_clustered_0.05']}")

    out = {
        "purpose": "G1 positive control (Arm B): construct a KNOWN monotone "
                   "information gradient (PCA-k on G_in, and PCA-512 on noise-"
                   "corrupted G_in) and show reconstruction Pearson AND energy "
                   "distance separate the encoders -> the pipeline is sensitive to "
                   "real encoder differences, so F2's 'encoders indistinguishable' "
                   "is a property of the encoders, not of the metric.",
        "arm": "B (encoders see G_in only; decoder predicts unseen G_out; offset "
               "= G_in depth only)",
        "seeds": SEEDS, "pca_ks": PCA_KS, "noise_stds": NOISE_STDS,
        "n_datasets": n_datasets, "n_perm": N_PERM, "redraw_k": REDRAW_K,
        "fm_vs_pca_gap_reference": FM_VS_PCA_GAP,
        "metric_note": ("reconstruction Pearson = deterministic decoder log1p "
                        "Pearson on G_out; energy = median over groups of NB "
                        "stochastic decode vs real G_out; rvar = expression-"
                        "restricted |log2 r_var|. Lower energy/rvar = closer to real."),
        "per_encoder": per_encoder,
        "ordering_recovered": ordering,
        "noise_gradient": noise_grad,
        "adjacent_pair_separation": adjacent,
        "smallest_resolved_step": smallest_step,
        "headline_pca2_vs_pca512_multidraw": headline,
    }
    (RESULTS / "g1_positive_control.json").write_text(json.dumps(out, indent=2))
    log(f"=== G1 done in {time.time()-t0:.0f}s -> {RESULTS / 'g1_positive_control.json'} ===")


if __name__ == "__main__":
    main()
