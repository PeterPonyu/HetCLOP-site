"""G5 -- zero-inflated negative binomial (ZINB): does explicit zero-inflation
help, hurt, or do nothing?

One extra likelihood family. A ZINB decoder (NB mean/dispersion heads + a LEARNED
per-gene zero-inflation logit) is trained on raw counts with the SAME library-size
offset and the SAME acceptance gate as the NB arm, then compared to nb_stochastic
and poisson_on_nb_mean on the two valid discriminators (energy_distance and
expression-restricted |log2 r_var|), pca_512, Arm A, 3 seeds, naive + clustered.

Svensson (2020) argues droplet scRNA-seq is NOT zero-inflated once a proper NB is
fit; a NULL here (ZINB indistinguishable from NB, learned pi ~ 0) would corroborate
that and is a perfectly good result. The learned per-gene dropout pi = sigmoid(zi)
is reported so a null is visible as pi ~ 0.

Reuses ``build_nb_variants`` (nb_stochastic, poisson_on_nb_mean, deterministic
gate anchor) and the new ``train_zinb_decoder`` / ``sample_zinb`` /
``zi_logit_stats`` (src.architecture.stochastic_decoder).
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
from sklearn.decomposition import PCA  # noqa: E402

from run_phase2b_v2 import CACHE, DEVICE, GATE_TOL  # noqa: E402
from run_nb_arm import LOGTHETA_MAX, LOGTHETA_MIN, build_nb_variants  # noqa: E402
from src.architecture.stochastic_decoder import (  # noqa: E402
    sample_zinb, train_zinb_decoder, zi_logit_stats,
)
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from decoder_eval import reconstruction_metrics  # noqa: E402
from clustered_inference import (  # noqa: E402
    clustered_dataset_medians, naive_paired, paired_group_diffs,
)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
METRICS = ["energy_distance", "r_var_abs_log2_expr"]
# a>b diff = a - b; both metrics lower-is-better, so a wins a group when diff < 0.
COMPARISONS = [("zinb_stochastic", "nb_stochastic"),
               ("zinb_stochastic", "poisson_on_nb_mean"),
               ("nb_stochastic", "poisson_on_nb_mean")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def paired_view(a_map, b_map, metric):
    recs = paired_group_diffs(a_map, b_map, metric)
    naive = naive_paired(recs)
    clust = clustered_dataset_medians(recs)
    return {
        "n_groups": naive["n_groups"],
        "median_a": float(np.median([r["a"] for r in recs])),
        "median_b": float(np.median([r["b"] for r in recs])),
        "median_paired_diff_a_minus_b": naive["median_paired_diff"],
        "a_win_fraction": float(np.mean([r["diff"] < 0 for r in recs])),
        "naive_p": naive["p_value"],
        "clustered_wilcoxon_p": clust["wilcoxon_p"],
        "clustered_sign_p": clust["sign_test_p"],
        "clustered_median_of_medians": clust["median_of_dataset_medians"],
        "clustered_a_win_fraction_datasets": float(
            np.mean(np.array(clust["dataset_medians"]) < 0)),
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
    real_te = logX[te]
    n_datasets = int(len(np.unique(ds_te)))
    log(f"Arm A: logX={logX.shape} test={int(te.sum())} n_datasets={n_datasets}")

    Z = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)
    log(f"pca_512 Z={Z.shape}")

    per_seed = {}
    for seed in SEEDS:
        log(f"=== ZINB pca_512 seed {seed} ===")
        # NB arm (nb_stochastic, poisson_on_nb_mean) + deterministic gate anchor.
        decoded_log, gate = build_nb_variants(
            Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
        det_pearson = gate["det_log_pearson"]

        # ZINB decoder (same offset/schedule/gate as NB).
        mu_te, theta_te, zi_logit, zinb_val, _, zinb_np = train_zinb_decoder(
            Z[tr], raw[tr], lib[tr], Z[va], raw[va], lib[va], Z[te], lib[te],
            mean_libsize=mean_lib, seed=seed, device=DEVICE,
            logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
            warmup_epochs=15, logtheta_lr_scale=0.1)
        zinb_mean_log = np.log1p(np.clip(mu_te, 0.0, None)).astype(np.float32)
        zinb_r = reconstruction_metrics(zinb_mean_log, logX[te])["mean_gene_pearson"]
        zinb_gate = mean_acceptance_gate(det_pearson, zinb_r, GATE_TOL)
        zi_stats = zi_logit_stats(zi_logit)
        decoded_log["zinb_stochastic"] = np.log1p(
            sample_zinb(mu_te, theta_te, zi_logit, seed=seed)).astype(np.float32)
        log(f"  det_r={det_pearson:.4f} nb_r={gate['nb_mean_log_pearson']:.4f} "
            f"zinb_r={zinb_r:.4f} zinb_gate={'PASS' if zinb_gate else 'FAIL'} "
            f"pi[med={zi_stats['pi_median']:.4f} max={zi_stats['pi_max']:.4f} "
            f"%>0.1={zi_stats['frac_genes_pi_gt_0.1']:.3f}]")

        amap = {v: per_group_map(aggregate_over_groups(
            real_te, decoded_log[v], lab_te, ds_te, min_cells=20,
            energy_subsample=300, top_k=200, seed=0))
            for v in ("zinb_stochastic", "nb_stochastic", "poisson_on_nb_mean")}

        comparisons = {}
        for a, b in COMPARISONS:
            for metric in METRICS:
                comparisons[f"{a}_vs_{b}__{metric}"] = paired_view(
                    amap[a], amap[b], metric)
        per_seed[str(seed)] = {
            "gate": {"det_log_pearson": float(det_pearson),
                     "nb_mean_log_pearson": float(gate["nb_mean_log_pearson"]),
                     "zinb_mean_log_pearson": float(zinb_r),
                     "nb_gate_passed": gate["gate_passed"],
                     "zinb_gate_passed": bool(zinb_gate),
                     "zinb_val_nll": float(zinb_val), "zinb_params": int(zinb_np)},
            "zi_logit_stats": zi_stats,
            "comparisons": comparisons,
        }

    # ---- Cross-seed verdict ----------------------------------------------------
    def summ(a, b, metric):
        key = f"{a}_vs_{b}__{metric}"
        rows = [per_seed[str(s)]["comparisons"][key] for s in SEEDS]
        return {
            "mean_median_paired_diff": float(np.mean([r["median_paired_diff_a_minus_b"] for r in rows])),
            "mean_a_win_fraction": float(np.mean([r["a_win_fraction"] for r in rows])),
            "worst_naive_p": float(max(r["naive_p"] for r in rows)),
            "worst_clustered_wilcoxon_p": float(max(r["clustered_wilcoxon_p"] for r in rows)),
            "clustered_significant_all_seeds_0.05": bool(all(r["clustered_wilcoxon_p"] < 0.05 for r in rows)),
            "a_favored_all_seeds": bool(all(r["median_paired_diff_a_minus_b"] < 0 for r in rows)),
            "b_favored_all_seeds": bool(all(r["median_paired_diff_a_minus_b"] > 0 for r in rows)),
        }

    verdict = {}
    for a, b in COMPARISONS:
        for metric in METRICS:
            verdict[f"{a}_vs_{b}__{metric}"] = summ(a, b, metric)
    pi_meds = [per_seed[str(s)]["zi_logit_stats"]["pi_median"] for s in SEEDS]
    pi_maxs = [per_seed[str(s)]["zi_logit_stats"]["pi_max"] for s in SEEDS]

    # Overall verdict on zero-inflation: does ZINB beat NB anywhere at clustered sig?
    zinb_vs_nb_energy = verdict["zinb_stochastic_vs_nb_stochastic__energy_distance"]
    zinb_vs_nb_rvar = verdict["zinb_stochastic_vs_nb_stochastic__r_var_abs_log2_expr"]
    overall = {
        "learned_pi_median_across_seeds": float(np.mean(pi_meds)),
        "learned_pi_max_across_seeds": float(np.mean(pi_maxs)),
        "zinb_beats_nb_energy_all_seeds": zinb_vs_nb_energy["a_favored_all_seeds"]
        and zinb_vs_nb_energy["clustered_significant_all_seeds_0.05"],
        "zinb_beats_nb_rvar_all_seeds": zinb_vs_nb_rvar["a_favored_all_seeds"]
        and zinb_vs_nb_rvar["clustered_significant_all_seeds_0.05"],
        "zinb_worse_than_nb_energy_all_seeds": zinb_vs_nb_energy["b_favored_all_seeds"]
        and zinb_vs_nb_energy["clustered_significant_all_seeds_0.05"],
        "zinb_worse_than_nb_rvar_all_seeds": zinb_vs_nb_rvar["b_favored_all_seeds"]
        and zinb_vs_nb_rvar["clustered_significant_all_seeds_0.05"],
    }
    log(f"  [VERDICT] learned pi_median~{overall['learned_pi_median_across_seeds']:.4f} "
        f"zinb_beats_nb_energy={overall['zinb_beats_nb_energy_all_seeds']} "
        f"zinb_beats_nb_rvar={overall['zinb_beats_nb_rvar_all_seeds']}")

    out = {
        "purpose": "G5: does zero-inflation help/hurt/nothing? ZINB decoder (NB + "
                   "learned per-gene zero-inflation logit) on raw counts with the "
                   "G_in-derived library offset, same acceptance gate; vs "
                   "nb_stochastic and poisson_on_nb_mean on energy_distance and "
                   "expression-restricted |log2 r_var|, pca_512, Arm A, 3 seeds, "
                   "naive + clustered.",
        "arm": "A", "encoder": "pca_512", "seeds": SEEDS, "metrics": METRICS,
        "n_datasets": n_datasets,
        "comparison_convention": ("diff = a - b (both metrics lower-is-better); a "
                                  "wins a group when diff < 0. a_favored_all_seeds "
                                  "means median paired diff < 0 in every seed."),
        "zero_inflation_note": ("Svensson 2020: droplet scRNA-seq is not zero-"
                                "inflated once a proper NB is fit. A null (ZINB ~ "
                                "NB, learned pi ~ 0) corroborates that."),
        "per_seed": per_seed,
        "verdict": verdict,
        "overall": overall,
    }
    (RESULTS / "g5_zinb.json").write_text(json.dumps(out, indent=2))
    log(f"=== G5 done in {time.time()-t0:.0f}s -> {RESULTS / 'g5_zinb.json'} ===")


if __name__ == "__main__":
    main()
