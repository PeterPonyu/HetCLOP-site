"""Phase 2c Task 1 -- SEED SWEEP: does Phase 2b v3's verdict depend on a lucky
seed?

Every decoder result in PHASE2B_V3_REPORT.md is single-seed (train seed 42,
noise seed 123). Here we re-run the two LOAD-BEARING paired comparisons over 5
independent seeds, on the two encoders that BRACKET the result:

  * pca_512   -- a linear encoder, where the learned decoder LOST significantly
    to the label-using matched_noise_per_type oracle on energy_distance.
  * scgpt_512 -- an FM encoder, where it TIED.

For each seed the single integer drives BOTH the decoder init/shuffling (train
seed) AND the noise-control / stochastic-sample draws (noise seed). The two
comparisons re-computed per seed:

  (A) stochastic_betanll vs matched_noise_per_type  on energy_distance
  (B) stochastic_betanll vs matched_noise           on r_var_abs_log2

Per seed we report the learned win-fraction, median paired diff, and Wilcoxon p
(NOT Holm-corrected within a seed: only 1 encoder-comparison per family per seed,
so nothing to correct there; the across-seed stability is the actual multiplicity
we care about, handled by requiring consistent sign across all 5 seeds). Across
seeds we report: sign stability of the median paired diff, significance stability
at alpha=0.05, and mean +/- sd of the win-fraction.

ACCEPTANCE GATE (per seed): the stochastic mu Pearson must be within GATE_TOL
(0.02) of THAT seed's deterministic decoder. Seeds that fail are excluded LOUDLY
and reported, never silently dropped.

Reuses the Phase 2b v2 machinery (encoders, per-type residual std, logvar stats,
acceptance-gate logic) and the shared distributional / decoder modules.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
sys.path.insert(0, str(PHASE2B))  # for run_phase2b_v2 + its sys.path setup

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

# Importing run_phase2b_v2 also fixes sys.path for src.* / decoder_eval.
from run_phase2b_v2 import (  # noqa: E402
    CACHE, DEVICE, GATE_TOL, LOGVAR_MAX, LOGVAR_MIN, MIN_VAL_CELLS_PER_TYPE,
    load_geneformer_armA, logvar_stats, per_type_residual_std,
)
from src.architecture.stochastic_decoder import sample_from, train_stochastic_decoder  # noqa: E402
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
ENCODERS = ["pca_512", "scgpt_512"]
# (label, variant_a, variant_b, metric) -- lower is better for both metrics
COMPARISONS = [
    ("energy_vs_per_type", "stochastic_betanll", "matched_noise_per_type", "energy_distance"),
    ("rvar_vs_matched_noise", "stochastic_betanll", "matched_noise", "r_var_abs_log2"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_stochastic_with_gate_seeded(Ztr, Ytr, Zva, Yva, Zte, det_pearson, raw_te, seed):
    """v2's acceptance-gate logic, but the TRAIN seed is a parameter. Try
    beta=0.5 then beta=1.0; accept the first whose mu Pearson is within GATE_TOL
    of the deterministic decoder; else return the best-mu attempt, gate failed."""
    attempts = []
    chosen = None
    for beta in (0.5, 1.0):
        mu, lv, val, hist, npar = train_stochastic_decoder(
            Ztr, Ytr, Zva, Yva, Zte, seed=seed, device=DEVICE,
            logvar_min=LOGVAR_MIN, logvar_max=LOGVAR_MAX,
            beta=beta, warmup_epochs=15, logvar_lr_scale=0.1)
        mu_pearson = reconstruction_metrics(mu, raw_te)["mean_gene_pearson"]
        gap = det_pearson - mu_pearson
        # one-sided underfit guard: FAIL only if the stochastic mean is WORSE than
        # deterministic by > GATE_TOL (see src/evaluation/acceptance.py).
        passed = mean_acceptance_gate(det_pearson, mu_pearson, GATE_TOL)
        attempts.append({"beta": beta, "mu_pearson": float(mu_pearson),
                         "gap_vs_det": float(gap), "gate_passed": bool(passed),
                         "logvar_stats": logvar_stats(lv)})
        log(f"    [stoch beta={beta} seed={seed}] mu_r={mu_pearson:.4f} "
            f"(det={det_pearson:.4f} gap={gap:+.4f}) gate={'PASS' if passed else 'FAIL'}")
        if passed and chosen is None:
            chosen = (mu, lv, beta)
            break
    if chosen is None:
        best = max(range(len(attempts)), key=lambda i: attempts[i]["mu_pearson"])
        beta = attempts[best]["beta"]
        mu, lv, val, hist, npar = train_stochastic_decoder(
            Ztr, Ytr, Zva, Yva, Zte, seed=seed, device=DEVICE,
            logvar_min=LOGVAR_MIN, logvar_max=LOGVAR_MAX,
            beta=beta, warmup_epochs=15, logvar_lr_scale=0.1)
        chosen = (mu, lv, beta)
        gate_passed = False
    else:
        gate_passed = True
    mu, lv, beta = chosen
    return mu, lv, {"beta_used": beta, "gate_passed": gate_passed, "attempts": attempts}


def build_variants(Z, tr, va, te, nva, target, raw_te, lab_va, lab_te, seed):
    """Deterministic decoder + 2 noise controls + gated stochastic, all at one
    seed. The single ``seed`` drives train init AND every noise draw."""
    Ztr, Yva_t = Z[tr], target[va]
    Ztest_arg = np.vstack([Z[va], Z[te]])
    pred_all, det_val, _, det_np = train_decoder(
        Ztr, target[tr], Z[va], Yva_t, Ztest_arg, seed=seed, device=DEVICE)
    pred_val = pred_all[:nva]
    det_test = pred_all[nva:]
    det_pearson = reconstruction_metrics(det_test, raw_te)["mean_gene_pearson"]

    s_g = (target[va] - pred_val).std(axis=0).astype(np.float32)
    eps = np.random.default_rng(seed).standard_normal(det_test.shape).astype(np.float32)
    ctrl_test = (det_test + s_g * eps).astype(np.float32)

    std_by_type, n_ok, n_fb = per_type_residual_std(
        target[va], pred_val, lab_va, s_g, Z.shape[1])
    std_test = np.vstack([std_by_type[t] for t in lab_te]).astype(np.float32)
    ctrl_pt_test = (det_test + std_test * eps).astype(np.float32)

    mu_test, lv_test, gate = fit_stochastic_with_gate_seeded(
        Ztr, target[tr], Z[va], Yva_t, Z[te], det_pearson, raw_te, seed)
    stoch_test = sample_from(mu_test, lv_test, seed=seed)

    return {
        "deterministic": det_test,
        "stochastic_betanll": stoch_test,
        "matched_noise": ctrl_test,
        "matched_noise_per_type": ctrl_pt_test,
    }, {"det_pearson": float(det_pearson), "gate": gate}


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def paired(a_map, b_map, metric):
    keys = sorted(set(a_map) & set(b_map))
    a, b = [], []
    for k in keys:
        av, bv = a_map[k][metric], b_map[k][metric]
        if np.isfinite(av) and np.isfinite(bv):
            a.append(av)
            b.append(bv)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b  # negative => learned lower => better (both metrics lower=better)
    if np.count_nonzero(diff) >= 1:
        stat, pval = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        stat, pval = float(stat), float(pval)
    else:
        stat, pval = float("nan"), 1.0
    return {
        "n_pairs": int(a.size),
        "median_paired_diff": float(np.median(diff)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "wilcoxon_stat": stat,
        "p_value": pval,
        "learned_win_fraction": float(np.mean(a < b)),
        "significant_0.05": bool(pval < 0.05),
    }


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
    target = logX
    raw_te, lab_te, ds_te = logX[te], labels[te], dataset_ids[te]
    lab_va = labels[va]
    log(f"logX={logX.shape} train={int(tr.sum())} val={nva} test={int(te.sum())}")

    # Encoders (identical construction to v2/v3; fixed across seeds so the sweep
    # isolates decoder+noise seed sensitivity, not encoder randomness).
    enc = {}
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)
    p = CACHE / "enc_A_scgpt.npy"
    if p.exists():
        z = np.load(p)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
    encoders = [e for e in ENCODERS if e in enc]
    log(f"encoders available: {encoders}")

    # per_seed_results[encoder][seed] = {comparison_label: paired_dict}
    per_seed = {name: {} for name in encoders}
    gate_records = {name: {} for name in encoders}
    n_groups = None

    for name in encoders:
        Z = enc[name]
        for seed in SEEDS:
            log(f"=== encoder {name} seed {seed} ===")
            decoded, info = build_variants(
                Z, tr, va, te, nva, target, raw_te, lab_va, lab_te, seed)
            gate_records[name][seed] = {
                "det_pearson": info["det_pearson"],
                "beta_used": info["gate"]["beta_used"],
                "gate_passed": info["gate"]["gate_passed"],
                "attempts": info["gate"]["attempts"],
            }
            aggs = {}
            for vname in ("stochastic_betanll", "matched_noise", "matched_noise_per_type"):
                agg = aggregate_over_groups(
                    raw_te, decoded[vname], lab_te, ds_te,
                    min_cells=20, energy_subsample=300, top_k=200, seed=0)
                aggs[vname] = per_group_map(agg)
                if n_groups is None:
                    n_groups = agg["n_groups"]
            seed_res = {}
            for label, va_name, vb_name, metric in COMPARISONS:
                seed_res[label] = paired(aggs[va_name], aggs[vb_name], metric)
                r = seed_res[label]
                log(f"    [{label}] n={r['n_pairs']} medDiff={r['median_paired_diff']:+.4f} "
                    f"win={r['learned_win_fraction']:.3f} p={r['p_value']:.2e} "
                    f"{'SIG' if r['significant_0.05'] else 'ns'} "
                    f"gate={'PASS' if info['gate']['gate_passed'] else 'FAIL'}")
            per_seed[name][seed] = seed_res

    # --- Across-seed stability -----------------------------------------------
    stability = {}
    for name in encoders:
        stability[name] = {}
        for label, _, _, metric in COMPARISONS:
            valid_seeds = [s for s in SEEDS if gate_records[name][s]["gate_passed"]]
            excluded = [s for s in SEEDS if not gate_records[name][s]["gate_passed"]]
            diffs = [per_seed[name][s][label]["median_paired_diff"] for s in valid_seeds]
            wins = [per_seed[name][s][label]["learned_win_fraction"] for s in valid_seeds]
            sigs = [per_seed[name][s][label]["significant_0.05"] for s in valid_seeds]
            signs = [np.sign(d) for d in diffs]
            stability[name][label] = {
                "metric": metric,
                "n_valid_seeds": len(valid_seeds),
                "excluded_seeds_gate_fail": excluded,
                "median_diff_per_seed": diffs,
                "win_fraction_per_seed": wins,
                "significant_per_seed": [bool(x) for x in sigs],
                "sign_stable": bool(len(set(signs)) == 1) if diffs else False,
                "sign": ("learned_better" if (diffs and signs[0] < 0)
                         else "control_better" if diffs else "n/a"),
                "significance_stable": bool(len(set(sigs)) == 1) if diffs else False,
                "all_significant": bool(all(sigs)) if diffs else False,
                "none_significant": bool(not any(sigs)) if diffs else False,
                "win_fraction_mean": float(np.mean(wins)) if wins else float("nan"),
                "win_fraction_sd": float(np.std(wins, ddof=1)) if len(wins) > 1 else float("nan"),
            }

    out = {
        "purpose": "Phase 2c Task 1 seed sweep: 5 seeds x {pca_512, scgpt_512}, "
                   "re-running the two load-bearing paired comparisons from "
                   "PHASE2B_V3_REPORT.md to test seed dependence of the verdict.",
        "seeds": SEEDS,
        "encoders": encoders,
        "n_groups": n_groups,
        "acceptance_gate": {
            "rule": "stochastic mu Pearson within %.2f of that seed's deterministic "
                    "decoder; failing seeds are excluded from the stability summary "
                    "and reported." % GATE_TOL,
            "tol": GATE_TOL,
        },
        "comparisons": {label: {"a": a, "b": b, "metric": m}
                        for (label, a, b, m) in COMPARISONS},
        "gate_records": gate_records,
        "per_seed": per_seed,
        "across_seed_stability": stability,
    }
    (RESULTS / "seed_sweep.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'seed_sweep.json'} ===")


if __name__ == "__main__":
    main()
