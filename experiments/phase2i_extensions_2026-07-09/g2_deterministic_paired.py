"""G2 -- substantiate (or refute) the paper's unsupported claim that the
deterministic decoder "loses every group".

manuscript/paper.tex:436 asserts, in the trap section, that on energy distance
"the deterministic decoder loses every group and matched noise is not the best
variant." No deterministic-vs-control PAIRED test exists in any results JSON -- the
number was never computed. Worse, in phase2b_v2_results.json the pca_512
deterministic energy MEDIAN is 1.0120 vs matched_noise 1.0122 (a 0.0002 gap), so
the marginal medians already suggest the deterministic decoder does NOT lose to
matched noise there. This runs the actual paired tests to settle it.

Setup: Arm A (encoders see the 2500-gene panel, decoder predicts the same genes),
pca_512 and scgpt_512, 3 seeds. Deterministic vs each of five variants:
  stochastic_betanll, matched_noise, matched_noise_per_type  (Gaussian arm), and
  nb_stochastic, poisson_on_nb_mean                          (NB arm),
on energy_distance and the expression-restricted |log2 r_var|. 314 groups, paired
Wilcoxon with Holm across the two encoders, PLUS the dataset-clustered inference.

Win fraction convention: paired diff = deterministic - variant (both metrics
lower-is-better), so the deterministic decoder LOSES a group when its metric is
larger (diff > 0). "loses every group" is literally true iff that fraction is
1.000. Exact per-encoder medians are reported so the 1.0120-vs-1.0122 case is
visible in the numbers.

Reuses ``build_nb_variants`` (deterministic + matched_noise[/per_type] +
nb_stochastic + poisson_on_nb_mean, all in log1p space) and the phase2b
beta-NLL decoder (train_stochastic_decoder + sample_from) for stochastic_betanll,
gated identically.
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

from run_phase2b_v2 import (  # noqa: E402
    CACHE, DEVICE, GATE_TOL, LOGVAR_MAX, LOGVAR_MIN,
)
from run_nb_arm import build_nb_variants, holm  # noqa: E402
from src.architecture.stochastic_decoder import (  # noqa: E402
    sample_from, train_stochastic_decoder,
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
ENCODERS = ["pca_512", "scgpt_512"]
METRICS = ["energy_distance", "r_var_abs_log2_expr"]
VARIANTS = ["stochastic_betanll", "matched_noise", "matched_noise_per_type",
            "nb_stochastic", "poisson_on_nb_mean"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def fit_betanll(Ztr, Ytr, Zva, Yva, Zte, det_pearson, real_te, seed):
    """Train the beta-NLL heteroscedastic decoder at a given SEED (phase2b's
    fit_stochastic_with_gate hardcodes seed=42, so it cannot be reused across
    seeds; this replicates its gate logic with train_stochastic_decoder +
    mean_acceptance_gate, both imported). Try beta=0.5 then 1.0; return the first
    that passes the one-sided underfit gate, else the higher-mu-Pearson attempt.
    Returns (mu, logvar, meta)."""
    attempts = []
    best = None
    for beta in (0.5, 1.0):
        mu, lv, val, _, npar = train_stochastic_decoder(
            Ztr, Ytr, Zva, Yva, Zte, seed=seed, device=DEVICE,
            logvar_min=LOGVAR_MIN, logvar_max=LOGVAR_MAX,
            beta=beta, warmup_epochs=15, logvar_lr_scale=0.1)
        mu_pearson = reconstruction_metrics(mu, real_te)["mean_gene_pearson"]
        passed = mean_acceptance_gate(det_pearson, mu_pearson, GATE_TOL)
        attempts.append({"beta": beta, "mu_pearson": float(mu_pearson),
                         "gap_vs_det": float(det_pearson - mu_pearson),
                         "gate_passed": bool(passed)})
        if best is None or mu_pearson > best[2]:
            best = (mu, lv, mu_pearson, beta)
        if passed:
            return mu, lv, {"beta_used": beta, "gate_passed": True, "attempts": attempts}
    return best[0], best[1], {"beta_used": best[3], "gate_passed": False, "attempts": attempts}


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

    enc = {"pca_512": PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)}
    pA = CACHE / "enc_A_scgpt.npy"
    if pA.exists():
        z = np.load(pA)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
    encoders = [e for e in ENCODERS if e in enc]
    log(f"encoders: {encoders}")

    # amap[(encoder, seed)] = {variant: per_group_map}; medians[(enc,seed,variant,metric)]
    amap = {}
    energy_medians = {}   # (enc, seed, variant) -> energy median (exact numbers)
    gate_info = {}
    for name in encoders:
        Z = enc[name]
        for seed in SEEDS:
            log(f"=== build {name} seed {seed} ===")
            decoded_log, gate = build_nb_variants(
                Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
            # beta-NLL stochastic decoder (Gaussian arm), gated to the SAME det.
            det_pearson = gate["det_log_pearson"]
            mu, lv, sgate = fit_betanll(
                Z[tr], logX[tr], Z[va], logX[va], Z[te], det_pearson, real_te, seed)
            decoded_log["stochastic_betanll"] = sample_from(mu, lv, seed=seed)
            gate_info[(name, seed)] = {
                "det_log_pearson": det_pearson,
                "nb_gate_passed": gate["gate_passed"],
                "betanll_gate_passed": sgate["gate_passed"],
                "betanll_beta_used": sgate["beta_used"]}
            vmap = {}
            for v in ["deterministic"] + VARIANTS:
                agg = aggregate_over_groups(real_te, decoded_log[v], lab_te, ds_te,
                                            min_cells=20, energy_subsample=300,
                                            top_k=200, seed=0)
                vmap[v] = per_group_map(agg)
                energy_medians[(name, seed, v)] = agg["energy_distance"]["median"]
            amap[(name, seed)] = vmap
            log(f"    det_energy={energy_medians[(name, seed, 'deterministic')]:.4f} "
                f"mn_energy={energy_medians[(name, seed, 'matched_noise')]:.4f} "
                f"betanll_gate={'PASS' if sgate['gate_passed'] else 'FAIL'}")

    # ---- Paired tests: deterministic vs each variant, Holm across encoders -----
    results = []
    loses_every_group = {}   # (variant, metric) -> bool over ALL encoder-seeds
    for metric in METRICS:
        for variant in VARIANTS:
            for seed in SEEDS:
                per_enc, raw_p = [], []
                for name in encoders:
                    recs = paired_group_diffs(amap[(name, seed)]["deterministic"],
                                              amap[(name, seed)][variant], metric)
                    naive = naive_paired(recs)
                    clust = clustered_dataset_medians(recs)
                    # deterministic LOSES a group when its metric > variant's (diff>0)
                    det_lose_frac = float(np.mean([r["diff"] > 0 for r in recs]))
                    row = {
                        "metric": metric, "variant": variant, "encoder": name,
                        "seed": seed, "n_groups": naive["n_groups"],
                        "det_median": float(np.median([r["a"] for r in recs])),
                        "variant_median": float(np.median([r["b"] for r in recs])),
                        "median_paired_diff_det_minus_variant": naive["median_paired_diff"],
                        "det_lose_fraction": det_lose_frac,
                        "det_loses_every_group": bool(det_lose_frac >= 1.0 - 1e-12),
                        "naive_p": naive["p_value"],
                        "clustered_wilcoxon_p": clust["wilcoxon_p"],
                        "clustered_sign_p": clust["sign_test_p"],
                        "clustered_median_of_medians": clust["median_of_dataset_medians"],
                    }
                    per_enc.append(row)
                    raw_p.append(naive["p_value"])
                for row, ph in zip(per_enc, holm(raw_p)):
                    row["p_holm"] = float(ph)
                    row["significant_holm_0.05"] = bool(ph < 0.05)
                    results.append(row)
            key = (variant, metric)
            loses_every_group[f"{variant}__{metric}"] = bool(all(
                r["det_loses_every_group"] for r in results
                if r["variant"] == variant and r["metric"] == metric))

    for metric in METRICS:
        for variant in VARIANTS:
            rows = [r for r in results if r["metric"] == metric and r["variant"] == variant]
            fr = [r["det_lose_fraction"] for r in rows]
            log(f"  [{metric:20s} det vs {variant:24s}] det_lose_frac "
                f"min={min(fr):.3f} max={max(fr):.3f} "
                f"loses_every_group_ALL={loses_every_group[f'{variant}__{metric}']}")

    # ---- The specific 1.0120 vs 1.0122 case, exact numbers ---------------------
    pca_energy_detail = {}
    if "pca_512" in encoders:
        for v in ["deterministic"] + VARIANTS:
            pca_energy_detail[v] = {
                str(s): float(energy_medians[("pca_512", s, v)]) for s in SEEDS}

    # Verdict on the exact "det vs matched_noise on energy, pca_512" claim.
    verdict = {}
    if "pca_512" in encoders:
        rows = [r for r in results if r["encoder"] == "pca_512"
                and r["metric"] == "energy_distance" and r["variant"] == "matched_noise"]
        verdict["det_vs_matched_noise_energy_pca512"] = {
            "per_seed_det_lose_fraction": {str(r["seed"]): r["det_lose_fraction"] for r in rows},
            "per_seed_det_median": {str(r["seed"]): r["det_median"] for r in rows},
            "per_seed_matched_noise_median": {str(r["seed"]): r["variant_median"] for r in rows},
            "loses_every_group": bool(all(r["det_loses_every_group"] for r in rows)),
        }

    out = {
        "purpose": "G2: substantiate/refute paper.tex:436 'the deterministic "
                   "decoder loses every group'. Paired deterministic-vs-control "
                   "tests (314 groups, Holm across encoders, + dataset-clustered) "
                   "on energy_distance and expression-restricted |log2 r_var|, "
                   "Arm A, pca_512 + scgpt_512, 3 seeds.",
        "arm": "A", "seeds": SEEDS, "encoders": encoders, "variants": VARIANTS,
        "metrics": METRICS, "n_datasets": n_datasets,
        "win_fraction_convention": ("paired diff = deterministic - variant (both "
                                    "metrics lower-is-better); deterministic LOSES a "
                                    "group when diff > 0; 'loses every group' iff "
                                    "det_lose_fraction == 1.000."),
        "gate_info": {f"{k[0]}__s{k[1]}": v for k, v in gate_info.items()},
        "det_loses_every_group_over_all_encoder_seeds": loses_every_group,
        "pca_512_energy_medians_by_variant": pca_energy_detail,
        "verdict_det_vs_matched_noise_energy_pca512": verdict,
        "per_test": results,
    }
    (RESULTS / "g2_deterministic_paired.json").write_text(json.dumps(out, indent=2))
    log(f"=== G2 done in {time.time()-t0:.0f}s -> {RESULTS / 'g2_deterministic_paired.json'} ===")


if __name__ == "__main__":
    main()
