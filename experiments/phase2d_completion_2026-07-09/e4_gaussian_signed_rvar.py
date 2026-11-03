"""Phase 2d E4 -- SIGNED r_var for the Gaussian-arm variants (cheap fill-in).

Phase 2b/2c stored only |log2 r_var| for the Gaussian (heteroscedastic-Gaussian-
on-log1p) arm, so the DIRECTION of each variant's variance miscalibration
(over- vs under-inflation) was never recorded. E4 fills that gap: for pca_512 and
scgpt_512, Arm A (encoders see all 2500 panel genes; decoder predicts them), it
reports the SIGNED r_var median + IQR over the 314 (dataset x cell type) groups
for four variants:
  * deterministic            (mean-regressed; expected << 1, the retained-variance
                              fraction)
  * stochastic_betanll       (learned heteroscedastic Gaussian sample; gated)
  * matched_noise            (global per-gene residual noise around det mean)
  * matched_noise_per_type   (per-type residual noise around det mean; uses labels)

This completes the picture E1 opens: whether the Gaussian stochastic decoder over-
or under-inflates within-type variance, alongside E1's count-model direction.

Reuses run_seed_sweep.build_variants (deterministic + 2 noise controls + gated
beta-NLL stochastic at one seed) VERBATIM, so the variants are byte-for-byte the
Phase 2c seed-sweep's. metric on log1p; real = logX[te]. energy uses <=300
cells/side. 3 seeds; per-seed signed r_var + across-seed mean/sd of the median.

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
sys.path.insert(0, str(PHASE2C))
sys.path.insert(0, str(PHASE2B))
sys.path.insert(0, str(PHASE2A))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from run_phase2b_v2 import CACHE  # noqa: E402
from run_seed_sweep import build_variants  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
ENCODERS = ["pca_512", "scgpt_512"]
VARIANTS = ["deterministic", "stochastic_betanll", "matched_noise",
            "matched_noise_per_type"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())
    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    target = logX
    raw_te, lab_te, ds_te = logX[te], labels[te], dataset_ids[te]
    lab_va = labels[va]
    log(f"logX={logX.shape} train={int(tr.sum())} val={nva} test={int(te.sum())}")

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

    per_enc = {}
    for name in encoders:
        Z = enc[name]
        # Unrestricted (degenerate on sparse panel) AND expression-restricted
        # signed r_var per variant per seed. The restricted stats reveal the
        # DIRECTION the sparsity-pinned unrestricted median hides.
        by_variant = {v: {
            "unrestr_median_per_seed": [], "unrestr_iqr_per_seed": [],
            "abs_log2_median_per_seed": [],
            "expr_signed_per_seed": [], "expr_frac_gt1_per_seed": [],
            "expr_meanlog2_per_seed": [], "n_expr_per_seed": [],
            "topk_signed_per_seed": [], "topk_frac_gt1_per_seed": [],
        } for v in VARIANTS}
        gate_per_seed = {}
        for seed in SEEDS:
            log(f"=== E4 {name} seed {seed} ===")
            decoded, info = build_variants(
                Z, tr, va, te, nva, target, raw_te, lab_va, lab_te, seed)
            gate_per_seed[str(seed)] = {
                "det_pearson": info["det_pearson"],
                "beta_used": info["gate"]["beta_used"],
                "gate_passed": info["gate"]["gate_passed"],
            }
            for v in VARIANTS:
                agg = aggregate_over_groups(
                    raw_te, decoded[v], lab_te, ds_te,
                    min_cells=20, energy_subsample=300, top_k=200, seed=0)
                bv = by_variant[v]
                bv["unrestr_median_per_seed"].append(float(agg["r_var_signed"]["median"]))
                bv["unrestr_iqr_per_seed"].append(
                    [float(x) for x in agg["r_var_signed"]["iqr"]])
                bv["abs_log2_median_per_seed"].append(float(agg["r_var_abs_log2"]["median"]))
                bv["expr_signed_per_seed"].append(float(agg["r_var_signed_expressed"]["median"]))
                bv["expr_frac_gt1_per_seed"].append(float(agg["r_var_frac_gt1_expressed"]["median"]))
                bv["expr_meanlog2_per_seed"].append(float(agg["r_var_meanlog2_expressed"]["median"]))
                bv["n_expr_per_seed"].append(float(agg["n_expr"]["median"]))
                bv["topk_signed_per_seed"].append(float(agg["r_var_signed_topk"]["median"]))
                bv["topk_frac_gt1_per_seed"].append(float(agg["r_var_frac_gt1_topk"]["median"]))
                log(f"    {v:24s} unrestr_signed={agg['r_var_signed']['median']:.4f} | "
                    f"EXPR signed={agg['r_var_signed_expressed']['median']:.4f} "
                    f"frac>1={agg['r_var_frac_gt1_expressed']['median']:.4f} "
                    f"n={agg['n_expr']['median']:.0f} | "
                    f"topk signed={agg['r_var_signed_topk']['median']:.4f} "
                    f"frac>1={agg['r_var_frac_gt1_topk']['median']:.4f}")
        # summarise across seeds
        for v in VARIANTS:
            bv = by_variant[v]
            for key, src in [("unrestr_median", "unrestr_median_per_seed"),
                             ("expr_signed", "expr_signed_per_seed"),
                             ("expr_frac_gt1", "expr_frac_gt1_per_seed"),
                             ("topk_signed", "topk_signed_per_seed"),
                             ("topk_frac_gt1", "topk_frac_gt1_per_seed")]:
                arr = bv[src]
                bv[f"{key}_mean"] = float(np.mean(arr))
                bv[f"{key}_sd"] = float(np.std(arr, ddof=1))
            bv["direction_expr"] = ("over_inflated(>1)" if bv["expr_frac_gt1_mean"] > 0.5
                                    else "under_inflated(<1)")
        per_enc[name] = {"gate_per_seed": gate_per_seed, "variants": by_variant}
        for v in VARIANTS:
            bv = by_variant[v]
            log(f"  [{name} {v}] unrestr_signed={bv['unrestr_median_mean']:.4f} | "
                f"EXPR signed={bv['expr_signed_mean']:.4f}+/-{bv['expr_signed_sd']:.4f} "
                f"frac>1={bv['expr_frac_gt1_mean']:.4f} ({bv['direction_expr']}) | "
                f"topk signed={bv['topk_signed_mean']:.4f} frac>1={bv['topk_frac_gt1_mean']:.4f}")

    out = {
        "purpose": "Phase 2d E4: SIGNED r_var for the Gaussian-arm variants "
                   "(deterministic, stochastic_betanll, matched_noise, "
                   "matched_noise_per_type) on pca_512 and scgpt_512, Arm A, 3 "
                   "seeds -- recording the DIRECTION of variance miscalibration "
                   "that only |log2 r_var| captured before. Reports BOTH the "
                   "unrestricted signed r_var (degenerate/sparsity-pinned for "
                   "resampled variants) and the expression-restricted direction "
                   "stats (mean-count>=0.5 and top-200-variance: median signed "
                   "ratio + fraction of genes with ratio>1), so the artefact is "
                   "visible rather than hidden.",
        "setting": "Arm A: encoders see 2500 panel genes; decoder predicts them; "
                   "metric on log1p; real = logX[te].",
        "seeds": SEEDS, "encoders": encoders, "variants": VARIANTS,
        "energy_subsample": 300,
        "per_encoder": per_enc,
    }
    (RESULTS / "e4_gaussian_signed_rvar.json").write_text(json.dumps(out, indent=2))
    log(f"=== E4 done in {time.time()-t0:.0f}s -> {RESULTS / 'e4_gaussian_signed_rvar.json'} ===")


if __name__ == "__main__":
    main()
