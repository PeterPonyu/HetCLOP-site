"""Phase 2b v3: PAIRED statistics on top of the v2 fair re-run.

v2 (run_phase2b_v2.py) established, by comparing 4 medians by eye, that a
correctly-fit learned beta-NLL stochastic decoder beats both matched-noise
controls on energy distance. Four consistent signs is only a sign test at
p=0.0625 -- not enough for a paper's central positive claim.

This script re-runs the EXACT v2 pipeline (same seeds/config), persists the
per-(dataset x cell type) group arrays that v2 discarded, and runs proper paired
statistics:

  * For each encoder, paired Wilcoxon signed-rank across the shared groups,
    comparing stochastic_betanll vs matched_noise_per_type, and
    stochastic_betanll vs matched_noise, on:
      (a) energy_distance                 (lower = better)
      (b) r_var_abs_log2  = |log2 r_var|  (per-group variance-calibration
          error, lower = better)
  * Median paired difference (stochastic - control; negative => learned better),
    the Wilcoxon statistic, the two-sided p-value, and the fraction of groups in
    which the learned decoder wins (strictly lower metric).
  * Holm correction across the 4 encoders within each comparison family
    (a family = one (metric, control) pair).

REPRODUCIBILITY GATE: the recomputed aggregate medians must match the v2 medians
to ~1e-6 (energy, |log2 r_var|, signed r_var, r_var dispersion, corr_att, cov).
The run ASSERTS this and fails loudly otherwise -- that is the check that the
predictions (not cached) were regenerated identically.

Writes results/phase2b_v3_paired.json. """
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

# Reuse the v2 machinery verbatim (importing runs only module-level setup, which
# also fixes sys.path for decoder_eval / parent_metric_probe / src.*).
import run_phase2b_v2 as v2  # noqa: E402
from run_phase2b_v2 import (  # noqa: E402
    CACHE, DEVICE, MIN_VAL_CELLS_PER_TYPE, NOISE_SEED, SEED,
    fit_stochastic_with_gate, load_geneformer_armA, log, per_type_residual_std,
)

from src.architecture.stochastic_decoder import sample_from  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

RESULTS = HERE / "results"
V2_JSON = RESULTS / "phase2b_v2_results.json"
REPRO_TOL = 1e-6

# metrics for the reproducibility gate: (v3-aggregate-key, v2-row-key)
REPRO_FIELDS = [
    ("energy_distance", "energy_distance_median"),
    ("r_var_abs_log2", "r_var_abs_log2_median"),
    ("r_var_signed", "r_var_signed_median"),
    ("correlation_attenuation", "correlation_attenuation_median"),
    ("cov_preservation", "cov_preservation_median"),
]
# paired-test config: metric -> "lower is better" (True for both here)
PAIRED_METRICS = ["energy_distance", "r_var_abs_log2"]
CONTROLS = ["matched_noise_per_type", "matched_noise"]
ENCODER_ORDER = ["pca_512", "randproj_512", "scgpt_512", "geneformer"]


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, preserving input order.
    m = len(pvals); sort ascending, adjusted_(k) = max_{j<=k} (m-j+1)*p_(j),
    clipped to 1 and made monotone non-decreasing."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj_sorted = [0.0] * m
    running = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * pvals[idx]
        running = max(running, val)
        adj_sorted[k] = min(1.0, running)
    out = [0.0] * m
    for k, idx in enumerate(order):
        out[idx] = adj_sorted[k]
    return out


def group_key(rec: dict) -> tuple[str, str]:
    return (rec["dataset"], rec["cell_type"])


def build_decoded_variants(Z, tr, va, te, nva, target, raw_te, lab_va, lab_te, name):
    """Reproduce v2's four decoded arrays for one encoder, verbatim."""
    Ztr, Yva_t = Z[tr], target[va]
    Ztest_arg = np.vstack([Z[va], Z[te]])
    pred_all, det_val, _, det_np = train_decoder(
        Ztr, target[tr], Z[va], Yva_t, Ztest_arg, seed=SEED, device=DEVICE)
    pred_val = pred_all[:nva]
    det_test = pred_all[nva:]
    det_pearson = reconstruction_metrics(det_test, raw_te)["mean_gene_pearson"]
    log(f"  [det] val_mse={det_val:.4f} params={det_np} r_pt={det_pearson:.4f}")

    s_g = (target[va] - pred_val).std(axis=0).astype(np.float32)
    eps = np.random.default_rng(NOISE_SEED).standard_normal(det_test.shape).astype(np.float32)
    ctrl_test = (det_test + s_g * eps).astype(np.float32)

    std_by_type, n_ok, n_fb = per_type_residual_std(
        target[va], pred_val, lab_va, s_g, Z.shape[1])
    std_test = np.vstack([std_by_type[t] for t in lab_te]).astype(np.float32)
    ctrl_pt_test = (det_test + std_test * eps).astype(np.float32)
    n_fb_cells = int(sum(1 for t in lab_te if (lab_va == t).sum() < MIN_VAL_CELLS_PER_TYPE))
    log(f"  [ctrl_per_type] types_estimated={n_ok} types_fallback={n_fb} "
        f"test_cells_on_fallback={n_fb_cells}/{len(lab_te)}")

    mu_test, lv_test, gate = fit_stochastic_with_gate(
        Ztr, target[tr], Z[va], Yva_t, Z[te], det_pearson, raw_te)
    stoch_test = sample_from(mu_test, lv_test, seed=NOISE_SEED)

    return {
        "deterministic": det_test,
        "stochastic_betanll": stoch_test,
        "matched_noise": ctrl_test,
        "matched_noise_per_type": ctrl_pt_test,
    }, {"gate_passed": gate["gate_passed"], "beta_used": gate["beta_used"]}


def main() -> None:
    t0 = time.time()
    v2res = json.loads(V2_JSON.read_text())
    v2_by = {(r["encoder"], r["variant"]): r for r in v2res["results"]}

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
    log(f"logX={logX.shape} train={tr.sum()} val={nva} test={te.sum()}")

    # --- Encoders (identical construction to v2) -----------------------------
    rng = np.random.default_rng(0)
    enc: dict[str, np.ndarray] = {}
    log("fitting PCA-512 on train cells")
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)
    W = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    enc["randproj_512"] = (logX @ W).astype(np.float32)
    p = CACHE / "enc_A_scgpt.npy"
    if p.exists():
        z = np.load(p)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
            log(f"loaded cached scgpt_512 {z.shape}")
    gf = load_geneformer_armA(ordered)
    if gf is not None and gf.shape[0] == logX.shape[0]:
        enc["geneformer"] = gf
        log(f"loaded cached geneformer (Arm A) {gf.shape}")

    # --- Recompute per-group distributional metrics, persist per-group arrays -
    # per_group_store[encoder][variant] = {group_key: {metric: value}}
    per_group_store: dict[str, dict[str, dict]] = {}
    recomputed_medians: dict[tuple[str, str], dict] = {}
    gate_info: dict[str, dict] = {}
    n_groups_seen = None

    for name, Z in enc.items():
        log(f"=== encoder {name} dim={Z.shape[1]} ===")
        decoded, gate = build_decoded_variants(
            Z, tr, va, te, nva, target, raw_te, lab_va, lab_te, name)
        gate_info[name] = gate
        per_group_store[name] = {}
        for vname, dec in decoded.items():
            agg = aggregate_over_groups(
                raw_te, dec, lab_te, ds_te,
                min_cells=20, energy_subsample=300, top_k=200, seed=0)
            if n_groups_seen is None:
                n_groups_seen = agg["n_groups"]
            per_group_store[name][vname] = {
                group_key(g): {
                    "energy_distance": g["energy_distance"],
                    "r_var_abs_log2": g["r_var_abs_log2"],
                    "r_var_signed": g["r_var_signed"],
                    "n_cells": g["n_cells"],
                }
                for g in agg["per_group"]
            }
            recomputed_medians[(name, vname)] = {
                "energy_distance": agg["energy_distance"]["median"],
                "r_var_abs_log2": agg["r_var_abs_log2"]["median"],
                "r_var_signed": agg["r_var_signed"]["median"],
                "correlation_attenuation": agg["correlation_attenuation"]["median"],
                "cov_preservation": agg["cov_preservation"]["median"],
                "r_var_dispersion": agg["r_var_dispersion"],
                "n_groups": agg["n_groups"],
            }
            log(f"  [{vname:22s}] n_groups={agg['n_groups']} "
                f"energy={agg['energy_distance']['median']:.6f} "
                f"|log2 r_var|={agg['r_var_abs_log2']['median']:.6f} "
                f"r_var={agg['r_var_signed']['median']:.6f}")

    # --- REPRODUCIBILITY GATE ------------------------------------------------
    log("=== reproducibility gate vs v2 medians (tol=%g) ===" % REPRO_TOL)
    repro_report = []
    failures = []
    for (name, vname), med in recomputed_medians.items():
        v2row = v2_by[(name, vname)]
        for agg_key, v2_key in REPRO_FIELDS:
            got = med[agg_key]
            exp = v2row[v2_key]
            d = abs(float(got) - float(exp))
            rec = {"encoder": name, "variant": vname, "field": v2_key,
                   "v3": float(got), "v2": float(exp), "abs_diff": d}
            repro_report.append(rec)
            if d > REPRO_TOL:
                failures.append(rec)
        # dispersion too (v2 stored r_var_dispersion at row level)
        d = abs(float(med["r_var_dispersion"]) - float(v2row["r_var_dispersion"]))
        rec = {"encoder": name, "variant": vname, "field": "r_var_dispersion",
               "v3": float(med["r_var_dispersion"]), "v2": float(v2row["r_var_dispersion"]),
               "abs_diff": d}
        repro_report.append(rec)
        if d > REPRO_TOL:
            failures.append(rec)
    max_diff = max((r["abs_diff"] for r in repro_report), default=0.0)
    log(f"  max abs median diff across {len(repro_report)} checks = {max_diff:.3e}")
    if failures:
        for f in failures:
            log(f"  MISMATCH {f['encoder']}/{f['variant']} {f['field']}: "
                f"v3={f['v3']:.8f} v2={f['v2']:.8f} diff={f['abs_diff']:.3e}")
    assert not failures, (
        f"Reproducibility gate FAILED: {len(failures)} median(s) differ from v2 by "
        f">{REPRO_TOL:g} (max abs diff {max_diff:.3e}). Predictions did not "
        f"regenerate identically -- do not trust the paired stats.")
    log("  reproducibility gate PASSED")

    # --- PAIRED WILCOXON (learned stochastic vs each control) ----------------
    encoders = [e for e in ENCODER_ORDER if e in enc]
    paired = {}
    for metric in PAIRED_METRICS:
        for control in CONTROLS:
            fam_key = f"{metric}__stochastic_vs_{control}"
            per_enc = []
            raw_p = []
            for name in encoders:
                stoch = per_group_store[name]["stochastic_betanll"]
                ctrl = per_group_store[name][control]
                keys = sorted(set(stoch) & set(ctrl))
                a, b = [], []
                for k in keys:
                    sv = stoch[k][metric]
                    cv = ctrl[k][metric]
                    if np.isfinite(sv) and np.isfinite(cv):
                        a.append(sv)
                        b.append(cv)
                a = np.asarray(a, dtype=np.float64)
                b = np.asarray(b, dtype=np.float64)
                diff = a - b  # negative => stochastic lower => learned better
                nz = np.count_nonzero(diff)
                if nz >= 1:
                    stat, pval = wilcoxon(a, b, alternative="two-sided",
                                          zero_method="wilcox")
                    stat = float(stat)
                    pval = float(pval)
                else:
                    stat, pval = float("nan"), 1.0
                win_frac = float(np.mean(a < b))
                per_enc.append({
                    "encoder": name,
                    "n_pairs": int(a.size),
                    "median_paired_diff": float(np.median(diff)),
                    "median_stochastic": float(np.median(a)),
                    "median_control": float(np.median(b)),
                    "wilcoxon_stat": stat,
                    "p_value": pval,
                    "learned_win_fraction": win_frac,
                })
                raw_p.append(pval)
            adj = holm(raw_p)
            for rec, ap in zip(per_enc, adj):
                rec["p_holm"] = float(ap)
                rec["significant_holm_0.05"] = bool(ap < 0.05)
            paired[fam_key] = {
                "metric": metric,
                "control": control,
                "direction": "lower is better (both metrics); negative median diff "
                             "and win_fraction>0.5 => learned decoder better",
                "holm_family": "4 encoders",
                "per_encoder": per_enc,
                "all_four_significant_holm": bool(all(r["significant_holm_0.05"] for r in per_enc)),
            }
            log(f"  [{fam_key}]")
            for r in per_enc:
                log(f"     {r['encoder']:12s} n={r['n_pairs']:3d} "
                    f"medDiff={r['median_paired_diff']:+.4f} "
                    f"win={r['learned_win_fraction']:.3f} "
                    f"p={r['p_value']:.2e} p_holm={r['p_holm']:.2e} "
                    f"{'SIG' if r['significant_holm_0.05'] else 'ns'}")

    # --- persist -------------------------------------------------------------
    # per-group arrays (compact: aligned lists per encoder over the shared keys)
    per_group_out = {}
    for name in encoders:
        variants = per_group_store[name]
        common = sorted(set.intersection(*[set(v) for v in variants.values()]))
        per_group_out[name] = {
            "group_keys": [list(k) for k in common],
            "n_cells": [variants["stochastic_betanll"][k]["n_cells"] for k in common],
            "variants": {
                vname: {
                    m: [variants[vname][k][m] for k in common]
                    for m in ("energy_distance", "r_var_abs_log2", "r_var_signed")
                }
                for vname in variants
            },
        }

    out = {
        "purpose": "Phase 2b v3: paired Wilcoxon signed-rank tests (Holm-corrected "
                   "across encoders) turning v2's by-eye 4-median comparison into a "
                   "proper paired test. Central claim: the label-free learned beta-NLL "
                   "stochastic decoder beats the label-using matched_noise_per_type "
                   "control (and the global matched_noise control) on energy distance "
                   "and |log2 r_var| per-group variance-calibration error.",
        "builds_on": "phase2b_v2_results.json (same seeds/config); this run regenerated "
                     "the uncached predictions and passed a 1e-6 reproducibility gate "
                     "against the v2 medians.",
        "reproducibility_gate": {
            "tol": REPRO_TOL, "max_abs_median_diff": max_diff, "passed": True,
            "n_checks": len(repro_report),
        },
        "seed": SEED, "noise_seed": NOISE_SEED,
        "n_groups": n_groups_seen,
        "encoders": encoders,
        "gate_per_encoder": gate_info,
        "paired_tests": paired,
        "notes": {
            "metrics": {
                "energy_distance": "proper distributional divergence (=0 iff equal); "
                                   "lower is better; the load-bearing discriminator.",
                "r_var_abs_log2": "|log2(var_decoded/var_real)| per group, lower=better "
                                  "(per-group variance-calibration error).",
            },
            "wilcoxon": "two-sided, zero_method='wilcox'; paired across shared "
                        "(dataset x cell type) groups.",
            "holm": "Holm-Bonferroni step-down across the 4 encoders within each "
                    "(metric, control) family.",
            "win_fraction": "fraction of shared groups where the learned decoder's "
                            "metric is strictly lower (better) than the control's.",
            "auxiliary_metrics_that_FAILED_to_discriminate": (
                "correlation_attenuation and covariance_preservation, the two metrics "
                "proposed as 'independent noise cannot fake this', both failed as clean "
                "discriminators. cov_preservation is scale-invariant (independent noise "
                "attenuates all correlations by a common factor, preserving rank order). "
                "correlation_attenuation is monotone in the amount of injected noise in "
                "this regime (every value > 1; deterministic is over-correlated, any "
                "noise pulls it down toward 1, nothing overshoots below 1), so it "
                "measures how much you diluted, not whether the dilution was structured "
                "-- which is why the label-informed control 'wins' it. energy_distance is "
                "the metric that actually discriminates."),
            "r_var_dispersion_confound": (
                "r_var_dispersion is confounded by magnitude and must never be quoted "
                "alone: a near-zero-variance predictor (deterministic, r_var median "
                "~0.12-0.26) gets the LOWEST dispersion of all. Restrict the dispersion "
                "comparison to variance-restoring variants (r_var median in [0.8,1.25]) "
                "and exclude deterministic."),
        },
        "per_group": per_group_out,
        "recomputed_medians": {
            f"{n}|{v}": recomputed_medians[(n, v)] for (n, v) in recomputed_medians
        },
    }
    (RESULTS / "phase2b_v3_paired.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'phase2b_v3_paired.json'} ===")


if __name__ == "__main__":
    main()
