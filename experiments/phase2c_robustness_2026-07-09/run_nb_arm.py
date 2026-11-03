"""Phase 2c Task 2 -- NEGATIVE-BINOMIAL arm: does Phase 2b's verdict depend on
the Gaussian-on-log1p likelihood?

Phase 2b fit a heteroscedastic Gaussian on log1p and concluded the heterogeneity
restored by stochastic decoding is calibrated noise, not learned structure. That
rests on ONE likelihood family. Here we re-test it with a NEGATIVE-BINOMIAL count
model fit on RAW counts (raw_full.npy), with a per-cell library-size offset.

Decoder: shared trunk + two heads (log_mu, log_theta), trained by NB NLL with a
Poisson mean warm-up; log_theta clamped and its saturation logged. NB mean is
``mu_ij = exp(log_mu_ij) * libsize_i / mean_libsize``. Counts are sampled at eval.

ACCEPTANCE GATE (in spirit of Phase 2b's): the NB decoder's MEAN must predict
about as well as a deterministic count-space regressor. We reuse the existing
MSE-on-log1p decoder as that deterministic baseline and compare mean-gene Pearson
of log1p(NB mean) vs log1p(real counts) against the baseline's log1p Pearson. If
the NB mean underfits badly (gap > GATE_TOL) the distributional numbers are
flagged UNTRUSTWORTHY -- exactly how the v1 Gaussian collapse was caught.

METRIC COMPARABILITY: every distributional metric is computed on ``log1p(counts)``
for both the NB samples and the real counts (real = logX[te] = log1p(raw[te])),
so the numbers sit on the SAME SCALE as the Phase 2b Gaussian arm.

CONTROL CORRECTNESS (fixed after an initial bug): ``matched_noise`` and
``matched_noise_per_type`` are built with residual noise sized and injected IN
LOG1P SPACE -- byte-for-byte the Gaussian arm's recipe
(run_phase2b_v2.build_decoded_variants), applied to the SAME deterministic
log1p decoder used as the NB gate baseline. An earlier version of this script
sized the residual noise in COUNT space (raw-count residuals, magnitude in the
hundreds given library sizes in the thousands) and only log1p'd afterward; that
swamps/clips the signal under log1p and annihilates the controls (energy
distance ~6-45 instead of ~1), which is not a real null. A sanity-check
assertion below verifies the rebuilt log1p controls reproduce the Gaussian
arm's v3 medians (pca_512: matched_noise_per_type ~0.951, matched_noise ~1.012
on energy_distance) before any paired test is trusted.

MEAN CONFOUND (fixed after a second bug): the first fix built ``poisson_resample``
by sampling ``Poisson(deterministic_decoder_mean)`` -- but NB's own fitted mean
is measurably BETTER than the deterministic decoder's (log1p Pearson 0.7479 vs
0.7420 on pca_512). So ``nb_stochastic`` vs ``poisson_resample`` varied TWO
things at once (a better mean AND learned dispersion), which does not isolate
whether the learned dispersion buys anything. Fixed by adding a second family of
controls anchored on NB's OWN fitted count mean ``mu`` (library-size-offset
included), holding the mean fixed and varying only the noise model:

  * ``poisson_on_nb_mean``          -- Poisson(nb_mu), then log1p. This is NB
    with theta -> infinity: identical mean, NO learned overdispersion.
    ``nb_stochastic`` vs this isolates the learned dispersion exactly.
  * ``nbmean_matched_noise``        -- log1p(nb_mu) + per-gene log1p residual
    noise (global, label-free), same recipe as ``matched_noise`` but anchored
    on nb_mu instead of the deterministic decoder's mean.
  * ``nbmean_matched_noise_per_type`` -- same, per (cell type, gene); uses true
    labels, deliberately advantaged.

Eight variants total, in two families that answer DIFFERENT questions:

  PROCEDURE family (different mean AND different noise model vs nb_stochastic;
  "is a count-likelihood generative procedure better end-to-end than a
  Gaussian-on-log1p deterministic mean + noise injection?"):
    * deterministic           -- deterministic count-space mean, no sampling
    * matched_noise            -- deterministic LOG1P mean + global log1p noise
    * matched_noise_per_type   -- deterministic LOG1P mean + per-type log1p noise
    * poisson_resample         -- Poisson(deterministic count mean), then log1p

  DISPERSION-ISOLATING family (SAME mean as nb_stochastic -- NB's own mu --
  only the noise model varies; "does the learned per-gene dispersion buy
  anything beyond restoring count-space shot noise around the SAME mean?"):
    * poisson_on_nb_mean          -- Poisson(nb_mu), then log1p
    * nbmean_matched_noise        -- log1p(nb_mu) + global log1p noise
    * nbmean_matched_noise_per_type -- log1p(nb_mu) + per-type log1p noise

  * nb_stochastic -- sample counts from the fitted NB (mean nb_mu, learned
    per-gene dispersion theta)

Same paired Wilcoxon + Holm on energy_distance and r_var_abs_log2, on pca_512 and
scgpt_512, against every control in both families. Default 1 seed; pass --seeds 3
for a 3-seed check. Pass --encoders to restrict which encoders run (for fast
iteration).

"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
sys.path.insert(0, str(PHASE2B))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from run_phase2b_v2 import (  # noqa: E402
    CACHE, DEVICE, GATE_TOL, MIN_VAL_CELLS_PER_TYPE, per_type_residual_std,
)
from src.architecture.stochastic_decoder import (  # noqa: E402
    log_theta_stats, sample_nb, train_nb_decoder,
)
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

ENCODERS = ["pca_512", "scgpt_512"]
LOGTHETA_MIN, LOGTHETA_MAX = -4.0, 8.0
PAIRED_METRICS = ["energy_distance", "r_var_abs_log2"]
# PROCEDURE family: different mean AND different noise model vs nb_stochastic.
# Answers "is a count-likelihood generative procedure better end-to-end?"
CONTROLS_PROCEDURE = ["matched_noise_per_type", "matched_noise", "poisson_resample"]
# DISPERSION family: SAME mean as nb_stochastic (NB's own mu); only the noise
# model varies. Answers "does the learned dispersion buy anything beyond
# restoring shot noise around the same mean?"
CONTROLS_DISPERSION = [
    "nbmean_matched_noise_per_type", "nbmean_matched_noise", "poisson_on_nb_mean",
]
CONTROL_QUESTION = {c: "procedure" for c in CONTROLS_PROCEDURE}
CONTROL_QUESTION.update({c: "dispersion" for c in CONTROLS_DISPERSION})
CONTROLS = CONTROLS_PROCEDURE + CONTROLS_DISPERSION
# Gaussian-arm (PHASE2B_V3_REPORT.md) pca_512 energy_distance control medians,
# used as the sanity-check target for the rebuilt log1p-space controls.
V3_PCA512_ENERGY_MEDIAN = {"matched_noise_per_type": 0.951, "matched_noise": 1.012}
SANITY_TOL = 0.2  # absolute; different noise RNG draws than v2/v3, same recipe


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (m - k) * pvals[idx])
        adj[k] = min(1.0, running)
    out = [0.0] * m
    for k, idx in enumerate(order):
        out[idx] = adj[k]
    return out


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def build_nb_variants(Z, tr, va, te, nva, logX, raw, lib, mean_lib,
                      lab_va, lab_te, seed):
    """Train the NB decoder + deterministic log1p baseline at one seed, build
    the eight variants (2 base + 3 procedure controls + 3 dispersion-isolating
    controls), and return them all mapped to log1p space along with the
    acceptance-gate record.

    real reference for metrics is logX[te] (= log1p(raw[te])); every decoded
    variant is converted to log1p(counts) (or built directly in log1p space for
    the noise controls) so the scale matches the Gaussian arm.
    """
    raw_tr, raw_va, raw_te = raw[tr], raw[va], raw[te]
    lib_tr, lib_va, lib_te = lib[tr], lib[va], lib[te]

    # (0) Deterministic MSE-on-log1p baseline. This IS the Gaussian arm's
    # deterministic decoder (same architecture/recipe/target/seed). Its log1p
    # predictions anchor the PROCEDURE family's log1p noise controls directly
    # (no count round-trip), and its expm1 gives the count-space mean for the
    # NB gate comparison and the procedure-family Poisson-resample null.
    Ztest_arg = np.vstack([Z[va], Z[te]])
    det_log_all, det_val, _, det_np = train_decoder(
        Z[tr], logX[tr], Z[va], logX[va], Ztest_arg, seed=seed, device=DEVICE)
    det_log_val = det_log_all[:nva]
    det_log_te = det_log_all[nva:]
    det_count_te = np.clip(np.expm1(det_log_te), 0.0, None).astype(np.float32)
    det_log_pearson = reconstruction_metrics(det_log_te, logX[te])["mean_gene_pearson"]
    log(f"    [det baseline] val_mse={det_val:.4f} log1p_r={det_log_pearson:.4f}")

    # (1) NB decoder on raw counts with library-size offset. Evaluated on
    # val+test (not just test) so its OWN fitted mean is available on
    # validation cells too -- needed to size the dispersion-isolating controls'
    # residual noise (mirrors how the deterministic decoder's det_log_val is
    # used for the procedure family). This does not change training (Z_test is
    # only used in the final chunked eval pass, never in the training loop).
    lib_arg = np.concatenate([lib_va, lib_te]).astype(np.float32)
    mu_all, theta_all, nb_val, nb_hist, nb_np = train_nb_decoder(
        Z[tr], raw_tr, lib_tr, Z[va], raw_va, lib_va, Ztest_arg, lib_arg,
        mean_libsize=mean_lib, seed=seed, device=DEVICE,
        logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
        warmup_epochs=15, logtheta_lr_scale=0.1)
    mu_val, mu_te = mu_all[:nva], mu_all[nva:]
    theta_val, theta_te = theta_all[:nva], theta_all[nva:]
    # NB mean quality on the comparable log1p scale (test cells, matches gate).
    nb_mean_log_te = np.log1p(np.clip(mu_te, 0.0, None)).astype(np.float32)
    nb_mean_log_val = np.log1p(np.clip(mu_val, 0.0, None)).astype(np.float32)
    nb_log_pearson = reconstruction_metrics(nb_mean_log_te, logX[te])["mean_gene_pearson"]
    gap = det_log_pearson - nb_log_pearson
    # ONE-SIDED underfit guard (see src/evaluation/acceptance.py): FAIL only if
    # the NB mean is WORSE than deterministic by > GATE_TOL. A better-fitting NB
    # mean (negative gap) is healthy and PASSES.
    gate_passed = mean_acceptance_gate(det_log_pearson, nb_log_pearson, GATE_TOL)
    lt = np.log(np.clip(theta_te, 1e-8, None))
    lt_stats = log_theta_stats(lt, LOGTHETA_MIN, LOGTHETA_MAX)
    log(f"    [NB] mean_log1p_r={nb_log_pearson:.4f} (det={det_log_pearson:.4f} "
        f"gap={gap:+.4f}) gate={'PASS' if gate_passed else 'FAIL'} "
        f"logtheta[min={lt_stats['min']:.2f} med={lt_stats['median']:.2f} "
        f"max={lt_stats['max']:.2f} %@min={lt_stats['frac_at_min_clamp']:.1%} "
        f"%@max={lt_stats['frac_at_max_clamp']:.1%}]")

    nb_counts = sample_nb(mu_te, theta_te, seed=seed)

    # (2) PROCEDURE family, log1p-space noise controls anchored on the
    # DETERMINISTIC decoder's mean -- byte-for-byte the Gaussian arm's recipe
    # (run_phase2b_v2.build_decoded_variants): residual std of LOG1P residuals,
    # noise injected directly on the log1p scale. (An earlier bug sized this
    # noise in COUNT space, which given library sizes in the thousands is
    # orders of magnitude larger than a log1p unit and annihilates the
    # control; fixed by moving the whole computation to log1p space.)
    s_g = (logX[va] - det_log_val).std(axis=0).astype(np.float32)
    eps = np.random.default_rng(seed).standard_normal(det_log_te.shape).astype(np.float32)
    mn_log = (det_log_te + s_g * eps).astype(np.float32)

    std_by_type, n_ok, n_fb = per_type_residual_std(
        logX[va], det_log_val, lab_va, s_g, Z.shape[1])
    std_te = np.vstack([std_by_type[t] for t in lab_te]).astype(np.float32)
    mnt_log = (det_log_te + std_te * eps).astype(np.float32)
    n_fb_cells = int(sum(1 for t in lab_te if (lab_va == t).sum() < MIN_VAL_CELLS_PER_TYPE))
    log(f"    [ctrl_per_type/det-mean] types_ok={n_ok} types_fallback={n_fb} "
        f"test_cells_on_fallback={n_fb_cells}/{len(lab_te)}")

    # Poisson resampling around the DETERMINISTIC mean -- procedure-family
    # count-space null (different mean AND no learned dispersion vs nb_stochastic).
    rng_pois = np.random.default_rng((seed, 1))
    pois_counts = rng_pois.poisson(det_count_te).astype(np.float32)

    # (3) DISPERSION-ISOLATING family, anchored on NB's OWN fitted mean mu --
    # holds the mean fixed at exactly what nb_stochastic uses, so a win here is
    # attributable to the LEARNED NOISE MODEL, not to a better mean. Fixes the
    # confound in the first version of this control set (poisson_resample was
    # built around the deterministic decoder's mean, which is measurably worse
    # than NB's own mean, so it varied mean AND dispersion at once).
    rng_pois_nb = np.random.default_rng((seed, 2))
    pois_nb_counts = rng_pois_nb.poisson(np.clip(mu_te, 0.0, None)).astype(np.float32)

    s_g_nb = (logX[va] - nb_mean_log_val).std(axis=0).astype(np.float32)
    eps_nb = np.random.default_rng((seed, 3)).standard_normal(nb_mean_log_te.shape).astype(np.float32)
    nbmean_mn_log = (nb_mean_log_te + s_g_nb * eps_nb).astype(np.float32)

    std_by_type_nb, n_ok_nb, n_fb_nb = per_type_residual_std(
        logX[va], nb_mean_log_val, lab_va, s_g_nb, Z.shape[1])
    std_te_nb = np.vstack([std_by_type_nb[t] for t in lab_te]).astype(np.float32)
    nbmean_mnt_log = (nb_mean_log_te + std_te_nb * eps_nb).astype(np.float32)
    log(f"    [ctrl_per_type/nb-mean] types_ok={n_ok_nb} types_fallback={n_fb_nb}")

    # All variants -> log1p scale so metrics match the Gaussian arm exactly.
    decoded_log = {
        "deterministic": np.log1p(det_count_te),
        "nb_stochastic": np.log1p(nb_counts),
        "matched_noise": mn_log,
        "matched_noise_per_type": mnt_log,
        "poisson_resample": np.log1p(pois_counts),
        "poisson_on_nb_mean": np.log1p(pois_nb_counts),
        "nbmean_matched_noise": nbmean_mn_log,
        "nbmean_matched_noise_per_type": nbmean_mnt_log,
    }
    gate = {
        "det_log_pearson": float(det_log_pearson),
        "nb_mean_log_pearson": float(nb_log_pearson),
        "gap_vs_det": float(gap),
        "gate_passed": gate_passed,
        "nb_val_nll": float(nb_val),
        "nb_params": int(nb_np),
        "logtheta_stats": lt_stats,
        "types_estimated": int(n_ok), "types_fallback": int(n_fb),
        "test_cells_on_fallback": n_fb_cells,
        "types_estimated_nb_mean": int(n_ok_nb), "types_fallback_nb_mean": int(n_fb_nb),
    }
    return decoded_log, gate


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
    diff = a - b
    if np.count_nonzero(diff) >= 1:
        stat, pval = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        stat, pval = float(stat), float(pval)
    else:
        stat, pval = float("nan"), 1.0
    return {
        "n_pairs": int(a.size),
        "median_paired_diff": float(np.median(diff)),
        "median_stochastic": float(np.median(a)),
        "median_control": float(np.median(b)),
        "wilcoxon_stat": stat, "p_value": pval,
        "learned_win_fraction": float(np.mean(a < b)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (1 or 3)")
    ap.add_argument("--encoders", type=str, default=None,
                    help="comma-separated subset of {pca_512,scgpt_512}; default both")
    args = ap.parse_args()
    seeds = [42, 43, 44][:args.seeds]

    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())
    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    assert np.allclose(np.log1p(raw[:2000]), logX[:2000], atol=1e-5), \
        "logX must equal log1p(raw) for metric comparability"
    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())
    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    log(f"raw={raw.shape} logX={logX.shape} mean_train_libsize={mean_lib:.1f} "
        f"train={int(tr.sum())} val={nva} test={int(te.sum())}")

    enc = {}
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)
    p = CACHE / "enc_A_scgpt.npy"
    if p.exists():
        z = np.load(p)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
    encoders = [e for e in ENCODERS if e in enc]
    if args.encoders:
        wanted = set(args.encoders.split(","))
        encoders = [e for e in encoders if e in wanted]
    log(f"encoders available: {encoders}; seeds={seeds}")

    # per_seed[encoder][seed] = {"gate":..., "aggs": {variant: per_group_map}}
    per_seed = {name: {} for name in encoders}
    n_groups = None
    sanity_report = None
    for name in encoders:
        Z = enc[name]
        for seed in seeds:
            log(f"=== NB arm encoder {name} seed {seed} ===")
            decoded_log, gate = build_nb_variants(
                Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
            aggs = {}
            aggs_full = {}
            for vname, dec in decoded_log.items():
                agg = aggregate_over_groups(
                    logX[te], dec, lab_te, ds_te,
                    min_cells=20, energy_subsample=300, top_k=200, seed=0)
                aggs[vname] = per_group_map(agg)
                aggs_full[vname] = agg
                if n_groups is None:
                    n_groups = agg["n_groups"]
            per_seed[name][seed] = {"gate": gate, "aggs": aggs}

            # --- SANITY CHECK: the rebuilt log1p-space controls, on pca_512's
            # first seed, must reproduce PHASE2B_V3_REPORT.md's medians for
            # matched_noise / matched_noise_per_type on energy_distance. This is
            # the correctness test for the fix: same deterministic decoder
            # recipe/target/seed as the Gaussian arm, so if this control were
            # still mis-specified in COUNT space (the original bug) these
            # medians would be off by 6-45x, not within SANITY_TOL.
            if name == "pca_512" and seed == seeds[0]:
                sanity_report = {}
                for vname, expected in V3_PCA512_ENERGY_MEDIAN.items():
                    got = aggs_full[vname]["energy_distance"]["median"]
                    diff = abs(got - expected)
                    ok = diff <= SANITY_TOL
                    sanity_report[vname] = {
                        "got": float(got), "expected_v3": expected,
                        "abs_diff": float(diff), "passed": bool(ok),
                    }
                    log(f"  [SANITY CHECK] {vname} energy_distance median: "
                        f"got={got:.4f} v3_expected~={expected:.3f} "
                        f"diff={diff:.4f} {'PASS' if ok else 'FAIL'}")
                failed = [v for v, r in sanity_report.items() if not r["passed"]]
                assert not failed, (
                    f"NB-arm log1p noise-control sanity check FAILED for {failed}: "
                    f"rebuilt controls do not reproduce the Gaussian arm's v3 "
                    f"medians within {SANITY_TOL} -- do not trust the paired "
                    f"tests. Report: {sanity_report}")
                log("  [SANITY CHECK] PASSED -- log1p-space controls reproduce "
                    "the Gaussian arm's v3 medians within tolerance.")

    # --- Paired Wilcoxon + Holm (per seed, Holm across the 2 encoders) --------
    # nb_stochastic vs each control, on each metric.
    paired_tests = {}
    for seed in seeds:
        paired_tests[seed] = {}
        for metric in PAIRED_METRICS:
            for control in CONTROLS:
                fam = f"{metric}__nb_stochastic_vs_{control}"
                per_enc, raw_p = [], []
                for name in encoders:
                    a_map = per_seed[name][seed]["aggs"]["nb_stochastic"]
                    b_map = per_seed[name][seed]["aggs"][control]
                    r = paired(a_map, b_map, metric)
                    r["encoder"] = name
                    r["gate_passed"] = per_seed[name][seed]["gate"]["gate_passed"]
                    per_enc.append(r)
                    raw_p.append(r["p_value"])
                for r, ap in zip(per_enc, holm(raw_p)):
                    r["p_holm"] = float(ap)
                    r["significant_holm_0.05"] = bool(ap < 0.05)
                paired_tests[seed][fam] = {
                    "metric": metric, "control": control,
                    "question": CONTROL_QUESTION[control],
                    "question_meaning": (
                        "different mean AND different noise model vs nb_stochastic; "
                        "tests whether a count-likelihood generative procedure beats "
                        "the Gaussian-arm's deterministic-mean+noise procedure end-to-end"
                        if CONTROL_QUESTION[control] == "procedure" else
                        "SAME mean as nb_stochastic (NB's own mu); only the noise model "
                        "varies; tests whether the learned per-gene dispersion buys "
                        "anything beyond restoring shot/residual noise around that mean"
                    ),
                    "holm_family": "encoders within seed", "per_encoder": per_enc,
                }
                log(f"  seed={seed} [{fam}]")
                for r in per_enc:
                    log(f"     {r['encoder']:12s} n={r['n_pairs']:3d} "
                        f"medDiff={r['median_paired_diff']:+.4f} "
                        f"win={r['learned_win_fraction']:.3f} "
                        f"p={r['p_value']:.2e} p_holm={r['p_holm']:.2e} "
                        f"{'SIG' if r['significant_holm_0.05'] else 'ns'} "
                        f"{'[gate FAIL]' if not r['gate_passed'] else ''}")

    gate_records = {
        name: {seed: per_seed[name][seed]["gate"] for seed in seeds}
        for name in encoders
    }
    out = {
        "purpose": "Phase 2c Task 2 NB arm: negative-binomial count decoder on raw "
                   "counts (library-size offset), metrics on log1p(counts) for "
                   "scale-comparability with the Gaussian arm. Tests whether the "
                   "Phase 2b verdict depends on the Gaussian-on-log1p likelihood.",
        "seeds": seeds, "encoders": encoders, "n_groups": n_groups,
        "mean_train_libsize": mean_lib,
        "logtheta_clamp": [LOGTHETA_MIN, LOGTHETA_MAX],
        "acceptance_gate": {
            "rule": "NB mean log1p Pearson within %.2f of the deterministic "
                    "MSE-on-log1p baseline; else distributional numbers flagged "
                    "UNTRUSTWORTHY." % GATE_TOL,
            "tol": GATE_TOL,
        },
        "metric_scale": "all distributional metrics on log1p(counts); real = "
                        "logX[te] = log1p(raw[te]).",
        "control_sanity_check": {
            "rule": "matched_noise / matched_noise_per_type (log1p-space, pca_512, "
                    "first seed) must reproduce PHASE2B_V3_REPORT.md's energy_distance "
                    "medians within tol=%.2f absolute." % SANITY_TOL,
            "tol": SANITY_TOL, "result": sanity_report,
        },
        "gate_records": gate_records,
        "paired_tests": paired_tests,
    }
    (RESULTS / "nb_arm.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'nb_arm.json'} ===")


if __name__ == "__main__":
    main()
