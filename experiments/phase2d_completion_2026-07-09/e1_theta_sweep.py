"""Phase 2d E1 -- THETA SWEEP: convert the §15 over-inflation mechanism from
inference to evidence.

PHASE2C_REPORT.md §Task-2 found that on pca_512 a Poisson resampling of NB's OWN
fitted count mean (poisson_on_nb_mean = NB with theta -> inf, i.e. zero learned
overdispersion) BEATS the full learned-dispersion NB (nb_stochastic) on both
energy_distance and |log2 r_var|, 3/3 seeds. The PROPOSED but UNVERIFIED mechanism:
NB's learned per-gene dispersion (fitted median theta ~= 1.31, variance mu+0.76 mu^2)
OVER-INFLATES conditional within-type variance, because the decoder already
conditions on each cell's latent (which absorbs biological cell-to-cell variance),
so fitting MARGINAL overdispersion double-counts it. Signed r_var was never
separated per NB variant, so the DIRECTION of the miscalibration was unproven.

This script measures it directly. Train the NB decoder ONCE per seed on pca_512
(gate it exactly as §Task-2). Then, holding NB's OWN fitted count mean ``mu_te``
FIXED, sample counts from NB with a fixed, IMPOSED scalar dispersion ``theta`` over
a grid {0.25, 0.5, 1.0, fitted_median_theta, 2, 5, 20, 100, inf} (inf = Poisson).
NO retraining per theta -- only the sampling changes (sample_nb_fixed_theta).

For each theta, on held-out TEST cells over the 314 (dataset x cell type) groups
(energy_distance uses <=300 cells/side, fixed RNG), report:
  * signed r_var  median + IQR  (>1 = over-inflated variance, <1 = under-inflated)
  * |log2 r_var|  median         (calibration error)
  * energy_distance median

PREDICTION (over-inflation story): signed r_var DECREASES monotonically as theta
increases, crossing 1.0 somewhere; energy improves monotonically toward theta->inf.
We report the theta at which signed r_var crosses 1.0 (log-theta interpolation).
If energy is instead MINIMISED at a finite theta, the "learned dispersion is always
counterproductive" claim is too strong and is narrowed here explicitly.

Also: paired Wilcoxon (learned-theta NB ``nb_stochastic`` vs Poisson-on-nb-mean)
per seed, to confirm §Task-2's dispersion-isolating result reproduces here.

Reuses the Phase 2c NB machinery and the shared distributional module.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
sys.path.insert(0, str(PHASE2B))
sys.path.insert(0, str(PHASE2A))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from run_phase2b_v2 import CACHE, DEVICE, GATE_TOL  # noqa: E402
from src.architecture.stochastic_decoder import (  # noqa: E402
    log_theta_stats, sample_nb, sample_nb_fixed_theta, train_nb_decoder,
)
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import (  # noqa: E402
    aggregate_over_groups, energy_distance,
)
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
LOGTHETA_MIN, LOGTHETA_MAX = -4.0, 8.0
# Fixed imposed-theta grid; "fitted_median" is inserted per seed at run time.
FIXED_THETAS = [0.25, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, float("inf")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def paired(a_map, b_map, metric):
    """Paired Wilcoxon of variant A vs B over shared groups (lower=better)."""
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
        "median_a_nb_stochastic": float(np.median(a)),
        "median_b_poisson": float(np.median(b)),
        "wilcoxon_stat": stat, "p_value": pval,
        "nb_stochastic_win_fraction": float(np.mean(a < b)),
    }


def flip_theta(theta_vals, series, neutral):
    """Log-theta-interpolated theta at which a (finite-theta) ``series`` crosses
    its ``neutral`` value (1.0 for a variance ratio, 0.5 for a fraction>1). This
    replaces the meaningless unrestricted-r_var crossing statistic: it is applied
    ONLY to the expression-restricted direction stats, which actually move with
    theta. ``theta_vals`` may include inf (dropped). Returns the flip theta and
    the bracket, or a no-crossing record with the observed range and the sign of
    the series relative to neutral across the whole grid.

    Convention: over-inflation is series > neutral; the flip theta marks where
    the decoded variance transitions from over- to under-inflated as theta rises."""
    finite = [(t, s) for t, s in zip(theta_vals, series)
              if np.isfinite(t) and np.isfinite(s)]
    if len(finite) < 2:
        return {"theta": None, "method": "insufficient_points"}
    ts = np.array([t for t, _ in finite], dtype=np.float64)
    ss = np.array([s for _, s in finite], dtype=np.float64)
    order = np.argsort(ts)
    ts, ss = ts[order], ss[order]
    for i in range(len(ts) - 1):
        d0, d1 = ss[i] - neutral, ss[i + 1] - neutral
        if d0 == 0.0:
            return {"theta": float(ts[i]), "method": "node", "neutral": neutral}
        if d0 * d1 < 0.0:
            lt0, lt1 = np.log(ts[i]), np.log(ts[i + 1])
            frac = (neutral - ss[i]) / (ss[i + 1] - ss[i])
            return {"theta": float(np.exp(lt0 + frac * (lt1 - lt0))),
                    "method": "loginterp", "neutral": neutral,
                    "bracket_theta": [float(ts[i]), float(ts[i + 1])],
                    "bracket_series": [float(ss[i]), float(ss[i + 1])]}
    all_over = bool(np.all(ss > neutral))
    all_under = bool(np.all(ss < neutral))
    return {"theta": None, "method": "no_crossing", "neutral": neutral,
            "series_min": float(ss.min()), "series_max": float(ss.max()),
            "all_above_neutral": all_over, "all_below_neutral": all_under,
            "note": ("decoded variance stays OVER-inflated across the whole finite "
                     "grid" if all_over else
                     "decoded variance stays UNDER-inflated across the whole finite "
                     "grid" if all_under else
                     "series does not cross neutral but is not uniformly one-sided")}


def _group_masks(labels, dataset_ids, min_cells):
    labels = np.asarray(labels)
    dataset_ids = np.asarray(dataset_ids)
    groups = []
    for d in np.unique(dataset_ids):
        dm = dataset_ids == d
        for t in np.unique(labels[dm]):
            mask = dm & (labels == t)
            if int(mask.sum()) >= min_cells:
                groups.append(mask)
    return groups


def energy_ci_over_count_draws(real, mu, theta, labels, dataset_ids, min_cells,
                               subsample, k_draws, base_seed):
    """Median-across-groups energy_distance recomputed over ``k_draws`` INDEPENDENT
    count-sample draws (each with its own subsample RNG), to put honest error bars
    on the energy estimate.

    The dominant noise here is the COUNT SAMPLING (a fresh NB/Poisson draw per
    theta), NOT the energy subsample -- because on this panel the median group has
    < ``subsample`` cells, so subsampling is a no-op for most groups and the
    subsample-only CI collapses to ~0, which would falsely make a 1.5e-4 plateau
    gap look 'distinguishable'. Sampling the counts K times (as the real data is
    itself one draw) gives the CI that a strictly-finite-optimum claim must clear.
    ``theta`` may be inf (Poisson). Returns (mean, sd, per_draw_list)."""
    real = np.asarray(real, dtype=np.float64)
    groups = _group_masks(labels, dataset_ids, min_cells)
    per_draw = []
    for k in range(k_draws):
        counts = sample_nb_fixed_theta(mu, theta, seed=base_seed + 100 + k)
        dec = np.log1p(counts).astype(np.float64)
        rng = np.random.default_rng(base_seed + 500 + k)
        eds = []
        for mask in groups:
            R = real[mask]
            D = dec[mask]
            n = R.shape[0]
            if n > subsample:
                ri = rng.choice(n, subsample, replace=False)
                di = rng.choice(n, subsample, replace=False)
                eds.append(energy_distance(R[ri], D[di]))
            else:
                eds.append(energy_distance(R, D))
        per_draw.append(float(np.median(eds)))
    return float(np.mean(per_draw)), float(np.std(per_draw, ddof=1)), per_draw


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
    assert np.allclose(np.log1p(raw[:2000]), logX[:2000], atol=1e-5), \
        "logX must equal log1p(raw) for metric comparability"
    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())
    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    log(f"raw={raw.shape} mean_train_lib={mean_lib:.1f} "
        f"train={int(tr.sum())} val={nva} test={int(te.sum())}")

    Z = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)
    log(f"pca_512 encoder Z={Z.shape}")

    per_seed = {}
    for seed in SEEDS:
        log(f"=== E1 theta-sweep pca_512 seed {seed} ===")
        raw_tr, raw_va = raw[tr], raw[va]
        lib_tr, lib_va, lib_te = lib[tr], lib[va], lib[te]

        # Deterministic MSE-on-log1p baseline (gate anchor; same as §Task-2).
        Ztest_arg = np.vstack([Z[va], Z[te]])
        det_log_all, det_val, _, _ = train_decoder(
            Z[tr], logX[tr], Z[va], logX[va], Ztest_arg, seed=seed, device=DEVICE)
        det_log_te = det_log_all[nva:]
        det_log_pearson = reconstruction_metrics(det_log_te, logX[te])["mean_gene_pearson"]

        # NB decoder (train once), evaluate mu/theta on test cells.
        mu_te, theta_te, nb_val, _, nb_np = train_nb_decoder(
            Z[tr], raw_tr, lib_tr, Z[va], raw_va, lib_va, Z[te], lib_te,
            mean_libsize=mean_lib, seed=seed, device=DEVICE,
            logtheta_min=LOGTHETA_MIN, logtheta_max=LOGTHETA_MAX,
            warmup_epochs=15, logtheta_lr_scale=0.1)
        nb_mean_log_te = np.log1p(np.clip(mu_te, 0.0, None)).astype(np.float32)
        nb_log_pearson = reconstruction_metrics(nb_mean_log_te, logX[te])["mean_gene_pearson"]
        gap = det_log_pearson - nb_log_pearson
        # one-sided underfit guard (src/evaluation/acceptance.py): a BETTER NB
        # mean (negative gap) is healthy and passes.
        gate_passed = mean_acceptance_gate(det_log_pearson, nb_log_pearson, GATE_TOL)
        lt = np.log(np.clip(theta_te, 1e-8, None))
        lt_stats = log_theta_stats(lt, LOGTHETA_MIN, LOGTHETA_MAX)
        fitted_median_theta = float(lt_stats["theta_median"])
        log(f"  [NB gate] det_log_r={det_log_pearson:.4f} nb_mean_log_r={nb_log_pearson:.4f} "
            f"gap={gap:+.4f} gate={'PASS' if gate_passed else 'FAIL'} "
            f"fitted_median_theta={fitted_median_theta:.3f}")

        # Build the full theta grid for this seed (insert the fitted median),
        # ascending with inf last.
        grid = sorted(set(FIXED_THETAS) | {fitted_median_theta},
                      key=lambda x: (np.isinf(x), x))

        # nb_stochastic: LEARNED per-gene theta around the SAME mu (the model).
        nb_counts = sample_nb(mu_te, theta_te, seed=seed)
        nb_stoch_log = np.log1p(nb_counts).astype(np.float32)
        agg_nb = aggregate_over_groups(
            logX[te], nb_stoch_log, lab_te, ds_te,
            min_cells=20, energy_subsample=300, top_k=200, seed=0)
        map_nb = per_group_map(agg_nb)

        # Imposed fixed-theta variants (mu held fixed = NB's own mu). For each
        # theta we record BOTH the (degenerate) unrestricted signed r_var and the
        # expression-restricted direction stats (median signed ratio, mean
        # log2-ratio, fraction of genes with ratio>1) on the two restrictions
        # (mean-count>=0.5 and top-200-variance), plus energy with a subsample CI.
        K_ENERGY = 5
        sweep = []
        maps_fixed = {}
        for i, theta in enumerate(grid):
            counts = sample_nb_fixed_theta(mu_te, theta, seed=(seed * 1000 + i))
            dec_log = np.log1p(counts).astype(np.float32)
            agg = aggregate_over_groups(
                logX[te], dec_log, lab_te, ds_te,
                min_cells=20, energy_subsample=300, top_k=200, seed=0)
            e_mean, e_sd, e_seeds = energy_ci_over_count_draws(
                logX[te], mu_te, theta, lab_te, ds_te, min_cells=20,
                subsample=300, k_draws=K_ENERGY, base_seed=(seed * 1000 + i))
            tkey = "inf" if np.isinf(theta) else round(float(theta), 4)
            maps_fixed[tkey] = per_group_map(agg)
            rec = {
                "theta": ("inf" if np.isinf(theta) else float(theta)),
                "is_fitted_median": bool(abs(theta - fitted_median_theta) < 1e-9),
                # unrestricted (degenerate on sparse panel -- reported to show it)
                "r_var_signed_median": agg["r_var_signed"]["median"],
                "r_var_signed_iqr": agg["r_var_signed"]["iqr"],
                "r_var_abs_log2_median": agg["r_var_abs_log2"]["median"],
                # expression-restricted direction stats (the informative ones)
                "expr_signed_median": agg["r_var_signed_expressed"]["median"],
                "expr_signed_iqr": agg["r_var_signed_expressed"]["iqr"],
                "expr_meanlog2_median": agg["r_var_meanlog2_expressed"]["median"],
                "expr_frac_gt1_median": agg["r_var_frac_gt1_expressed"]["median"],
                "n_expr_median": agg["n_expr"]["median"],
                "topk_signed_median": agg["r_var_signed_topk"]["median"],
                "topk_meanlog2_median": agg["r_var_meanlog2_topk"]["median"],
                "topk_frac_gt1_median": agg["r_var_frac_gt1_topk"]["median"],
                "n_topk_median": agg["n_topk"]["median"],
                # energy with subsample CI
                "energy_distance_median": agg["energy_distance"]["median"],
                "energy_distance_iqr": agg["energy_distance"]["iqr"],
                "energy_ci_mean": e_mean, "energy_ci_sd": e_sd,
                "energy_ci_seeds": e_seeds,
                "n_groups": agg["n_groups"],
            }
            sweep.append(rec)
        for rec in sweep:
            tstr = rec["theta"] if isinstance(rec["theta"], str) else f"{rec['theta']:.3f}"
            log(f"    theta={tstr:>8s} unrestr_signed={rec['r_var_signed_median']:.3f} | "
                f"EXPR signed={rec['expr_signed_median']:.3f} "
                f"meanlog2={rec['expr_meanlog2_median']:+.3f} "
                f"frac>1={rec['expr_frac_gt1_median']:.3f} n={rec['n_expr_median']:.0f} | "
                f"energy={rec['energy_ci_mean']:.4f}+/-{rec['energy_ci_sd']:.4f}"
                f"{'  <-fitted' if rec['is_fitted_median'] else ''}")

        # DIRECTION-FLIP analysis on the RESTRICTED stats (the unrestricted median
        # is pinned at 1.0 by sparsity and cannot localize a crossing).
        theta_vals = [np.inf if r["theta"] == "inf" else r["theta"] for r in sweep]
        flips = {
            "expr_signed_median_vs_1": flip_theta(
                theta_vals, [r["expr_signed_median"] for r in sweep], 1.0),
            "expr_frac_gt1_vs_0.5": flip_theta(
                theta_vals, [r["expr_frac_gt1_median"] for r in sweep], 0.5),
            "topk_signed_median_vs_1": flip_theta(
                theta_vals, [r["topk_signed_median"] for r in sweep], 1.0),
            "topk_frac_gt1_vs_0.5": flip_theta(
                theta_vals, [r["topk_frac_gt1_median"] for r in sweep], 0.5),
        }
        for nm, fl in flips.items():
            log(f"  [flip {nm}] {fl}")

        # Direction AT the fitted theta (the ML solution): over- or under-inflated?
        fit_rec = next(r for r in sweep if r["is_fitted_median"])
        inf_rec = next(r for r in sweep if r["theta"] == "inf")
        direction_at_fitted = {
            "fitted_theta": fitted_median_theta,
            "expr_signed_median": fit_rec["expr_signed_median"],
            "expr_frac_gt1_median": fit_rec["expr_frac_gt1_median"],
            "topk_signed_median": fit_rec["topk_signed_median"],
            "topk_frac_gt1_median": fit_rec["topk_frac_gt1_median"],
            "over_inflated_expr": bool(fit_rec["expr_frac_gt1_median"] > 0.5),
            "over_inflated_topk": bool(fit_rec["topk_frac_gt1_median"] > 0.5),
        }
        log(f"  [direction@fitted theta={fitted_median_theta:.3f}] "
            f"expr frac>1={fit_rec['expr_frac_gt1_median']:.3f} "
            f"topk frac>1={fit_rec['topk_frac_gt1_median']:.3f} "
            f"(>0.5 => decoded variance OVER-inflated)")

        # Energy plateau vs Poisson: is the finite-theta argmin distinguishable
        # from theta->inf given the subsample CI? (Do NOT claim a finite optimum
        # unless the gap exceeds the CI.)
        energy_ci = {("inf" if r["theta"] == "inf" else r["theta"]):
                     (r["energy_ci_mean"], r["energy_ci_sd"]) for r in sweep}
        finite_items = [(t, m, s) for t, (m, s) in energy_ci.items() if t != "inf"]
        argmin_t, argmin_m, argmin_s = min(finite_items, key=lambda x: x[1])
        inf_m, inf_s = energy_ci["inf"]
        sep = abs(argmin_m - inf_m)
        pooled_sd = float(np.hypot(argmin_s, inf_s))
        energy_optimum = {
            "finite_argmin_theta": argmin_t,
            "finite_argmin_energy_mean": argmin_m, "finite_argmin_energy_sd": argmin_s,
            "inf_energy_mean": inf_m, "inf_energy_sd": inf_s,
            "argmin_vs_inf_separation": float(sep),
            "pooled_subsample_sd": pooled_sd,
            "distinguishable_from_poisson": bool(sep > pooled_sd),
            "note": ("finite-theta argmin is WITHIN subsample noise of theta->inf "
                     "-> NOT a strictly finite optimum; energy is on a Poisson-like "
                     "plateau" if sep <= pooled_sd else
                     "finite-theta argmin separates from theta->inf beyond subsample "
                     "noise -> possible finite optimum (verify)"),
        }
        log(f"  [energy optimum] finite_argmin theta={argmin_t} "
            f"energy={argmin_m:.5f}+/-{argmin_s:.5f} vs inf {inf_m:.5f}+/-{inf_s:.5f} "
            f"sep={sep:.5f} pooled_sd={pooled_sd:.5f} "
            f"distinguishable={energy_optimum['distinguishable_from_poisson']}")

        # Paired Wilcoxon: learned-theta NB vs Poisson-on-nb-mean (theta=inf).
        pois_map = maps_fixed["inf"]
        pw = {
            "energy_distance": paired(map_nb, pois_map, "energy_distance"),
            "r_var_abs_log2": paired(map_nb, pois_map, "r_var_abs_log2"),
        }
        for m, r in pw.items():
            log(f"  [paired nb_stochastic vs poisson_on_nb_mean | {m}] "
                f"n={r['n_pairs']} medDiff={r['median_paired_diff']:+.5f} "
                f"nb_win={r['nb_stochastic_win_fraction']:.3f} p={r['p_value']:.2e}")

        per_seed[str(seed)] = {
            "gate": {
                "det_log_pearson": float(det_log_pearson),
                "nb_mean_log_pearson": float(nb_log_pearson),
                "gap_vs_det": float(gap), "gate_passed": gate_passed,
                "nb_val_nll": float(nb_val), "nb_params": int(nb_np),
                "logtheta_stats": lt_stats,
                "fitted_median_theta": fitted_median_theta,
            },
            "nb_stochastic_learned": {
                "r_var_signed_median": agg_nb["r_var_signed"]["median"],
                "expr_signed_median": agg_nb["r_var_signed_expressed"]["median"],
                "expr_frac_gt1_median": agg_nb["r_var_frac_gt1_expressed"]["median"],
                "topk_signed_median": agg_nb["r_var_signed_topk"]["median"],
                "topk_frac_gt1_median": agg_nb["r_var_frac_gt1_topk"]["median"],
                "r_var_abs_log2_median": agg_nb["r_var_abs_log2"]["median"],
                "energy_distance_median": agg_nb["energy_distance"]["median"],
            },
            "theta_sweep": sweep,
            "restricted_direction_flips": flips,
            "direction_at_fitted_theta": direction_at_fitted,
            "energy_optimum": energy_optimum,
            "paired_nb_stochastic_vs_poisson_on_nb_mean": pw,
        }

    out = {
        "purpose": "Phase 2d E1 theta-sweep on pca_512 (3 seeds): with NB's own "
                   "fitted count mean HELD FIXED, impose a scalar dispersion theta "
                   "over a grid incl. inf (Poisson) and measure signed r_var, "
                   "|log2 r_var|, energy_distance per theta to evidence/refute the "
                   "§15 over-inflation mechanism. NO retraining per theta.",
        "encoder": "pca_512", "seeds": SEEDS,
        "fixed_theta_grid": [("inf" if np.isinf(t) else t) for t in FIXED_THETAS],
        "note_fitted_median_inserted_per_seed": True,
        "acceptance_gate": {
            "rule": "ONE-SIDED underfit guard: NB mean log1p Pearson not WORSE than "
                    "deterministic MSE-on-log1p baseline by > %.2f (same seed)." % GATE_TOL,
            "tol": GATE_TOL, "sided": "one_sided_underfit",
        },
        "metric_scale": "all metrics on log1p(counts); real=logX[te]=log1p(raw[te]).",
        "energy_subsample": 300, "energy_ci_count_draws": 5,
        "energy_ci_note": ("energy CI is over 5 independent COUNT-sample draws "
                           "(the dominant noise source; the energy subsample is a "
                           "no-op for most groups on this panel). A finite-theta "
                           "optimum is claimed distinguishable from Poisson only if "
                           "the argmin-vs-inf gap exceeds this CI's pooled sd."),
        "direction_metric_note": (
            "Unrestricted signed r_var median is pinned at ~1.0 by the sparse count "
            "panel (majority of genes have within-type mean count << 1; NB variance "
            "mu+mu^2/theta barely moves them, discrete low counts pin ratios at 1) and "
            "cannot localize a direction flip -- reported to SHOW the degeneracy. "
            "Direction is measured on expression-restricted stats: genes with real "
            "within-type mean count >= 0.5 ('expressed') and top-200 by real within-type "
            "variance ('topk'), reporting median signed ratio, mean log2-ratio, and "
            "fraction of genes with ratio>1 (the point-mass-robust direction statistic)."),
        "per_seed": per_seed,
    }
    (RESULTS / "e1_theta_sweep.json").write_text(json.dumps(out, indent=2))
    log(f"=== E1 done in {time.time()-t0:.0f}s -> {RESULTS / 'e1_theta_sweep.json'} ===")


if __name__ == "__main__":
    main()
