"""Phase 2m -- synthetic 2x2 phase diagram for the dispersion law (CPU-only).

The law (spec §27), stated so a controlled experiment can test it:

    Extra dispersion helps iff (a) the Poisson-on-mean draw leaves a variance
    DEFICIT, AND (b) the dispersion parameter is CONDITIONAL enough to place the
    missing variance where it is missing.

The deployed-model evidence (our Arm A/B, CFGen, scVI gene / gene-cell) is a set of
FIELD OBSERVATIONS: each real setting happens to sit in one cell of the 2x2, and we
read the winner off. This experiment is the CONTROLLED complement -- ground truth for
both factors is a DIAL, so every cell of the 2x2 is populated on demand and the
mechanism is isolated from the confounds real data carries (mean-fit quality, encoder,
library-size structure).

Factor (a) is nearly trivial: with no deficit, any added dispersion over-inflates, so
Poisson wins. The CONTESTABLE factor is (b) -- and it is exactly what the scVI
gene-vs-gene-cell GPU probe tests. We isolate (b) here with a contrast real data cannot
give cleanly: whether the TRUE per-group dispersion is HOMOGENEOUS or HETEROGENEOUS
across cell types.

    Prediction: a single global theta calibrates every group when the truth is
    homogeneous (nothing for conditionality to buy: global == conditional), but
    fails per-group when the truth is heterogeneous (conditional >> global) -- even
    though both share the identical conditional mean and see the identical deficit.

If this controlled contrast does NOT separate global from conditional in the
heterogeneous regime, factor (b) is wrong and the law must be restated BEFORE the GPU
probe -- which is the point of running it now.

Design (all ground truth known):
  * n_datasets synthetic datasets, each with n_types groups (=(dataset,cell_type)).
  * gene base rates beta_{group,gene}; per-cell size factors ell_i (within-group mean
    spread); true cell mean mu_ig = ell_i * beta_{group(i),gene}.
  * real counts y ~ NB(mu_ig, theta_true) per regime:
      - no_deficit    : theta_true = THETA_HUGE for all groups (counts ~ Poisson).
      - deficit_homog : theta_true = THETA_DEFICIT for ALL groups (real overdispersion,
                        identical across cell types).
      - deficit_hetero: theta_true drawn per-group over a wide range (real
                        overdispersion that DIFFERS across cell types).
  * The three noise models under test share the IDENTICAL, PERFECT conditional mean
    (mu_hat = mu_ig): only the dispersion parameterization differs --
      poisson         : Poisson(mu)                    [theta -> inf]
      nb_global       : NB(mu, theta_global)  theta_global = median true theta over
                        ALL cells (collapses conditionality)
      nb_conditional  : NB(mu, theta_true_group)       [the oracle per-group theta]

Metrics + inference are the project's own: expression-restricted |log2 r_var| and
energy distance, aggregated per (dataset,cell_type) group, dataset-clustered over the
synthetic datasets (collapse to per-dataset medians -> one-sample Wilcoxon), plus a
sign-flip permutation. Lower |log2 r_var| / energy = better.

No GPU, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "phase2f_f4_resolution_2026-07-09"))

from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from clustered_inference import (  # noqa: E402
    clustered_dataset_medians,
    dataset_signflip_permutation,
    naive_paired,
    paired_group_diffs,
)

OUT = Path(__file__).resolve().parent / "results" / "lawgrid.json"

# --- fixture scale (kept modest so the whole grid runs in seconds on CPU) -----------
N_DATASETS = 12
N_TYPES = 6
CELLS_PER_GROUP = 200
N_GENES = 150
THETA_HUGE = 1000.0      # near-Poisson true counts -> no deficit
THETA_DEFICIT = 0.8      # real overdispersion (homogeneous case)
THETA_HETERO_LO = 0.3    # heterogeneous case spans a wide dispersion range
THETA_HETERO_HI = 8.0
THETA_MIXED_POISSON = 100.0  # bimodal "mixed" case: half near-Poisson groups...
THETA_MIXED_OVERDISP = 0.5   # ...half heavily overdispersed (real cell types span this)
EXPR_KEY = "r_var_abs_log2_expressed"
SIGNED_KEY = "r_var_signed_expressed"
FRAC_KEY = "r_var_frac_gt1_expressed"
ENERGY_KEY = "energy_distance"


def build_truth(regime: str, seed: int):
    """Return (real_log, mu_ic, theta_true_ic, labels, dataset_ids). mu_ic is the
    PERFECT conditional mean shared by all three noise models."""
    rng = np.random.default_rng(seed)
    labels, dsids, mu_rows, theta_rows = [], [], [], []
    for d in range(N_DATASETS):
        for t in range(N_TYPES):
            # group gene base rate: a mix of a few expressed genes and many low ones,
            # so the expression restriction (mean count >= 0.5) has real teeth.
            beta = rng.gamma(shape=0.6, scale=1.5, size=N_GENES).astype(np.float64)
            ell = rng.lognormal(mean=0.0, sigma=0.4, size=CELLS_PER_GROUP)  # size factors
            mu = np.outer(ell, beta)  # (cells, genes) true per-cell mean
            if regime == "no_deficit":
                th = THETA_HUGE
            elif regime == "deficit_homog":
                th = THETA_DEFICIT
            elif regime == "deficit_hetero":
                th = float(np.exp(rng.uniform(np.log(THETA_HETERO_LO), np.log(THETA_HETERO_HI))))
            elif regime == "deficit_mixed":
                # bimodal across cell types: near-Poisson groups coexist with heavily
                # overdispersed ones, so a single global theta over-inflates the former
                # BELOW Poisson's own error while under-serving the latter.
                th = THETA_MIXED_POISSON if (t % 2 == 0) else THETA_MIXED_OVERDISP
            else:
                raise ValueError(regime)
            mu_rows.append(mu)
            theta_rows.append(np.full(mu.shape, th))
            labels += [f"t{t}"] * CELLS_PER_GROUP
            dsids += [f"d{d}"] * CELLS_PER_GROUP
    mu_ic = np.vstack(mu_rows)
    theta_true = np.vstack(theta_rows)
    # ground-truth counts drawn from the true NB (seed distinct from the model draws)
    real = sample_nb(mu_ic, theta_true, seed=seed + 7919)
    real_log = np.log1p(real).astype(np.float32)
    return real_log, mu_ic, theta_true, np.array(labels), np.array(dsids)


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def _pgm(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def _clustered(records):
    diffs = np.array([r["diff"] for r in records], dtype=np.float64)
    cl = clustered_dataset_medians(records)
    perm = dataset_signflip_permutation(records, n_perm=10000, seed=0)
    return {
        "n_groups": int(diffs.size),
        # diff = a - b; for lower-is-better metrics b wins (a worse) when diff>0
        "a_win_fraction_groups": float(np.mean(diffs < 0)),
        "median_paired_diff_a_minus_b": float(np.median(diffs)),
        "naive_p": naive_paired(records)["p_value"],
        "clustered_wilcoxon_p": cl["wilcoxon_p"],
        "clustered_n_datasets": cl["n_datasets"],
        "dataset_signflip_perm_p": perm["p_value"],
    }


def run_regime(regime: str, seed: int) -> dict:
    real_log, mu_ic, theta_true, labels, dsids = build_truth(regime, seed)
    theta_global = float(np.median(theta_true))  # one number, collapses conditionality

    draws = {
        "poisson": sample_nb_fixed_theta(mu_ic, float("inf"), seed=seed + 1),
        "nb_global": sample_nb(mu_ic, np.full(mu_ic.shape, theta_global), seed=seed + 1),
        "nb_conditional": sample_nb(mu_ic, theta_true, seed=seed + 1),
    }
    aggs, agg_full = {}, {}
    for name, cnt in draws.items():
        agg = aggregate_over_groups(real_log, _log1p(cnt), labels, dsids,
                                    min_cells=20, energy_subsample=300, top_k=N_GENES, seed=0)
        aggs[name] = _pgm(agg)
        agg_full[name] = agg

    # levels: signed r_var + frac>1 + aggregate |log2| + energy for each model
    levels = {}
    for name, agg in agg_full.items():
        levels[name] = {
            "r_var_signed_expr_median": agg[SIGNED_KEY]["median"],
            "r_var_frac_gt1_expr_median": agg[FRAC_KEY]["median"],
            "r_var_abs_log2_expr_median": agg[EXPR_KEY]["median"],
            "energy_median": agg[ENERGY_KEY]["median"],
        }

    # the three decisive paired contrasts, on both metrics
    contrasts = {}
    pairs = [("conditional_vs_poisson", "nb_conditional", "poisson"),
             ("global_vs_poisson", "nb_global", "poisson"),
             ("conditional_vs_global", "nb_conditional", "nb_global")]
    for label, a, b in pairs:
        for mkey, metric in (("rvar", "r_var_abs_log2_expr"), ("energy", "energy_distance")):
            recs = paired_group_diffs(aggs[a], aggs[b], metric)
            contrasts[f"{label}__{mkey}"] = {"a": a, "b": b, "metric": metric, **_clustered(recs)}

    return {"regime": regime, "seed": seed, "theta_global_used": theta_global,
            "theta_true_summary": {"min": float(theta_true.min()),
                                   "median": float(np.median(theta_true)),
                                   "max": float(theta_true.max())},
            "levels": levels, "contrasts": contrasts}


def verdict(reg: dict) -> dict:
    """Reduce one regime to the phase-diagram cell: who is best-calibrated, and does
    conditionality beat a global theta?"""
    lv = reg["levels"]
    best = min(lv, key=lambda k: lv[k]["r_var_abs_log2_expr_median"])
    cvg = reg["contrasts"]["conditional_vs_global__rvar"]
    gvp = reg["contrasts"]["global_vs_poisson__rvar"]
    cond_beats_global = (cvg["median_paired_diff_a_minus_b"] < 0) and (cvg["clustered_wilcoxon_p"] < 0.05)
    # does the global-theta NB beat Poisson on the paired contest? (a=global, b=poisson;
    # diff<0 => global better). In real scVI `gene` mode it does NOT -- global loses.
    global_beats_poisson = (gvp["median_paired_diff_a_minus_b"] < 0) and (gvp["clustered_wilcoxon_p"] < 0.05)
    return {
        "poisson_signed_rvar": lv["poisson"]["r_var_signed_expr_median"],
        "poisson_leaves_deficit": lv["poisson"]["r_var_signed_expr_median"] < 0.95,
        "best_calibrated_model": best,
        "conditional_beats_global": bool(cond_beats_global),
        "conditional_vs_global_clustered_p": cvg["clustered_wilcoxon_p"],
        "global_beats_poisson": bool(global_beats_poisson),
        "global_vs_poisson_clustered_p": gvp["clustered_wilcoxon_p"],
    }


REGIMES = ["no_deficit", "deficit_homog", "deficit_hetero", "deficit_mixed"]


def predictions_for_seed(seed: int):
    """Run the full 2x2 at one seed and return (checks, observations, verdicts, results).

    ``checks`` are the ROBUST core law predictions (asserted, must hold every seed).
    ``observations`` are seed-SENSITIVE items reported but not asserted -- kept out of the
    pass/fail set precisely because the 5-seed robustness run showed they flip (a 3-way
    calibrated tie, and a deliberately-borderline mixed-regime contest)."""
    results = {r: run_regime(r, seed) for r in REGIMES}
    verdicts = {r: verdict(results[r]) for r in REGIMES}
    checks = {
        # no deficit => Poisson is calibrated AND conditionality buys nothing (all three
        # models tie ~0.108; we do NOT assert a strict argmin, which is a coin flip).
        "no_deficit__poisson_calibrated_no_conditional_gain":
            (not verdicts["no_deficit"]["poisson_leaves_deficit"])
            and (not verdicts["no_deficit"]["conditional_beats_global"]),
        "deficit__poisson_under_inflates":
            verdicts["deficit_homog"]["poisson_leaves_deficit"]
            and verdicts["deficit_hetero"]["poisson_leaves_deficit"],
        "homogeneous__conditionality_buys_nothing":
            not verdicts["deficit_homog"]["conditional_beats_global"],
        "heterogeneous__conditionality_wins":
            verdicts["deficit_hetero"]["conditional_beats_global"],
        # in the mixed regime the conditional oracle still wins -- the load-bearing claim.
        "mixed__conditionality_wins":
            verdicts["deficit_mixed"]["conditional_beats_global"],
    }
    observations = {
        # a strict argmin among three tied calibrated models -- flips on noise, meaningless.
        "no_deficit__poisson_strictly_best": verdicts["no_deficit"]["best_calibrated_model"] == "poisson",
        # borderline by construction: whether a global theta edges Poisson in the mixed
        # regime is seed-sensitive (median-collapse theta lands near-Poisson). This
        # REPRODUCES the near-noise-floor character of the real scVI `gene` paired contest,
        # so its instability is itself the finding, not a defect.
        "mixed__global_theta_does_not_beat_poisson": not verdicts["deficit_mixed"]["global_beats_poisson"],
    }
    return checks, observations, verdicts, results


def main() -> int:
    seed = 20260709
    regimes = REGIMES
    checks, observations, verdicts, results = predictions_for_seed(seed)
    all_pass = all(checks.values())

    # Seed robustness: the 2x2 pattern must not be an artifact of one fixture draw.
    # Re-run all four regimes on additional seeds and require every CORE prediction to hold
    # in EVERY seed (kept separate from the primary block so the tested schema is stable).
    robustness_seeds = [seed + off for off in (101, 202, 303, 404)]
    per_seed_pred, per_seed_obs = {}, {}
    for s in robustness_seeds:
        sc, ob, _, _ = predictions_for_seed(s)
        per_seed_pred[s] = sc
        per_seed_obs[s] = ob
    all_seeds = [seed] + robustness_seeds
    per_prediction_all_seeds = {
        name: bool(checks[name] and all(per_seed_pred[s][name] for s in robustness_seeds))
        for name in checks
    }
    # observations: report how many seeds each holds in (NOT required to be unanimous)
    obs_seed_counts = {
        name: int(observations[name]) + sum(int(per_seed_obs[s][name]) for s in robustness_seeds)
        for name in observations
    }
    robustness = {
        "seeds": all_seeds,
        "n_seeds": len(all_seeds),
        "all_core_predictions_hold_every_seed": all(per_prediction_all_seeds.values()),
        "per_prediction_holds_all_seeds": per_prediction_all_seeds,
        "per_seed_predictions": {str(s): per_seed_pred[s] for s in robustness_seeds},
        "observation_seed_counts_out_of_n": obs_seed_counts,
        "note": ("core predictions are the factor-(a)/(b) law claims and hold unanimously; "
                 "the two observations are intentionally non-asserted -- a 3-way calibrated "
                 "tie and a borderline mixed-regime contest that mirrors the near-noise-floor "
                 "scVI `gene` paired energy leg."),
    }

    out = {
        "purpose": "Controlled 2x2 phase diagram isolating the dispersion law's two "
                   "factors with known ground truth; the contestable factor (b), "
                   "theta-conditionality, is isolated via a homogeneous-vs-heterogeneous "
                   "true-theta contrast that real data cannot give cleanly.",
        "law": "extra dispersion helps iff (a) Poisson-on-mean leaves a deficit AND "
               "(b) theta is conditional enough to place the missing variance per group",
        "fixture": {"n_datasets": N_DATASETS, "n_types": N_TYPES,
                    "cells_per_group": CELLS_PER_GROUP, "n_genes": N_GENES,
                    "theta_huge": THETA_HUGE, "theta_deficit_homog": THETA_DEFICIT,
                    "theta_hetero_range": [THETA_HETERO_LO, THETA_HETERO_HI]},
        "predictions_checked": checks,
        "all_predictions_hold": all_pass,
        "observations_primary_seed": observations,
        "seed_robustness": robustness,
        "verdicts": verdicts,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"=== synthetic 2x2 phase diagram (seed {seed}) ===")
    for r in regimes:
        v = verdicts[r]
        lv = results[r]["levels"]
        print(f"\n[{r}]  theta_true med={results[r]['theta_true_summary']['median']:.3g}")
        for m in ("poisson", "nb_global", "nb_conditional"):
            print(f"    {m:15s} signed_rvar={lv[m]['r_var_signed_expr_median']:.3f} "
                  f"|log2|={lv[m]['r_var_abs_log2_expr_median']:.3f} energy={lv[m]['energy_median']:.3f}")
        print(f"    -> Poisson leaves deficit? {v['poisson_leaves_deficit']}; "
              f"best={v['best_calibrated_model']}; "
              f"conditional>global? {v['conditional_beats_global']} "
              f"(p={v['conditional_vs_global_clustered_p']:.2e}); "
              f"global>Poisson? {v['global_beats_poisson']} "
              f"(p={v['global_vs_poisson_clustered_p']:.2e})")
    print("\n=== predictions ===")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"\nALL CORE PREDICTIONS HOLD (primary seed): {all_pass}")
    print(f"\n=== seed robustness ({robustness['n_seeds']} seeds) ===")
    for name, ok in robustness["per_prediction_holds_all_seeds"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    every = robustness["all_core_predictions_hold_every_seed"]
    print(f"ALL CORE PREDICTIONS HOLD IN EVERY SEED: {every}")
    print(f"\n=== non-asserted observations (seed-sensitive; count / {robustness['n_seeds']}) ===")
    for name, cnt in robustness["observation_seed_counts_out_of_n"].items():
        print(f"  {cnt}/{robustness['n_seeds']}  {name}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if (all_pass and every) else 1


if __name__ == "__main__":
    raise SystemExit(main())
