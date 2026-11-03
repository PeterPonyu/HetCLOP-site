"""Phase 2j -- theta-conditionality confound probe (coordinator follow-up).

Question: when scVI's amortized (``dispersion="gene-cell"``) mode makes NB beat
Poisson (the reversal observed in the main run), is that because the per-cell
dispersion is CONDITIONAL on the cell (theta_ic varies cell-to-cell), or merely
because that arm's mean is worse (0.549 vs 0.629 Pearson) -- a deeper decoded
variance deficit that ANY extra dispersion, global or amortized, would fill?

Proposed 2x2 law (coordinator): NB only wins where there is BOTH (a) a real
variance deficit under Poisson (signed r_var < 1) AND (b) a conditional
(per-cell) dispersion. If NB's advantage survives collapsing theta_ic to a
per-gene GLOBAL value (same mu, same deficit, no conditionality) then the deficit
alone explains the reversal and conditionality is irrelevant. If the global-theta
draw does NOT beat Poisson (or beats it far less than the amortized draw does),
conditionality is doing real work.

THE PROBE (mu held fixed in every arm -- identical logic to poisson_on_nb_mean):
Using scVI's ALREADY-TRAINED gene-cell model's own mu_ic (count-space mean) and
its amortized theta_ic, build three draws that share IDENTICAL mu:
  (i)   poisson_on_mean   -- Poisson(mu_ic)                         [theta -> inf]
  (ii)  nb_global_theta   -- NB(mu_ic, theta_g), theta_g = per-gene MEDIAN over
                             cells of theta_ic (collapses conditionality, keeps
                             the same "typical" per-gene dispersion magnitude)
  (iii) nb_amortized      -- NB(mu_ic, theta_ic)                    [the real model]

Deviation from the coordinator's literal suggestion, stated explicitly:
``sample_nb_fixed_theta`` (src/architecture/stochastic_decoder.py) only accepts a
true SCALAR (or inf) theta -- it does ``float(theta)`` internally, which raises on
an array. theta_g here is a PER-GENE vector (2500 values), not one number for the
whole panel, so (ii) is drawn with ``sample_nb`` instead, passing theta_g
broadcast to (n_cells, n_genes). This is the mathematically identical Gamma-Poisson
mixture ``sample_nb_fixed_theta`` implements (``lam ~ Gamma(theta, mu/theta), y ~
Poisson(lam)``); the only difference is that ``sample_nb`` accepts an array shape
argument for ``theta`` where ``sample_nb_fixed_theta`` enforces a Python scalar.
No behaviour is altered by using the array-capable function instead.

"NO RETRAINING" caveat, stated explicitly: the main run (run_scvi.py) computed
mu_ic/theta_ic for scvi_gene_cell_seed0 in memory and only persisted the
AGGREGATE metrics to scvi.json, not the raw per-cell arrays -- there was no
checkpoint saved to re-load them from. Extracting mu_ic/theta_ic for this probe
therefore requires one re-fit of scVI with dispersion="gene-cell" at the SAME
seed/config already used in the main run (same scvi.settings.seed, same
setup_anndata/batch_key, same defaults) -- not a new model, not a new
architecture, not any hyperparameter change. This is stated loudly per the
boundary/honesty rules; it is unavoidable without a saved checkpoint.

GPU
discipline: run only after the main scVI/CFGen run has finished and the GPU is
re-polled free.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import phase2j_lib as L  # noqa: E402
from run_scvi import fit_and_extract  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from clustered_inference import (  # noqa: E402
    clustered_dataset_medians, dataset_signflip_permutation, naive_paired,
    paired_group_diffs, within_dataset_signflip_permutation,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def _clustered(records):
    nv = naive_paired(records)
    cl = clustered_dataset_medians(records)
    perm = dataset_signflip_permutation(records, n_perm=10000, seed=0)
    wperm = within_dataset_signflip_permutation(records, n_perm=10000, seed=0)
    diffs = np.array([r["diff"] for r in records], dtype=np.float64)
    return {
        "n_groups": int(diffs.size),
        "b_win_fraction_groups": float(np.mean(diffs > 0)),  # diff=a-b; b wins if a>b
        "median_paired_diff_a_minus_b": float(np.median(diffs)),
        "naive_p": nv["p_value"],
        "clustered_wilcoxon_p": cl["wilcoxon_p"],
        "clustered_sign_p": cl["sign_test_p"],
        "clustered_n_datasets": cl["n_datasets"],
        "dataset_signflip_perm_p": perm["p_value"],
        "within_dataset_signflip_perm_p": wperm["p_value"],
    }


def run_one_seed(data, det_log_pearson, seed, n_param_samples, device):
    L.log(f"=== theta probe: fitting scVI gene-cell seed={seed} (re-fit; see NO "
          f"RETRAINING caveat in module docstring -- raw mu/theta were not "
          f"persisted from the main run) ===")
    mu_ic, theta_ic, meta = fit_and_extract(
        data, "gene-cell", seed, n_param_samples, None, device)
    mu_ic = np.clip(np.asarray(mu_ic, dtype=np.float64), 0.0, None)
    theta_ic = np.clip(np.asarray(theta_ic, dtype=np.float64), 1e-8, None)

    theta_g = np.median(theta_ic, axis=0)  # per-gene GLOBAL theta (collapses conditionality)
    L.log(f"  theta_g (per-gene median over cells): "
          f"min={theta_g.min():.4g} med={np.median(theta_g):.4g} max={theta_g.max():.4g}")

    # Reproduce the main run's mean-fit gate record for context (informational
    # only; mu is IDENTICAL across all three arms below, so the gate cannot
    # confound this contrast -- same logic as poisson_on_nb_mean).
    from decoder_eval import reconstruction_metrics
    mean_log = _log1p(mu_ic)
    logX, te = data["logX"], data["te"]
    real_te = logX[te]
    model_log_pearson = float(reconstruction_metrics(mean_log, real_te)["mean_gene_pearson"])
    gap = det_log_pearson - model_log_pearson
    gate_passed = mean_acceptance_gate(det_log_pearson, model_log_pearson, L.GATE_TOL)
    L.log(f"  mean_log1p_r={model_log_pearson:.4f} (det={det_log_pearson:.4f} "
          f"gap={gap:+.4f}) gate={'PASS' if gate_passed else 'FAIL'} "
          f"[informational; mu identical in all 3 arms below]")

    # Three draws, IDENTICAL mu_ic in every arm.
    pois = sample_nb_fixed_theta(mu_ic, float("inf"), seed=seed)             # (i)
    theta_g_bc = np.broadcast_to(theta_g, mu_ic.shape).copy()
    nb_global = sample_nb(mu_ic, theta_g_bc, seed=seed)                      # (ii)
    nb_amort = sample_nb(mu_ic, theta_ic, seed=seed)                        # (iii)
    decoded = {
        "poisson_on_mean": _log1p(pois),
        "nb_global_theta": _log1p(nb_global),
        "nb_amortized": _log1p(nb_amort),
    }

    labels, dataset_ids = data["labels"], data["dataset_ids"]
    lab_te, ds_te = labels[te], dataset_ids[te]
    aggs, agg_full = {}, {}
    for vname, dec in decoded.items():
        agg = aggregate_over_groups(real_te, dec, lab_te, ds_te,
                                    min_cells=20, energy_subsample=300,
                                    top_k=200, seed=0)
        aggs[vname] = per_group_map(agg)
        agg_full[vname] = agg
    n_groups = agg_full["poisson_on_mean"]["n_groups"]

    contrasts = {}
    pairs = [
        ("nb_amortized_vs_poisson", "nb_amortized", "poisson_on_mean"),
        ("nb_global_vs_poisson", "nb_global_theta", "poisson_on_mean"),
        ("nb_amortized_vs_nb_global", "nb_amortized", "nb_global_theta"),
    ]
    for label, a_name, b_name in pairs:
        for mkey, metric in [("rvar_expressed", L.RVAR_METRIC),
                             ("rvar_topk", L.RVAR_METRIC_TOPK),
                             ("energy", L.ENERGY_METRIC)]:
            recs = paired_group_diffs(aggs[a_name], aggs[b_name], metric)
            key = f"{label}__{mkey}"
            contrasts[key] = {"a": a_name, "b": b_name, "metric": metric,
                              **_clustered(recs)}
            c = contrasts[key]
            L.log(f"    [{key:38s}] b_win_grp={c['b_win_fraction_groups']:.3f} "
                  f"medDiff(a-b)={c['median_paired_diff_a_minus_b']:+.4f} "
                  f"naive_p={c['naive_p']:.2e} clustWil_p={c['clustered_wilcoxon_p']:.2e}")

    direction = {}
    for vname in decoded:
        a = agg_full[vname]
        direction[vname] = {
            "r_var_signed_expr_median": a["r_var_signed_expressed"]["median"],
            "r_var_frac_gt1_expr_median": a["r_var_frac_gt1_expressed"]["median"],
            "r_var_abs_log2_expr_median": a["r_var_abs_log2_expressed"]["median"],
            "energy_median": a["energy_distance"]["median"],
        }
    L.log(f"  [direction] poisson signed_expr="
          f"{direction['poisson_on_mean']['r_var_signed_expr_median']:.3f} | "
          f"nb_global signed_expr={direction['nb_global_theta']['r_var_signed_expr_median']:.3f} | "
          f"nb_amortized signed_expr={direction['nb_amortized']['r_var_signed_expr_median']:.3f}")

    return {
        "seed": seed, "n_groups": int(n_groups),
        "mu_identical_across_arms": True,
        "gate_informational": {
            "det_log_pearson": det_log_pearson,
            "model_mean_log_pearson": model_log_pearson,
            "gap_vs_det": float(gap), "gate_passed": bool(gate_passed),
        },
        "theta_g_summary": {"min": float(theta_g.min()), "median": float(np.median(theta_g)),
                            "max": float(theta_g.max())},
        "theta_ic_summary": {"min": float(theta_ic.min()), "median": float(np.median(theta_ic)),
                             "max": float(theta_ic.max())},
        "contrasts": contrasts,
        "direction": direction,
        "train_meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="1 (seed0 only) or up to 3")
    ap.add_argument("--n-param-samples", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    L.log(f"theta-conditionality probe; seeds={seeds}")
    data = L.load_corpus()
    det = L.deterministic_baseline_log_pearson(data, seed=42, device=args.device)
    L.log(f"deterministic PCA-512 baseline log1p Pearson = {det:.4f}")

    out = {
        "purpose": "Confound check: does NB beat Poisson under scVI gene-cell "
                   "amortized dispersion because of theta's CELL-CONDITIONALITY, "
                   "or merely because that arm's mean is worse (deeper variance "
                   "deficit)? mu is IDENTICAL in all three arms, so a mean "
                   "deficit cannot confound this contrast.",
        "deviation_from_literal_request": (
            "sample_nb_fixed_theta only accepts a scalar/inf theta; theta_g is a "
            "PER-GENE vector (median over cells of the amortized theta_ic), so "
            "(ii) nb_global_theta was drawn with sample_nb(mu, theta_g_broadcast) "
            "instead -- the identical Gamma-Poisson mixture, just via the "
            "array-capable function. See module docstring."),
        "no_retraining_caveat": (
            "scVI gene-cell was NOT checkpointed in the main run (only aggregate "
            "metrics were persisted), so extracting raw mu_ic/theta_ic required "
            "one re-fit at the SAME seed/config already used -- not a new model "
            "or hyperparameter change. See module docstring."),
        "deterministic_baseline_log_pearson": det, "seeds": seeds,
        "results": {},
    }
    for seed in seeds:
        out["results"][seed] = run_one_seed(data, det, seed, args.n_param_samples, args.device)
        (RESULTS / "scvi_theta_probe.json").write_text(json.dumps(out, indent=2))
    L.log(f"=== theta probe done -> {RESULTS / 'scvi_theta_probe.json'} ===")


if __name__ == "__main__":
    main()
