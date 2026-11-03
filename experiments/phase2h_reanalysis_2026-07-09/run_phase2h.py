"""Phase 2h reanalysis orchestrator (CPU-only; recomputation from persisted arrays).

T1 -- leave-two-out robustness for the two species-misnamed (actually mouse)
      datasets, recomputing F4's load-bearing paired inference with and without
      them, per leg, from phase2f.json per-group arrays.
T2 -- paper-level multiplicity accounting: enumerate every reported hypothesis
      test, apply paper-level Benjamini-Hochberg FDR and paper-level Holm.

Nothing is retrained. All quantities are recomputed from:
  * phase2f.json                (E6 naive_p; E7 per_group arrays)
  * nb_arm.json                 (pca energy raw/Holm; family pca+scgpt)
  * e2_nb_generalization.json   (randproj energy raw/Holm)
  * e5_expressed_rvar.json      (pca/randproj expressed r_var raw/Holm)

Standing-rule guardrails enforced here:
  * SANITY ASSERT: recomputing the FULL 50-dataset tests from the per-group
    arrays reproduces phase2f.json's stored naive + clustered + permutation
    values to <=1e-6 (permutation stats exactly). Fails loudly otherwise.
  * Paired data only -- never conclude from marginal medians.
  * Permutation statistic = Wilcoxon signed-rank sum of per-dataset medians.

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from leave_two_out import (  # noqa: E402
    GATE_ENCODERS,
    METRICS,
    MISNAMED_MOUSE_DATASETS,
    SEEDS,
    drop_datasets,
    holm_family_over_encoders,
    load_per_comparison,
    recompute_leg,
)
from multiplicity import apply_corrections  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

P2F = REPO / "experiments/phase2f_f4_resolution_2026-07-09/results/phase2f.json"
NBA = REPO / "experiments/phase2c_robustness_2026-07-09/results/nb_arm.json"
E2J = REPO / "experiments/phase2d_completion_2026-07-09/results/e2_nb_generalization.json"
E5J = REPO / "experiments/phase2e_expressed_rvar_2026-07-09/results/e5_expressed_rvar.json"

TOL = 1e-6


# ---------------------------------------------------------------------------
# T1 -- sanity assert + leave-two-out
# ---------------------------------------------------------------------------
def sanity_assert(idx, stored):
    """Recompute FULL-corpus naive + clustered + permutation from per_group and
    assert they reproduce phase2f.json's stored values. Fail loudly."""
    failures = []
    checks = 0
    for key, records in idx.items():
        enc, seed, metric = key
        st = stored[key]
        got = recompute_leg(records)
        # naive
        exp_p = st["naive_314group"]["p"]
        if abs(got["naive_paired"]["p_value"] - exp_p) > TOL:
            failures.append(f"naive p {key}: got {got['naive_paired']['p_value']} exp {exp_p}")
        checks += 1
        # clustered wilcoxon + sign
        cl = st["clustered_50dataset"]
        if abs(got["clustered_dataset_medians"]["wilcoxon_p"] - cl["wilcoxon_p"]) > TOL:
            failures.append(f"clustered wilcoxon p {key}")
        if abs(got["clustered_dataset_medians"]["sign_test_p"] - cl["sign_test_p"]) > TOL:
            failures.append(f"clustered sign p {key}")
        checks += 2
        # permutation observed stat (exact) + p (same seed/n_perm)
        ps = st["permutation_dataset_signflip"]
        if abs(got["permutation_dataset_signflip"]["observed_stat"] - ps["observed_stat"]) > TOL:
            failures.append(f"perm stat {key}")
        if abs(got["permutation_dataset_signflip"]["p_value"] - ps["p"]) > TOL:
            failures.append(f"perm p {key}: got {got['permutation_dataset_signflip']['p_value']} exp {ps['p']}")
        checks += 2
    if failures:
        raise AssertionError(
            f"SANITY ASSERT FAILED ({len(failures)} mismatches):\n" + "\n".join(failures[:20])
        )
    return {"n_checks": checks, "max_tol": TOL, "status": "PASS"}


def leave_two_out(idx):
    """For each (metric, encoder, seed): recompute with all 50 datasets and with
    the two misnamed datasets removed (48). Holm within {pca,randproj} per
    (metric, seed). Report per-leg worst-across-seed Holm p and clustered p range.
    """
    per_test = []  # atomic rows
    # accumulate raw p per (metric, seed, variant) -> {encoder: p} for Holm family
    naive_raw = {}
    clustered = {}
    for metric in METRICS:
        for seed in SEEDS:
            for variant, drop in (("full50", ()), ("leave2out48", MISNAMED_MOUSE_DATASETS)):
                for enc in GATE_ENCODERS:
                    recs = idx[(enc, seed, metric)]
                    recs = drop_datasets(recs, drop) if drop else recs
                    res = recompute_leg(recs)
                    nds = res["clustered_dataset_medians"]["n_datasets"]
                    row = {
                        "metric": metric, "encoder": enc, "seed": seed, "variant": variant,
                        "n_datasets": nds,
                        "n_groups": res["naive_paired"]["n_groups"],
                        "naive_p": res["naive_paired"]["p_value"],
                        "naive_nb_win_fraction": res["naive_paired"]["nb_win_fraction"],
                        "naive_median_paired_diff": res["naive_paired"]["median_paired_diff"],
                        "clustered_wilcoxon_p": res["clustered_dataset_medians"]["wilcoxon_p"],
                        "clustered_sign_p": res["clustered_dataset_medians"]["sign_test_p"],
                        "clustered_nb_win_fraction_datasets": res["clustered_dataset_medians"]["nb_win_fraction_datasets"],
                        "perm_dataset_signflip_p": res["permutation_dataset_signflip"]["p_value"],
                        "perm_within_dataset_signflip_p": res["permutation_within_dataset_signflip"]["p_value"],
                    }
                    per_test.append(row)
                    naive_raw.setdefault((metric, seed, variant), {})[enc] = res["naive_paired"]["p_value"]
                    clustered.setdefault((metric, seed, variant), {})[enc] = res["clustered_dataset_medians"]["wilcoxon_p"]
    # Holm within the 2-encoder family per (metric, seed, variant)
    for row in per_test:
        fam = naive_raw[(row["metric"], row["seed"], row["variant"])]
        row["naive_holm_p"] = holm_family_over_encoders(fam)[row["encoder"]]

    # Per-leg summary (metric, encoder): worst Holm across seeds + clustered p range
    legs = {}
    for metric in METRICS:
        for enc in GATE_ENCODERS:
            leg = {}
            for variant in ("full50", "leave2out48"):
                rows = [r for r in per_test if r["metric"] == metric and r["encoder"] == enc and r["variant"] == variant]
                leg[variant] = {
                    "max_naive_holm_p": max(r["naive_holm_p"] for r in rows),
                    "max_naive_raw_p": max(r["naive_p"] for r in rows),
                    "max_clustered_wilcoxon_p": max(r["clustered_wilcoxon_p"] for r in rows),
                    "max_clustered_sign_p": max(r["clustered_sign_p"] for r in rows),
                    "max_perm_dataset_signflip_p": max(r["perm_dataset_signflip_p"] for r in rows),
                    "n_datasets": rows[0]["n_datasets"],
                    "min_naive_nb_win_fraction": min(r["naive_nb_win_fraction"] for r in rows),
                    "max_naive_nb_win_fraction": max(r["naive_nb_win_fraction"] for r in rows),
                    "seeds_naive_holm_sig_0.05": sum(r["naive_holm_p"] <= 0.05 for r in rows),
                    "seeds_clustered_wilcoxon_sig_0.05": sum(r["clustered_wilcoxon_p"] <= 0.05 for r in rows),
                }
            legs[f"{metric}__{enc}"] = leg
    return per_test, legs


# ---------------------------------------------------------------------------
# T2 -- hypothesis inventory + paper-level multiplicity
# ---------------------------------------------------------------------------
def build_inventory(idx):
    """Enumerate every hypothesis test the manuscript reports for the F4
    dispersion contrast (control = poisson_on_nb_mean), plus the E6 monotonicity.

    raw p is taken uniformly from phase2f E6 naive_p (which reproduces nb_arm /
    e2 / e5 raw values exactly -- verified). "current Holm p" is the value the
    manuscript actually reports, pulled from its own source JSON and family:
      * energy pca / scgpt : nb_arm.json p_holm (family = {pca,scgpt} within seed)
      * energy randproj     : e2_nb_generalization.json p_holm
      * energy geneformer   : not Holm-corrected in ms (reported as effect + clustered)
      * r_var_expr pca/rp   : e5_expressed_rvar.json p_holm (family = {pca,rp} within seed)
      * r_var_expr scgpt/gf : not Holm-corrected in ms (reported as effect in tab:dispersion)
    """
    p2f = json.load(open(P2F))
    nba = json.load(open(NBA))
    e2 = json.load(open(E2J))
    e5 = json.load(open(E5J))

    # raw p per (enc, seed, metric) from E6
    e6 = {}
    for r in p2f["E6_four_encoder_dispersion"]["per_encoder_seed"]:
        e6[(r["encoder"], int(r["seed"]))] = {
            "energy_distance": r["energy_distance"]["naive_p"],
            "r_var_abs_log2_expr": r["r_var_abs_log2_expr"]["naive_p"],
            "mean_fit_gap": r["mean_fit_gap"],
            "gate_passed": r["gate_passed"],
        }

    # current Holm lookups
    def holm_energy(enc, seed):
        if enc in ("pca_512", "scgpt_512"):
            k = "energy_distance__nb_stochastic_vs_poisson_on_nb_mean"
            for r in nba["paired_tests"][str(seed)][k]["per_encoder"]:
                if r["encoder"] == enc:
                    return r["p_holm"], "nb_arm.json (family {pca,scgpt} within seed)"
        if enc == "randproj_512":
            for c in e2["results"]["randproj_512"]["paired"][str(seed)]["energy_distance"]:
                if c["control"] == "poisson_on_nb_mean":
                    return c["p_holm"], "e2_nb_generalization.json"
        return None, "not Holm-corrected in manuscript (reported as effect/clustered)"

    def holm_rvar(enc, seed):
        if enc in ("pca_512", "randproj_512"):
            for e in e5["nb_arm"]["paired"]["r_var_abs_log2_expr"]["poisson_on_nb_mean"]["per_seed"]:
                if e["seed"] == seed:
                    for r in e["per_encoder"]:
                        if r["encoder"] == enc:
                            return r["p_holm"], "e5_expressed_rvar.json (family {pca,randproj} within seed)"
        return None, "not Holm-corrected in manuscript (reported as effect in tab:dispersion)"

    encoders = ["pca_512", "randproj_512", "scgpt_512", "geneformer"]
    inv = []
    tid = 0
    for metric in ("energy_distance", "r_var_abs_log2_expr"):
        for enc in encoders:
            for seed in SEEDS:
                raw = e6[(enc, seed)][metric]
                if metric == "energy_distance":
                    cur_holm, src = holm_energy(enc, seed)
                else:
                    cur_holm, src = holm_rvar(enc, seed)
                tid += 1
                inv.append({
                    "test_id": f"H{tid:02d}",
                    "family": "F4_dispersion_contrast",
                    "metric": metric,
                    "control": "poisson_on_nb_mean",
                    "encoder": enc,
                    "seed": seed,
                    "gate_passed": e6[(enc, seed)]["gate_passed"],
                    "inference": "naive_paired_wilcoxon",
                    "raw_p": float(raw),
                    "current_holm_p": (float(cur_holm) if cur_holm is not None else None),
                    "current_holm_source": src,
                    "reported_in": ("tab:nb" if enc in GATE_ENCODERS else "tab:dispersion"),
                })

    # E6 monotonicity test (energy effect vs mean-fit gap across 4 encoders).
    # Honest p per defect #12: exact permutation over 4! orderings, two-sided.
    tid += 1
    inv.append({
        "test_id": f"H{tid:02d}",
        "family": "F4_mechanism_monotonicity",
        "metric": "energy_effect_vs_mean_fit_gap",
        "control": "n/a (rank correlation across 4 encoders)",
        "encoder": "all4",
        "seed": "pooled",
        "gate_passed": None,
        "inference": "exact_permutation_spearman_4orderings_two_sided",
        "raw_p": 0.083,  # 2/24, honest two-sided value (defect #12); Pearson-r p=0.020
        "current_holm_p": None,
        "current_holm_source": "reported as suggestive-not-conclusive (defect #12)",
        "reported_in": "prose + tab:dispersion",
    })
    return inv


def paper_level_correction(inv):
    """Apply paper-level BH-FDR and paper-level Holm across the full inventory
    (raw p). Also apply to the CLUSTERED-primary set for the gate-passing legs.
    """
    corrected = apply_corrections(inv, p_key="raw_p")
    return corrected


def main():
    idx, stored = load_per_comparison(P2F)

    # T1
    sanity = sanity_assert(idx, stored)
    per_test, legs = leave_two_out(idx)
    t1 = {
        "purpose": "Leave-two-out robustness for two species-misnamed (mouse) datasets.",
        "misnamed_mouse_datasets": list(MISNAMED_MOUSE_DATASETS),
        "sanity_assert": sanity,
        "gate_encoders": list(GATE_ENCODERS),
        "metrics": list(METRICS),
        "seeds": list(SEEDS),
        "per_test": per_test,
        "per_leg_summary": legs,
        "f2_encoder_span_note": (
            "F2's encoder-span reconstruction Pearson (FtwoSpan etc.) is sourced from "
            "phase2a_decoder_2026-07-08/cache/armB_fm_probe.json, aggregated per-encoder "
            "with NO per-dataset breakdown (no dataset/GSE keys). The provided per-group "
            "arrays hold NB-vs-control paired diffs, not per-encoder held-out reconstruction "
            "Pearson broken out by dataset. Therefore F2's encoder-span CANNOT be recomputed "
            "leave-two-out from the available arrays. Stated rather than guessed."
        ),
    }
    with open(RESULTS / "t1_leave_two_out.json", "w") as fh:
        json.dump(t1, fh, indent=1)

    # T2
    inv = build_inventory(idx)
    corrected = paper_level_correction(inv)
    n = len(corrected)
    survive_bh = [c for c in corrected if c["survives_bh_0.05"]]
    survive_holm = [c for c in corrected if c["survives_holm_0.05"]]
    fail_bh = [c for c in corrected if not c["survives_bh_0.05"]]
    fail_holm = [c for c in corrected if not c["survives_holm_0.05"]]
    t2 = {
        "purpose": "Paper-level multiplicity accounting (BH-FDR and Holm) across all reported tests.",
        "inventory_size": n,
        "families": sorted(set(c["family"] for c in corrected)),
        "correction_note": (
            "raw_p is the naive paired-Wilcoxon two-sided p (phase2f E6 naive_p, which "
            "reproduces nb_arm/e2/e5 raw p exactly). Paper-level BH and Holm are applied "
            "across the full inventory. The clustered/primary inference for the two "
            "gate-passing legs is checked separately (see f4_carrying_leg / demoted_energy_leg)."
        ),
        "inventory": corrected,
        "n_survive_bh_0.05": len(survive_bh),
        "n_survive_holm_0.05": len(survive_holm),
        "tests_failing_bh_0.05": [(c["test_id"], c["metric"], c["encoder"], c["seed"], c["raw_p"], c["bh_q_paper"]) for c in fail_bh],
        "tests_failing_holm_0.05": [(c["test_id"], c["metric"], c["encoder"], c["seed"], c["raw_p"], c["holm_p_paper"]) for c in fail_holm],
    }

    # Item 3: F4-carrying leg (rvar_expr, gate encoders, dataset-clustered) survival,
    # and demoted energy leg (pca energy). Use worst-across-seed clustered p, then
    # place it in the paper-level BH/Holm ranking as a sanity check.
    rvar_gate = [r for r in per_test if r["metric"] == "r_var_abs_log2_expr"
                 and r["encoder"] in GATE_ENCODERS and r["variant"] == "full50"]
    energy_pca = [r for r in per_test if r["metric"] == "energy_distance"
                  and r["encoder"] == "pca_512" and r["variant"] == "full50"]
    # Build a clustered-primary inventory: gate legs use clustered_wilcoxon_p; add the
    # naive FM/pca-energy legs as-is for a like-for-like paper-level correction.
    clustered_inv = []
    for r in per_test:
        if r["variant"] != "full50":
            continue
        clustered_inv.append({
            "test_id": f"C_{r['metric']}_{r['encoder']}_{r['seed']}",
            "family": "F4_clustered_primary",
            "metric": r["metric"], "encoder": r["encoder"], "seed": r["seed"],
            "raw_p": r["clustered_wilcoxon_p"],
        })
    clustered_corr = apply_corrections(clustered_inv, p_key="raw_p")
    def worst(rows, key):
        return max(x[key] for x in rows)
    t2["f4_carrying_leg"] = {
        "definition": "expression-restricted |log2 r_var|, gate encoders, dataset-clustered Wilcoxon",
        "worst_clustered_wilcoxon_p_across_seeds_and_encoders": worst(rvar_gate, "clustered_wilcoxon_p"),
        "worst_clustered_sign_p": worst(rvar_gate, "clustered_sign_p"),
        "worst_perm_dataset_signflip_p": worst(rvar_gate, "perm_dataset_signflip_p"),
        "paper_level_bh_q_worst": max(c["bh_q_paper"] for c in clustered_corr
                                      if c["metric"] == "r_var_abs_log2_expr" and c["encoder"] in GATE_ENCODERS),
        "paper_level_holm_p_worst": max(c["holm_p_paper"] for c in clustered_corr
                                        if c["metric"] == "r_var_abs_log2_expr" and c["encoder"] in GATE_ENCODERS),
        "survives_paper_level": True,  # filled below
    }
    fc = t2["f4_carrying_leg"]
    fc["survives_paper_level"] = bool(fc["paper_level_bh_q_worst"] <= 0.05 and fc["paper_level_holm_p_worst"] <= 0.05)
    t2["demoted_energy_leg"] = {
        "definition": "energy_distance, pca_512, dataset-clustered Wilcoxon (the marginal leg)",
        "worst_clustered_wilcoxon_p_across_seeds": worst(energy_pca, "clustered_wilcoxon_p"),
        "worst_naive_p_across_seeds": worst(energy_pca, "naive_p"),
        "paper_level_bh_q_worst_clustered": max(c["bh_q_paper"] for c in clustered_corr
                                                if c["metric"] == "energy_distance" and c["encoder"] == "pca_512"),
        "paper_level_holm_p_worst_clustered": max(c["holm_p_paper"] for c in clustered_corr
                                                  if c["metric"] == "energy_distance" and c["encoder"] == "pca_512"),
    }
    de = t2["demoted_energy_leg"]
    de["survives_paper_level_bh"] = bool(de["paper_level_bh_q_worst_clustered"] <= 0.05)
    de["survives_paper_level_holm"] = bool(de["paper_level_holm_p_worst_clustered"] <= 0.05)
    t2["clustered_primary_inventory"] = clustered_corr

    with open(RESULTS / "t2_multiplicity.json", "w") as fh:
        json.dump(t2, fh, indent=1)

    # console summary
    print("=== T1 sanity assert ===", sanity)
    print("\n=== T1 leave-two-out per-leg (full50 -> leave2out48) ===")
    for name, leg in legs.items():
        f, l = leg["full50"], leg["leave2out48"]
        print(f"{name}: nDS {f['n_datasets']}->{l['n_datasets']} | "
              f"maxHolm {f['max_naive_holm_p']:.2e}->{l['max_naive_holm_p']:.2e} | "
              f"maxClustW {f['max_clustered_wilcoxon_p']:.2e}->{l['max_clustered_wilcoxon_p']:.2e} | "
              f"clustSig/3 {f['seeds_clustered_wilcoxon_sig_0.05']}->{l['seeds_clustered_wilcoxon_sig_0.05']}")
    print("\n=== T2 inventory ===", n, "tests")
    print("survive paper BH@0.05:", len(survive_bh), "| survive paper Holm@0.05:", len(survive_holm))
    print("fail BH:", [(c['test_id'], c['metric'], c['encoder'], c['seed'], round(c['bh_q_paper'],4)) for c in fail_bh])
    print("fail Holm:", [(c['test_id'], c['metric'], c['encoder'], c['seed'], round(c['holm_p_paper'],4)) for c in fail_holm])
    print("F4-carrying leg:", t2["f4_carrying_leg"])
    print("Demoted energy leg:", t2["demoted_energy_leg"])
    return t1, t2


if __name__ == "__main__":
    main()
