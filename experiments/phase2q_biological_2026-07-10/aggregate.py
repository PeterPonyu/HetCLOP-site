"""Aggregate the Tier-1 biological readout across seeds and decide what may be claimed.

Reads results/biological_seed{0,1,2}.json (each a full mean-fixed run) and emits
results/biological_agg.json with, for every draw:

  * marker recall@50: mean +- sd across seeds
  * TSTR macro-F1 and rare-type F1: mean +- sd across seeds
  * the paired per-cell-type marker comparison (worst-across-seed Wilcoxon p, min win-fraction)

and then applies three explicit claim gates, so the manuscript wording is decided by the data:

  claim_global_is_worst      : nb_global is worst on BOTH readouts in EVERY seed.
  claim_cond_beats_pois_marker : conditional beats Poisson on marker recall in every seed AND
                                 the paired Wilcoxon is significant in every seed.
  claim_cond_beats_pois_tstr : conditional beats Poisson on TSTR macro-F1 by MORE than the
                               across-seed sd (i.e. the "tie" is genuinely broken). Expected FALSE.

A gate that comes back False is reported as False; the paper then states the tie/no-difference
rather than a win. This is the whole point of running three seeds before writing.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DRAWS = ("poisson", "nb_global", "nb_conditional")


def load_seeds():
    out = {}
    for s in (0, 1, 2):
        p = RES / f"biological_seed{s}.json"
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def ms(vals):
    return (st.fmean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0))


def main() -> int:
    seeds = load_seeds()
    if len(seeds) < 2:
        print(f"only {len(seeds)} seed(s) present; need >= 2")
        return 1
    ks = sorted(seeds)

    marker = {d: ms([seeds[s]["marker_recovery"][d]["mean_recall"] for s in ks]) for d in DRAWS}
    tstr_macro = {d: ms([seeds[s]["tstr"][d]["macro_f1"] for s in ks]) for d in DRAWS}
    tstr_rare = {d: ms([seeds[s]["tstr"][d]["rare_macro_f1"] for s in ks]) for d in DRAWS}
    # H1 robustness: rank-based (AUROC) marker recall and unmasked rare-type F1
    marker_auroc = ({d: ms([seeds[s]["marker_recovery_auroc"][d]["mean_recall"] for s in ks]) for d in DRAWS}
                    if all("marker_recovery_auroc" in seeds[s] for s in ks) else {})
    tstr_rare_unmasked = ({d: ms([seeds[s]["tstr"][d]["rare_macro_f1_unmasked"] for s in ks]) for d in DRAWS}
                          if all("rare_macro_f1_unmasked" in seeds[ks[0]]["tstr"][DRAWS[0]] for _ in [0]) else {})
    ceiling = {
        "macro": ms([seeds[s]["tstr"]["real_ceiling"]["macro_f1"] for s in ks]),
        "rare": ms([seeds[s]["tstr"]["real_ceiling"]["rare_macro_f1"] for s in ks]),
    }
    # marker reliability ceiling (real-vs-real split), mean +- sd across seeds
    marker_ceiling = ms([seeds[s]["marker_ceiling"] for s in ks
                         if seeds[s].get("marker_ceiling") is not None])
    # dataset-clustered marker inference: worst (largest) clustered sign p across seeds,
    # min win fraction, and the tightest (widest-lower) bootstrap CI across seeds.
    marker_clustered = {}
    if all("marker_clustered" in seeds[s] for s in ks):
        for other in ("poisson", "nb_global"):
            key = f"nb_conditional_vs_{other}"
            vs = [seeds[s]["marker_clustered"][key] for s in ks]
            marker_clustered[key] = {
                "worst_clustered_sign_p": max(v["clustered_sign_p"] for v in vs),
                "min_win_fraction_groups": min(v["win_fraction_groups"] for v in vs),
                "min_ci_lo": min(v["win_fraction_ci_lo"] for v in vs),
                "min_ci_hi": min(v["win_fraction_ci_hi"] for v in vs),
                "n_datasets": vs[0]["n_datasets"], "n_groups": vs[0]["n_groups"],
            }

    paired = {}
    for other in ("poisson", "nb_global"):
        key = f"nb_conditional_vs_{other}"
        paired[key] = {
            "worst_wilcoxon_p": max(seeds[s]["marker_paired"][key]["wilcoxon_p"] for s in ks),
            "min_win_fraction_types": min(seeds[s]["marker_paired"][key]["win_fraction_types"] for s in ks),
            "mean_diff": ms([seeds[s]["marker_paired"][key]["mean_diff"] for s in ks])[0],
        }

    # ---- claim gates ------------------------------------------------------------------------
    global_worst_marker = all(
        seeds[s]["marker_recovery"]["nb_global"]["mean_recall"] <
        min(seeds[s]["marker_recovery"][d]["mean_recall"] for d in ("poisson", "nb_conditional"))
        for s in ks)
    global_worst_tstr = all(
        seeds[s]["tstr"]["nb_global"]["macro_f1"] <
        min(seeds[s]["tstr"][d]["macro_f1"] for d in ("poisson", "nb_conditional"))
        for s in ks)
    claim_global_is_worst = bool(global_worst_marker and global_worst_tstr)

    claim_cond_beats_pois_marker = bool(all(
        seeds[s]["marker_recovery"]["nb_conditional"]["mean_recall"] >
        seeds[s]["marker_recovery"]["poisson"]["mean_recall"] and
        seeds[s]["marker_paired"]["nb_conditional_vs_poisson"]["wilcoxon_p"] < 0.05
        for s in ks))

    # is the TSTR conditional-vs-poisson gap bigger than across-seed noise?
    gap = tstr_macro["nb_conditional"][0] - tstr_macro["poisson"][0]
    noise = max(tstr_macro["nb_conditional"][1], tstr_macro["poisson"][1], 1e-9)
    claim_cond_beats_pois_tstr = bool(gap > noise)

    out = {
        "purpose": "3-seed aggregation of the mean-fixed biological readout; claim gates decide wording.",
        "seeds": ks,
        "n_types_scored": seeds[ks[0]]["n_types_scored"],
        "n_rare_types": seeds[ks[0]]["n_rare_types"],
        "topk": seeds[ks[0]]["topk"],
        "marker_recall_mean_sd": {d: marker[d] for d in DRAWS},
        "tstr_macro_f1_mean_sd": {d: tstr_macro[d] for d in DRAWS},
        "tstr_rare_f1_mean_sd": {d: tstr_rare[d] for d in DRAWS},
        "real_ceiling_mean_sd": ceiling,
        "marker_auroc_mean_sd": marker_auroc,
        "tstr_rare_unmasked_mean_sd": tstr_rare_unmasked,
        "marker_ceiling_mean_sd": marker_ceiling,
        "marker_clustered_worst_across_seeds": marker_clustered,
        "marker_paired_worst_across_seeds": paired,
        "claims": {
            "global_is_worst_on_both_readouts_every_seed": claim_global_is_worst,
            "conditional_beats_poisson_on_marker_every_seed": claim_cond_beats_pois_marker,
            "conditional_beats_poisson_on_tstr_beyond_noise": claim_cond_beats_pois_tstr,
        },
        "tstr_cond_minus_pois_macro": gap,
        "tstr_across_seed_sd": noise,
    }
    (RES / "biological_agg.json").write_text(json.dumps(out, indent=2))

    print(f"=== Tier-1 biological readout, {len(ks)} seeds ===")
    print(f"{'draw':16s} {'marker@50':>16s} {'TSTR macro':>16s} {'TSTR rare':>16s}")
    for d in DRAWS:
        print(f"{d:16s} {marker[d][0]:.4f}+-{marker[d][1]:.4f} "
              f"{tstr_macro[d][0]:.4f}+-{tstr_macro[d][1]:.4f} "
              f"{tstr_rare[d][0]:.4f}+-{tstr_rare[d][1]:.4f}")
    print(f"{'real (ceiling)':16s} {'1.0000':>16s} {ceiling['macro'][0]:.4f}+-{ceiling['macro'][1]:.4f} "
          f"{ceiling['rare'][0]:.4f}+-{ceiling['rare'][1]:.4f}")
    print("\npaired marker (conditional - other), worst across seeds:")
    for k, v in paired.items():
        print(f"  {k:30s} mean_diff={v['mean_diff']:+.4f} min_win={v['min_win_fraction_types']:.3f} "
              f"worst_p={v['worst_wilcoxon_p']:.2e}")
    print("\nCLAIM GATES:")
    for k, v in out["claims"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nTSTR conditional - poisson (macro) = {gap:+.4f} vs across-seed sd {noise:.4f} "
          f"-> {'separable' if claim_cond_beats_pois_tstr else 'WITHIN NOISE: report as a tie'}")
    print(f"wrote {(RES / 'biological_agg.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
