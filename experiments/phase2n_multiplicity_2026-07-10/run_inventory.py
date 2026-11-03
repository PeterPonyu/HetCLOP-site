"""Paper-level multiplicity inventory over the manuscript's asserted significance claims.

The manuscript asserts many dataset-clustered tests. A referee will ask whether they survive
correction for multiplicity. This script enumerates EVERY dataset-clustered p-value the paper
asserts as significant, sources each from its experiment JSON (never hand-typed), and applies
BOTH corrections over the union as a SINGLE family -- the most conservative choice, since it
does not rely on our own partition into confirmatory/exploratory families.

Reusing the tested machinery in phase2h/multiplicity.py (Holm step-down, BH step-up).

Convention: each entry is one dataset-clustered Wilcoxon p on the 50 dataset medians. Where a
claim spans seeds we take the WORST (largest) p across seeds, i.e. the claim's weakest form.

No GPU, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2h_reanalysis_2026-07-09"))

from multiplicity import apply_corrections  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "multiplicity_inventory.json"


def J(rel):
    return json.loads((ROOT / "experiments" / rel).read_text())


def build_inventory():
    tests = []

    def add(test_id, family, raw_p, source):
        tests.append({"test_id": test_id, "family": family, "raw_p": float(raw_p), "source": source})

    # ---- Arm B: NB beats Poisson on energy, 4 encoders (worst clustered p per encoder) ----
    p2g = J("phase2g_armb_f4_2026-07-09/results/phase2g.json")
    by_enc = {}
    for c in p2g["E7_clustered_inference"]["per_comparison"]:
        if c["metric"] == "energy_distance":
            by_enc.setdefault(c["encoder"], []).append(c["clustered_50dataset"]["wilcoxon_p"])
    for enc, ps in by_enc.items():
        add(f"armB_energy_{enc}", "F4_armB_deficit", max(ps), "phase2g.json E7")

    # ---- Deployed generators: Poisson (or NB) wins the rvar contest ----
    cfg = J("phase2j_external_2026-07-09/results/cfgen.json")
    add("cfgen_rvar_poisson_wins", "F4_deployed",
        max(r["contrasts"]["rvar_expressed"]["clustered_wilcoxon_p"] for r in cfg["results"].values()),
        "cfgen.json")
    scvi = J("phase2j_external_2026-07-09/results/scvi.json")
    for mode in ("gene", "gene-cell"):
        add(f"scvi_{mode}_rvar", "F4_deployed",
            max(r["contrasts"]["rvar_expressed"]["clustered_wilcoxon_p"] for r in scvi["results"][mode].values()),
            "scvi.json")

    # ---- The mean-fixed theta probe: three contrasts x three metrics ----
    pr = list(J("phase2j_external_2026-07-09/results/scvi_theta_probe.json")["results"].values())[0]
    for contrast in ("nb_amortized_vs_poisson", "nb_global_vs_poisson", "nb_amortized_vs_nb_global"):
        for metric in ("rvar_expressed", "rvar_topk", "energy"):
            k = f"{contrast}__{metric}"
            add(f"probe_{k}", "F4_probe_factor_b", pr["contrasts"][k]["clustered_wilcoxon_p"],
                "scvi_theta_probe.json")

    # ---- G1 positive control: headline separation (worst across redraws) ----
    g1 = J("phase2i_extensions_2026-07-09/results/g1_positive_control.json")
    add("g1_pca2_vs_pca512_energy", "F2_positive_control",
        max(g1["headline_pca2_vs_pca512_multidraw"]["clustered_wilcoxon_p_range"]),
        "g1_positive_control.json")
    add("g1_ordering_energy_permutation", "F2_positive_control",
        g1["ordering_recovered"]["spearman_logk_vs_energy"]["p_two_sided"],
        "g1_positive_control.json (exact permutation)")

    # ---- G4 monotonicity (exact permutation), G5 ZINB (worst clustered) ----
    g4 = J("phase2i_extensions_2026-07-09/results/g4_monotonicity.json")
    add("g4_gap_vs_energy_monotonicity", "F4_mechanism",
        g4["statistics"]["gap_vs_energy_effect"]["exact_permutation_p"]["p_two_sided"],
        "g4_monotonicity.json (exact permutation)")
    g5 = J("phase2i_extensions_2026-07-09/results/g5_zinb.json")
    for k in ("zinb_stochastic_vs_nb_stochastic__energy_distance",
              "zinb_stochastic_vs_nb_stochastic__r_var_abs_log2_expr"):
        add(f"g5_{k}", "F4_armA_zinb", g5["verdict"][k]["worst_clustered_wilcoxon_p"], "g5_zinb.json")

    # ---- F5 trap: deterministic decoder worse than every stochastic variant ----
    detp = J("phase2k_det_paired_2026-07-09/results/det_paired.json")
    add("det_vs_stochastic_worst", "F5_trap", detp["max_clustered_wilcoxon_p"], "det_paired.json")

    return tests


def main() -> int:
    tests = build_inventory()
    corrected = apply_corrections(tests, p_key="raw_p")
    n = len(corrected)
    surv_bh = [c for c in corrected if c["survives_bh_0.05"]]
    surv_holm = [c for c in corrected if c["survives_holm_0.05"]]
    fail_bh = [c for c in corrected if not c["survives_bh_0.05"]]
    fail_holm = [c for c in corrected if not c["survives_holm_0.05"]]

    out = {
        "purpose": "Paper-level multiplicity over EVERY dataset-clustered significance claim the "
                   "manuscript asserts, corrected as a SINGLE family (most conservative).",
        "n_tests": n,
        "n_survive_bh_0.05": len(surv_bh),
        "n_survive_holm_0.05": len(surv_holm),
        "all_survive_bh": len(fail_bh) == 0,
        "all_survive_holm": len(fail_holm) == 0,
        "worst_raw_p": max(c["raw_p"] for c in corrected),
        "worst_bh_q": max(c["bh_q_paper"] for c in corrected),
        "worst_holm_p": max(c["holm_p_paper"] for c in corrected),
        "families": sorted({c["family"] for c in corrected}),
        "failing_bh": [(c["test_id"], c["raw_p"], c["bh_q_paper"]) for c in fail_bh],
        "failing_holm": [(c["test_id"], c["raw_p"], c["holm_p_paper"]) for c in fail_holm],
        "tests": corrected,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"{n} dataset-clustered claims, corrected as ONE family")
    print(f"  survive BH  (q<=0.05): {len(surv_bh)}/{n}   worst q = {out['worst_bh_q']:.3g}")
    print(f"  survive Holm(p<=0.05): {len(surv_holm)}/{n}   worst p = {out['worst_holm_p']:.3g}")
    print(f"  worst raw p = {out['worst_raw_p']:.3g}")
    for c in sorted(corrected, key=lambda x: -x["raw_p"])[:5]:
        print(f"    {c['test_id']:44s} raw={c['raw_p']:.3g} bh_q={c['bh_q_paper']:.3g} holm={c['holm_p_paper']:.3g}")
    if fail_holm:
        print("  FAILING Holm:", [c[0] for c in out["failing_holm"]])
    if fail_bh:
        print("  FAILING BH:", [c[0] for c in out["failing_bh"]])
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
