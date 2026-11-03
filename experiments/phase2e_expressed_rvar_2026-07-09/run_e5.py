r"""Phase 2e E5 -- re-run F4's (and the Gaussian arm's) r_var paired tests on the
EXPRESSION-RESTRICTED |log2 r_var| statistic that F3/E4 established as the valid
one, and check whether the direction of each cited claim survives.

Motivation (the self-consistency defect this fixes).
--------------------------------------------------
Finding F3/E4 (spec S16) established that the UNRESTRICTED per-gene variance ratio
is *directionally* misleading: since var_NB = mu + mu^2/theta, the ratio to Poisson
is 1 + mu/theta, and the overwhelming majority of within-type genes have mu << 1,
so dispersion barely moves them; discrete low counts park a large mass of genes at
a ratio of exactly 1.0. E4 therefore mandated "use expression-restricted statistics
henceforth." Yet F4's paired-Wilcoxon win-fractions (\FfourPoisWinRvar,
\FfourRpWinRvar and the Gaussian-arm calibration claim) were all computed on the
UNRESTRICTED |log2 r_var| (r_var_abs_log2). This script recomputes them on the
expression-restricted statistic (median over expressed genes of |log2 r_var|),
using the existing r_var_per_gene_expressed support (mean-count >= 0.5 AND
top-200-by-real-variance; both reported), and asks: does each cited direction hold?

What is recomputed (only the cited comparisons):
  NB arm (F4), encoders pca_512 + randproj_512 (the two that pass the NB gate),
  3 seeds each (42/43/44), nb_stochastic vs
    * poisson_on_nb_mean       (dispersion-free, the LOAD-BEARING F4 comparison)
    * matched_noise            (label-free)
    * matched_noise_per_type   (label-using)
  Gaussian arm (spec S14/S15 calibration claim), encoders pca_512 + scgpt_512,
  seed 42 / noise_seed 123, stochastic_betanll vs matched_noise.

Reproducibility assertion (rule: "controls must live in the metric's space and be
sanity-asserted against known values"). Recomputing the UNRESTRICTED r_var_abs_log2
paired win-fractions from freshly regenerated decoded arrays must reproduce the
PUBLISHED win-fractions (nb_arm.json for pca_512, e2_nb_generalization.json for
randproj_512, phase2b_v3_paired.json for the Gaussian arm) to REPRO_TOL. This proves
the ONLY thing that changed between the published F4 numbers and these is the gene
restriction -- the decoder training, sampling, and grouping are byte-identical.
Fails loudly otherwise.

Win-fraction and median-paired-diff are Holm-independent; Holm is applied across the
gate-passing encoders within each seed for reporting the corrected p-values.

Only
experiments already produced by run_nb_arm.py / e2_nb_generalization.py /
run_phase2b_v3_paired.py are re-executed, with the identical seeds/config.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
P2C = ROOT / "experiments" / "phase2c_robustness_2026-07-09"
P2D = ROOT / "experiments" / "phase2d_completion_2026-07-09"
P2B = ROOT / "experiments" / "phase2b_stochastic_2026-07-09"
P2A = ROOT / "experiments" / "phase2a_decoder_2026-07-08"
for p in (ROOT, P2C, P2D, P2B, P2A):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from run_nb_arm import build_nb_variants, holm, paired, per_group_map  # noqa: E402
from run_phase2b_v3_paired import build_decoded_variants  # noqa: E402
from run_phase2b_v2 import CACHE, NOISE_SEED, SEED  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# statistics, in one family: the published unrestricted one plus the two restricted
UNRESTR = "r_var_abs_log2"
STAT_EXPR = "r_var_abs_log2_expr"      # median over expressed (mean count>=0.5) genes
STAT_TOPK = "r_var_abs_log2_topk"      # median over top-200-by-real-variance genes
RESTR_STATS = [STAT_EXPR, STAT_TOPK]
ALL_STATS = [UNRESTR] + RESTR_STATS

NB_ENCODERS = ["pca_512", "randproj_512"]
NB_SEEDS = [42, 43, 44]
NB_CONTROLS = ["poisson_on_nb_mean", "matched_noise", "matched_noise_per_type"]

GAUSS_ENCODERS = ["pca_512", "scgpt_512"]
GAUSS_CONTROL = "matched_noise"        # stochastic_betanll vs matched_noise

REPRO_TOL = 1e-6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# published win-fractions to reproduce (unrestricted r_var_abs_log2)
# ---------------------------------------------------------------------------
def load_published_nb():
    """pub[(encoder, seed, control)] = published unrestricted r_var_abs_log2 nb-win."""
    pub = {}
    nb = json.loads((P2C / "results" / "nb_arm.json").read_text())
    for s, fams in nb["paired_tests"].items():
        for control in NB_CONTROLS:
            fam = f"{UNRESTR}__nb_stochastic_vs_{control}"
            for r in fams[fam]["per_encoder"]:
                if r["encoder"] == "pca_512":
                    pub[("pca_512", int(s), control)] = float(r["learned_win_fraction"])
    e2 = json.loads((P2D / "results" / "e2_nb_generalization.json").read_text())
    for s, bym in e2["results"]["randproj_512"]["paired"].items():
        for r in bym[UNRESTR]:
            if r["control"] in NB_CONTROLS:
                pub[("randproj_512", int(s), r["control"])] = float(r["learned_win_fraction"])
    return pub


def load_published_gauss():
    """pub[encoder] = published unrestricted r_var_abs_log2 win for stochastic vs matched_noise."""
    d = json.loads((P2B / "results" / "phase2b_v3_paired.json").read_text())
    fam = d["paired_tests"][f"{UNRESTR}__stochastic_vs_{GAUSS_CONTROL}"]["per_encoder"]
    return {r["encoder"]: float(r["learned_win_fraction"]) for r in fam}


# ---------------------------------------------------------------------------
# encoder construction (identical to the published runs)
# ---------------------------------------------------------------------------
def build_nb_encoders(logX, tr):
    """pca_512 (nb_arm recipe) + randproj_512 (e2 build_encoders recipe, verbatim)."""
    rng = np.random.default_rng(0)
    enc = {}
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)
    W = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    enc["randproj_512"] = (logX @ W).astype(np.float32)
    return enc


def paired_family(a_map, b_map, stat):
    """paired() on one statistic; returns the record without Holm (added later)."""
    r = paired(a_map, b_map, stat)
    return r


def median_across_seeds(vals):
    return float(np.median(np.asarray(vals, dtype=np.float64)))


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
    log(f"raw={raw.shape} train={int(tr.sum())} val={nva} test={int(te.sum())}")

    pub_nb = load_published_nb()
    pub_gauss = load_published_gauss()

    # =====================================================================
    # NB ARM (F4)
    # =====================================================================
    nb_enc = build_nb_encoders(logX, tr)
    # per_seed[enc][seed] = {"gate":..., "aggs": {variant: per_group_map}}
    nb_store = {name: {} for name in NB_ENCODERS}
    nb_pergroup = {name: {} for name in NB_ENCODERS}
    n_groups = None
    for name in NB_ENCODERS:
        Z = nb_enc[name]
        for seed in NB_SEEDS:
            log(f"=== NB arm {name} seed {seed} ===")
            decoded_log, gate = build_nb_variants(
                Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
            aggs, pergroup = {}, {}
            for vname in ["nb_stochastic"] + NB_CONTROLS:
                agg = aggregate_over_groups(
                    logX[te], decoded_log[vname], lab_te, ds_te,
                    min_cells=20, energy_subsample=300, top_k=200, seed=0)
                aggs[vname] = per_group_map(agg)
                pergroup[vname] = agg["per_group"]
                if n_groups is None:
                    n_groups = agg["n_groups"]
            nb_store[name][seed] = {"gate": gate, "aggs": aggs}
            nb_pergroup[name][seed] = pergroup
            log(f"    gate={'PASS' if gate['gate_passed'] else 'FAIL'} "
                f"gap={gate['gap_vs_det']:+.4f}")

    # --- reproducibility gate (unrestricted win-fractions vs published) ---
    log("=== NB arm reproducibility gate (unrestricted r_var_abs_log2) ===")
    nb_repro = []
    nb_fail = []
    for name in NB_ENCODERS:
        for seed in NB_SEEDS:
            for control in NB_CONTROLS:
                got = paired(nb_store[name][seed]["aggs"]["nb_stochastic"],
                             nb_store[name][seed]["aggs"][control], UNRESTR)["learned_win_fraction"]
                exp = pub_nb[(name, seed, control)]
                d = abs(got - exp)
                rec = {"encoder": name, "seed": seed, "control": control,
                       "recomputed_win": float(got), "published_win": exp, "abs_diff": float(d)}
                nb_repro.append(rec)
                if d > REPRO_TOL:
                    nb_fail.append(rec)
    max_nb = max(r["abs_diff"] for r in nb_repro)
    log(f"  NB max abs win-fraction diff over {len(nb_repro)} checks = {max_nb:.3e}")
    for r in nb_fail:
        log(f"  MISMATCH {r['encoder']} s{r['seed']} {r['control']}: "
            f"got={r['recomputed_win']:.8f} pub={r['published_win']:.8f} d={r['abs_diff']:.2e}")
    assert not nb_fail, (
        f"NB reproducibility gate FAILED ({len(nb_fail)} mismatch, max {max_nb:.2e} > "
        f"{REPRO_TOL:g}): regenerated decoded arrays do not reproduce the published F4 "
        f"unrestricted win-fractions, so a restricted result would not be an apples-to-"
        f"apples restatement. Refusing to report restricted numbers.")
    log("  NB reproducibility gate PASSED -- only the gene restriction changes.")

    # --- restricted paired tests (Holm across the 2 gate-passing encoders / seed) ---
    nb_paired = {}  # nb_paired[stat][control] = {"per_seed":[...], "median_win":..., ...}
    for stat in ALL_STATS:
        nb_paired[stat] = {}
        for control in NB_CONTROLS:
            per_seed = []
            for seed in NB_SEEDS:
                recs, raw_p = [], []
                for name in NB_ENCODERS:
                    r = paired(nb_store[name][seed]["aggs"]["nb_stochastic"],
                               nb_store[name][seed]["aggs"][control], stat)
                    r["encoder"] = name
                    r["gate_passed"] = nb_store[name][seed]["gate"]["gate_passed"]
                    recs.append(r)
                    raw_p.append(r["p_value"])
                for r, ap in zip(recs, holm(raw_p)):
                    r["p_holm"] = float(ap)
                    r["significant_holm_0.05"] = bool(ap < 0.05)
                per_seed.append({"seed": seed, "per_encoder": recs})
            # median-across-seeds reduction (matches the paper's macro reduction)
            by_enc = {}
            for name in NB_ENCODERS:
                wins = [next(r for r in ps["per_encoder"] if r["encoder"] == name)["learned_win_fraction"]
                        for ps in per_seed]
                diffs = [next(r for r in ps["per_encoder"] if r["encoder"] == name)["median_paired_diff"]
                         for ps in per_seed]
                holms = [next(r for r in ps["per_encoder"] if r["encoder"] == name)["p_holm"]
                         for ps in per_seed]
                sigs = [next(r for r in ps["per_encoder"] if r["encoder"] == name)["significant_holm_0.05"]
                        for ps in per_seed]
                by_enc[name] = {
                    "nb_win_median": median_across_seeds(wins),
                    "control_win_median": float(1.0 - median_across_seeds(wins)),
                    "median_paired_diff_median": median_across_seeds(diffs),
                    "p_holm_max": float(max(holms)),
                    "all_seeds_significant": bool(all(sigs)),
                    "nb_win_per_seed": [float(w) for w in wins],
                }
            nb_paired[stat][control] = {"per_seed": per_seed, "by_encoder": by_enc}
            for name in NB_ENCODERS:
                b = by_enc[name]
                log(f"  [{stat:20s}] nb vs {control:24s} {name:12s} "
                    f"nb_win_med={b['nb_win_median']:.3f} "
                    f"ctrl_win_med={b['control_win_median']:.3f} "
                    f"medDiff={b['median_paired_diff_median']:+.4f} "
                    f"p_holm<={b['p_holm_max']:.1e} sig_all={b['all_seeds_significant']}")

    # =====================================================================
    # GAUSSIAN ARM (spec S14/S15 calibration claim)
    # =====================================================================
    log("=== Gaussian arm (stochastic_betanll vs matched_noise) ===")
    g_enc = {}
    rng = np.random.default_rng(0)
    g_enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)
    p = CACHE / "enc_A_scgpt.npy"
    z = np.load(p)
    assert z.shape[0] == logX.shape[0], "scgpt cache row mismatch"
    g_enc["scgpt_512"] = z.astype(np.float32)

    g_store = {}
    g_pergroup = {}
    for name in GAUSS_ENCODERS:
        log(f"--- Gaussian encoder {name} ---")
        decoded, gate = build_decoded_variants(
            g_enc[name], tr, va, te, nva, logX, logX[te], lab_va, lab_te, name)
        aggs, pergroup = {}, {}
        for vname in ["stochastic_betanll", GAUSS_CONTROL]:
            agg = aggregate_over_groups(
                logX[te], decoded[vname], lab_te, ds_te,
                min_cells=20, energy_subsample=300, top_k=200, seed=0)
            aggs[vname] = per_group_map(agg)
            pergroup[vname] = agg["per_group"]
        g_store[name] = {"gate": gate, "aggs": aggs}
        g_pergroup[name] = pergroup
        log(f"    gate_passed={gate['gate_passed']} beta={gate['beta_used']}")

    # reproducibility gate (unrestricted)
    log("=== Gaussian reproducibility gate (unrestricted r_var_abs_log2) ===")
    g_repro, g_fail = [], []
    for name in GAUSS_ENCODERS:
        got = paired(g_store[name]["aggs"]["stochastic_betanll"],
                     g_store[name]["aggs"][GAUSS_CONTROL], UNRESTR)["learned_win_fraction"]
        exp = pub_gauss[name]
        d = abs(got - exp)
        rec = {"encoder": name, "recomputed_win": float(got), "published_win": exp, "abs_diff": float(d)}
        g_repro.append(rec)
        if d > REPRO_TOL:
            g_fail.append(rec)
        log(f"  {name}: got={got:.8f} pub={exp:.8f} diff={d:.2e}")
    max_g = max(r["abs_diff"] for r in g_repro)
    assert not g_fail, (
        f"Gaussian reproducibility gate FAILED (max {max_g:.2e} > {REPRO_TOL:g}); "
        f"regenerated arrays do not reproduce the published Gaussian-arm win-fractions.")
    log("  Gaussian reproducibility gate PASSED.")

    # restricted paired tests (Holm across the 2 encoders)
    g_paired = {}
    for stat in ALL_STATS:
        recs, raw_p = [], []
        for name in GAUSS_ENCODERS:
            r = paired(g_store[name]["aggs"]["stochastic_betanll"],
                       g_store[name]["aggs"][GAUSS_CONTROL], stat)
            r["encoder"] = name
            r["gate_passed"] = g_store[name]["gate"]["gate_passed"]
            recs.append(r)
            raw_p.append(r["p_value"])
        for r, ap in zip(recs, holm(raw_p)):
            r["p_holm"] = float(ap)
            r["significant_holm_0.05"] = bool(ap < 0.05)
        g_paired[stat] = {"per_encoder": recs}
        for r in recs:
            log(f"  [{stat:20s}] stoch vs matched_noise {r['encoder']:12s} "
                f"win={r['learned_win_fraction']:.3f} medDiff={r['median_paired_diff']:+.4f} "
                f"p_holm={r['p_holm']:.1e} {'SIG' if r['significant_holm_0.05'] else 'ns'}")

    # =====================================================================
    # persist per-group arrays + paired tables
    # =====================================================================
    def pg_arrays(pergroup, variants):
        """Aligned per-group arrays for the three r_var statistics + keys."""
        # use nb_stochastic / stochastic_betanll ordering as the key reference
        ref = pergroup[variants[0]]
        keys = [(g["dataset"], g["cell_type"]) for g in ref]
        out = {"group_keys": [list(k) for k in keys],
               "n_cells": [g["n_cells"] for g in ref], "variants": {}}
        for v in variants:
            gm = {(g["dataset"], g["cell_type"]): g for g in pergroup[v]}
            out["variants"][v] = {
                s: [gm[k][s] for k in keys] for s in ALL_STATS
            }
        return out

    nb_pergroup_out = {}
    for name in NB_ENCODERS:
        nb_pergroup_out[name] = {
            str(seed): pg_arrays(nb_pergroup[name][seed], ["nb_stochastic"] + NB_CONTROLS)
            for seed in NB_SEEDS
        }
    g_pergroup_out = {
        name: pg_arrays(g_pergroup[name], ["stochastic_betanll", GAUSS_CONTROL])
        for name in GAUSS_ENCODERS
    }

    nb_gates = {name: {str(s): nb_store[name][s]["gate"]["gate_passed"] for s in NB_SEEDS}
                for name in NB_ENCODERS}

    out = {
        "purpose": "Phase 2e E5: re-run F4's r_var paired tests (and the Gaussian-arm "
                   "calibration claim) on the EXPRESSION-RESTRICTED |log2 r_var| that "
                   "F3/E4 established as the valid statistic; determine whether each "
                   "cited direction survives. Reproducibility-gated against the "
                   "published unrestricted win-fractions to prove only the gene "
                   "restriction changed.",
        "statistics": {
            "unrestricted": UNRESTR,
            "expressed": "median over genes with real within-type mean count>=0.5 of |log2(var_dec/var_real)|",
            "topk": "median over top-200-by-real-within-type-variance genes of |log2(var_dec/var_real)|",
        },
        "n_groups": n_groups,
        "reproducibility_gate": {
            "tol": REPRO_TOL,
            "nb_max_abs_win_diff": max_nb, "nb_checks": len(nb_repro), "nb_passed": True,
            "gauss_max_abs_win_diff": max_g, "gauss_checks": len(g_repro), "gauss_passed": True,
            "nb_detail": nb_repro, "gauss_detail": g_repro,
        },
        "nb_arm": {
            "encoders": NB_ENCODERS, "seeds": NB_SEEDS, "controls": NB_CONTROLS,
            "gates": nb_gates,
            "paired": nb_paired,
        },
        "gaussian_arm": {
            "encoders": GAUSS_ENCODERS, "seed": SEED, "noise_seed": NOISE_SEED,
            "control": GAUSS_CONTROL,
            "gates": {name: g_store[name]["gate"]["gate_passed"] for name in GAUSS_ENCODERS},
            "paired": g_paired,
        },
        "per_group": {"nb_arm": nb_pergroup_out, "gaussian_arm": g_pergroup_out},
    }
    (RESULTS / "e5_expressed_rvar.json").write_text(json.dumps(out, indent=2))
    log(f"=== E5 done in {time.time()-t0:.0f}s -> {RESULTS / 'e5_expressed_rvar.json'} ===")


if __name__ == "__main__":
    main()
