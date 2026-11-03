"""Phase 2d E2 -- does the corrected NB-arm result GENERALISE beyond pca_512?

PHASE2C §Task-2 established, on pca_512 only (scgpt_512 excluded on gate failure,
3/3 seeds): the full learned-dispersion NB (nb_stochastic) LOSES to a Poisson
resampling of NB's own fitted mean (poisson_on_nb_mean, theta->inf) on both
energy_distance and |log2 r_var|, all 3 seeds -- the learned per-gene
overdispersion is counterproductive. Scope was explicitly limited to pca_512.

E2 re-runs the EXACT corrected NB arm (all six controls incl. poisson_on_nb_mean,
nbmean_matched_noise[_per_type]) on TWO more encoders -- randproj_512 (linear,
random basis) and geneformer (FM) -- 3 seeds each, gating each. scgpt_512 is
already known to fail the NB gate reproducibly (3/3 seeds, gaps +0.0216..+0.0232)
and is NOT rerun; that exclusion is cited, not re-derived.

Key question: does poisson_on_nb_mean beat nb_stochastic on these encoders too? If
yes on both, the counterproductive-dispersion finding generalises. If it holds
only on pca_512, we say so plainly.

Reuses ``build_nb_variants`` / ``paired`` / ``holm`` / control lists DIRECTLY from
phase2c ``run_nb_arm`` (the already-fixed log1p-space control code path), so the
recipe is byte-for-byte identical. Two sanity guards per rule #3:
  (1) pca_512 seed-42 ANCHOR: re-run pca_512 through the imported build function
      in THIS process and assert its rebuilt matched_noise/_per_type energy
      medians reproduce PHASE2B_V3's 1.012/0.951 within SANITY_TOL -- proving the
      imported code is the log1p-space (not count-space) version here.
  (2) per-encoder O(1) guard: every rebuilt log1p noise control's energy median
      must lie in [0.3, 3.0] (the count-space bug produced 6-45); fail loudly.

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

from run_phase2b_v2 import CACHE, load_geneformer_armA  # noqa: E402
from run_nb_arm import (  # noqa: E402
    CONTROLS, CONTROL_QUESTION, PAIRED_METRICS, SANITY_TOL,
    V3_PCA512_ENERGY_MEDIAN, build_nb_variants, holm, paired,
)
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
TARGET_ENCODERS = ["randproj_512", "geneformer"]
O1_LO, O1_HI = 0.3, 3.0  # per-encoder sanity band for log1p noise-control energy


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def build_encoders(logX, tr):
    enc = {}
    # randproj_512: byte-for-byte run_phase2b_v2's Arm A construction (rng seed 0).
    rng = np.random.default_rng(0)
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        logX[tr]).transform(logX).astype(np.float32)  # anchor only
    W = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    enc["randproj_512"] = (logX @ W).astype(np.float32)
    return enc


def run_encoder(name, Z, tr, va, te, nva, logX, raw, lib, mean_lib,
                lab_va, lab_te, ds_te, seeds):
    """Build all 8 NB variants at each seed, aggregate, run paired Wilcoxon
    (nb_stochastic vs each control), return per-seed gate + paired records and
    the aggregate maps needed for sanity checks."""
    per_seed = {}
    for seed in seeds:
        log(f"=== E2 NB arm encoder {name} seed {seed} ===")
        decoded_log, gate = build_nb_variants(
            Z, tr, va, te, nva, logX, raw, lib, mean_lib, lab_va, lab_te, seed)
        aggs, aggs_full = {}, {}
        for vname, dec in decoded_log.items():
            agg = aggregate_over_groups(
                logX[te], dec, lab_te, ds_te,
                min_cells=20, energy_subsample=300, top_k=200, seed=0)
            aggs[vname] = per_group_map(agg)
            aggs_full[vname] = agg
        # per-encoder O(1) sanity guard on the log1p noise controls
        o1 = {}
        for vname in ("matched_noise", "matched_noise_per_type",
                      "nbmean_matched_noise", "nbmean_matched_noise_per_type"):
            med = aggs_full[vname]["energy_distance"]["median"]
            ok = bool(O1_LO <= med <= O1_HI)
            o1[vname] = {"energy_median": float(med), "in_band": ok}
            assert ok, (f"[{name} seed {seed}] log1p control {vname} energy median "
                        f"{med:.3f} outside sane O(1) band [{O1_LO},{O1_HI}] -- "
                        f"count-space control bug signature; refusing to trust results")
        per_seed[str(seed)] = {"gate": gate, "aggs": aggs,
                               "aggs_energy_median": {
                                   v: aggs_full[v]["energy_distance"]["median"]
                                   for v in decoded_log},
                               "o1_sanity": o1}
        log(f"    [{name} seed {seed}] gate={'PASS' if gate['gate_passed'] else 'FAIL'} "
            f"gap={gate['gap_vs_det']:+.4f} nb_mean_log_r={gate['nb_mean_log_pearson']:.4f}")
    return per_seed


def paired_block(per_seed, seeds):
    """nb_stochastic vs each control, per seed, Holm across the CONTROLS family
    within a seed (single encoder here -> Holm over the 6 controls)."""
    out = {}
    for seed in seeds:
        s = str(seed)
        out[s] = {}
        for metric in PAIRED_METRICS:
            raw_p, recs = [], []
            for control in CONTROLS:
                r = paired(per_seed[s]["aggs"]["nb_stochastic"],
                           per_seed[s]["aggs"][control], metric)
                r["control"] = control
                r["question"] = CONTROL_QUESTION[control]
                recs.append(r)
                raw_p.append(r["p_value"])
            for r, ap in zip(recs, holm(raw_p)):
                r["p_holm"] = float(ap)
                r["significant_holm_0.05"] = bool(ap < 0.05)
            out[s][metric] = recs
            for r in recs:
                log(f"  seed={seed} {metric} nb_stochastic vs {r['control']:32s} "
                    f"nb_win={r['learned_win_fraction']:.3f} "
                    f"medDiff={r['median_paired_diff']:+.5f} "
                    f"p_holm={r['p_holm']:.2e} "
                    f"{'SIG' if r['significant_holm_0.05'] else 'ns'} "
                    f"[{r['question']}]")
    return out


def main() -> None:
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())
    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    assert np.allclose(np.log1p(raw[:2000]), logX[:2000], atol=1e-5), \
        "logX must equal log1p(raw)"
    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())
    lab_va, lab_te, ds_te = labels[va], labels[te], dataset_ids[te]
    log(f"raw={raw.shape} mean_lib={mean_lib:.1f} train={int(tr.sum())} "
        f"val={nva} test={int(te.sum())}")

    enc = build_encoders(logX, tr)
    gf = load_geneformer_armA(ordered)
    if gf is not None and gf.shape[0] == logX.shape[0]:
        enc["geneformer"] = gf.astype(np.float32)
        log(f"loaded geneformer Arm A {gf.shape}")
    else:
        log("geneformer Arm A MISSING -> excluded")

    results = {}

    # (1) pca_512 seed-42 ANCHOR: re-validate the imported log1p control recipe
    # reproduces PHASE2B_V3's medians in THIS process.
    log("=== SANITY ANCHOR: pca_512 seed 42 (validate imported control recipe) ===")
    anchor = run_encoder("pca_512", enc["pca_512"], tr, va, te, nva, logX, raw,
                         lib, mean_lib, lab_va, lab_te, ds_te, [42])
    anchor_energy = anchor["42"]["aggs_energy_median"]
    sanity_anchor = {}
    for vname, expected in V3_PCA512_ENERGY_MEDIAN.items():
        got = anchor_energy[vname]
        diff = abs(got - expected)
        ok = bool(diff <= SANITY_TOL)
        sanity_anchor[vname] = {"got": float(got), "expected_v3": expected,
                                "abs_diff": float(diff), "passed": ok}
        log(f"  [ANCHOR] {vname} energy median got={got:.4f} v3~={expected:.3f} "
            f"diff={diff:.4f} {'PASS' if ok else 'FAIL'}")
    failed = [v for v, r in sanity_anchor.items() if not r["passed"]]
    assert not failed, (f"pca_512 anchor FAILED for {failed}: imported build_nb_variants "
                        f"does not reproduce v3 medians within {SANITY_TOL}; refusing to "
                        f"trust E2 results. {sanity_anchor}")
    log("  [ANCHOR] PASSED -- imported control recipe is the log1p-space version.")

    for name in TARGET_ENCODERS:
        if name not in enc:
            log(f"encoder {name} unavailable -> skipped")
            continue
        per_seed = run_encoder(name, enc[name], tr, va, te, nva, logX, raw, lib,
                               mean_lib, lab_va, lab_te, ds_te, SEEDS)
        gates = {s: per_seed[s]["gate"]["gate_passed"] for s in per_seed}
        pb = paired_block(per_seed, SEEDS)
        # Verdict per encoder: does poisson_on_nb_mean beat nb_stochastic on both
        # metrics across all gate-passing seeds?
        pass_seeds = [s for s in map(str, SEEDS) if gates[s]]
        verdict = {}
        for metric in PAIRED_METRICS:
            wins_poisson = []
            for s in pass_seeds:
                rec = next(r for r in pb[s][metric] if r["control"] == "poisson_on_nb_mean")
                # poisson beats nb_stochastic when nb win fraction < 0.5 and sig
                wins_poisson.append(rec["learned_win_fraction"] < 0.5 and
                                    rec["significant_holm_0.05"])
            verdict[metric] = {
                "n_gate_pass_seeds": len(pass_seeds),
                "poisson_beats_nb_all_pass_seeds": bool(pass_seeds and all(wins_poisson)),
                "per_seed_poisson_wins_sig": wins_poisson,
            }
        results[name] = {
            "gates": gates,
            "gate_records": {s: per_seed[s]["gate"] for s in per_seed},
            "paired": pb,
            "poisson_on_nb_mean_verdict": verdict,
        }
        log(f"  [{name} VERDICT] poisson_on_nb_mean beats nb_stochastic (all "
            f"gate-pass seeds, both metrics): "
            f"energy={verdict['energy_distance']['poisson_beats_nb_all_pass_seeds']} "
            f"|log2r_var|={verdict['r_var_abs_log2']['poisson_beats_nb_all_pass_seeds']}")

    out = {
        "purpose": "Phase 2d E2: does the pca_512 counterproductive-dispersion "
                   "result (poisson_on_nb_mean beats nb_stochastic) generalise to "
                   "randproj_512 and geneformer? Corrected NB arm, 6 controls, 3 "
                   "seeds each, gated. scgpt_512 cited as excluded (gate fail 3/3), "
                   "not rerun.",
        "seeds": SEEDS, "target_encoders": TARGET_ENCODERS,
        "scgpt_512_excluded": "gate fails reproducibly (PHASE2C §Task-2, gaps "
                              "+0.0216/+0.0222/+0.0232, 3/3 seeds); not rerun.",
        "sanity_anchor_pca512_seed42": sanity_anchor,
        "sanity_o1_band": [O1_LO, O1_HI],
        "results": results,
    }
    (RESULTS / "e2_nb_generalization.json").write_text(json.dumps(out, indent=2))
    log(f"=== E2 done in {time.time()-t0:.0f}s -> {RESULTS / 'e2_nb_generalization.json'} ===")


if __name__ == "__main__":
    main()
