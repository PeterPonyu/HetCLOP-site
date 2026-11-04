"""Phase 2x -- abundance-based rarity robustness for the rare-type TSTR headline.

The paper's rare-type
leg defines "rare" as < 50 held-out TEST cells -- an artefact of the 70/15/15
split and the 3,000-cell/dataset cap. This script re-runs the SAME TSTR readout
on the SAME mean-fixed draws with an abundance-based rarity definition:

    rare = cell types in the bottom quartile of TRUE CORPUS ABUNDANCE
           (total cells per type over all 133,845 corpus cells, all splits;
           abundance <= Q1 of the per-type totals). Corpus abundance is a fixed
           biological property of the assembled corpus, independent of the split.

Everything else is held identical to phase2q (run_biological.py): same corpus,
same scVI gene-cell mu/theta (cached), same three draws at the same sampling
seed, same A/B TSTR split, same classifier, same masked/unmasked rare-F1
computation. The original <50-test-cells definition is recomputed alongside as
an in-run anchor, and the seed-0 run is validated against
phase2q's biological_seed0.json before any new number is trusted.

mu/theta provenance: phase2t_umap_2026-07-10/cache_mu_theta.npz is the
fit_and_extract(data, "gene-cell", seed=0, n_param_samples=10) product -- the
same call phase2q seed 0 made. Seeds 1/2 are refit on CPU with the identical
call into cache_mu_theta_seed{S}.npz by fit_scvi_cpu.py (this directory).

Existing scripts are imported, never
modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHASE2J = REPO / "experiments" / "phase2j_external_2026-07-09"
PHASE2Q = REPO / "experiments" / "phase2q_biological_2026-07-10"
PHASE2T = REPO / "experiments" / "phase2t_umap_2026-07-10"
for p in (str(REPO), str(PHASE2J), str(PHASE2Q)):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase2j_lib as L  # noqa: E402
from run_biological import tstr  # noqa: E402  (same TSTR machinery as phase2q)
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

RARE_MAX_ORIG = 50          # original split-artefact definition (< 50 test cells)
ABUNDANCE_QUARTILE = 0.25   # abundance-based definition: bottom quartile of corpus totals


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-theta-cache", type=str, default=str(PHASE2T / "cache_mu_theta.npz"),
                    help="npz with mu/theta for THIS seed's scVI gene-cell fit")
    args = ap.parse_args()

    t0 = time.time()
    L.log(f"=== phase2x rarity robustness, seed {args.seed} ===")
    data = L.load_corpus()
    z = np.load(args.mu_theta_cache)
    mu = np.clip(np.asarray(z["mu"], np.float64), 0.0, None)
    theta_ic = np.clip(np.asarray(z["theta"], np.float64), 1e-8, None)
    theta_g = np.median(theta_ic, axis=0)

    te = data["te"]
    real_counts = data["raw"][te]
    lab = data["labels"][te]
    real = _log1p(real_counts)

    # ---- draws: identical to run_biological.py (same seed -> same counts) -----
    draws = {
        "poisson": sample_nb_fixed_theta(mu, float("inf"), seed=args.seed + 1),
        "nb_global": sample_nb(mu, np.broadcast_to(theta_g, mu.shape).copy(), seed=args.seed + 1),
        "nb_conditional": sample_nb(mu, theta_ic, seed=args.seed + 1),
    }
    syn = {k: _log1p(v) for k, v in draws.items()}

    # ---- rarity definitions ---------------------------------------------------
    # (a) original: < RARE_MAX_ORIG held-out test cells
    u_te, c_te = np.unique(lab, return_counts=True)
    test_counts = dict(zip(u_te.tolist(), c_te.tolist()))
    rare_orig = {t for t, n in test_counts.items() if n < RARE_MAX_ORIG}

    # (b) abundance-based: bottom quartile of TRUE corpus abundance (all splits)
    u_all, c_all = np.unique(data["labels"], return_counts=True)
    corpus_counts = dict(zip(u_all.tolist(), c_all.tolist()))
    q1 = float(np.quantile(c_all.astype(np.float64), ABUNDANCE_QUARTILE))
    rare_abund = {t for t, n in corpus_counts.items() if n <= q1}

    L.log(f"rare sets: original(<50 test)={len(rare_orig)} types; "
          f"abundance(<=Q1={q1:.0f} corpus cells)={len(rare_abund)} types; "
          f"overlap={len(rare_orig & rare_abund)}")

    # ---- TSTR: identical split and classifier to run_biological.py ------------
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(real.shape[0])
    half = idx.size // 2
    A, B = idx[:half], idx[half:]

    def run_tstr(rare_types):
        out = {}
        for name, X in syn.items():
            macro, rare_m, rare_f = tstr(X[A], lab[A], real[B], lab[B], rare_types, seed=args.seed)
            out[name] = {"macro_f1": macro, "rare_macro_f1": rare_m,
                         "rare_macro_f1_unmasked": rare_f}
        macro, rare_m, rare_f = tstr(real[A], lab[A], real[B], lab[B], rare_types, seed=args.seed)
        out["real_ceiling"] = {"macro_f1": macro, "rare_macro_f1": rare_m,
                               "rare_macro_f1_unmasked": rare_f}
        return out

    res_orig = run_tstr(rare_orig)
    res_abund = run_tstr(rare_abund)

    # per-class support for the abundance-rare types (test-set counts)
    support = {t: {"corpus_cells": int(corpus_counts[t]),
                   "test_cells": int(test_counts.get(t, 0))}
               for t in sorted(rare_abund, key=lambda t: corpus_counts[t])}

    def ratios(res):
        g = res["nb_global"]["rare_macro_f1"]; c = res["nb_conditional"]["rare_macro_f1"]
        return {"global_over_conditional": (g / c if c > 0 else float("nan")),
                "global": g, "conditional": c}

    cache_path = Path(args.mu_theta_cache)
    try:
        cache_provenance = cache_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        cache_provenance = f"external/{cache_path.name}"

    out = {
        "purpose": "rare-type TSTR robustness: abundance-based rarity (bottom quartile of "
                   "true corpus abundance) vs the original <50-held-out-test-cells definition, "
                   "on the identical mean-fixed draws and TSTR split as phase2q.",
        "seed": args.seed,
        "mu_theta_cache": cache_provenance,
        "rarity_definitions": {
            "original": f"types with < {RARE_MAX_ORIG} held-out test cells (split artefact)",
            "abundance": (f"types with total corpus abundance <= Q1 "
                          f"(the {ABUNDANCE_QUARTILE:.0%} quantile = {q1:.0f} cells) of per-type "
                          f"corpus totals over all {int(len(data['labels']))} cells"),
        },
        "n_types_corpus": int(len(u_all)),
        "rare_types_original": sorted(rare_orig),
        "rare_types_abundance": sorted(rare_abund),
        "rare_set_overlap": sorted(rare_orig & rare_abund),
        "abundance_rare_support": support,
        "tstr_original_definition": res_orig,
        "tstr_abundance_definition": res_abund,
        "headline_ratio_original": ratios(res_orig),
        "headline_ratio_abundance": ratios(res_abund),
        "elapsed_s": time.time() - t0,
    }
    outp = RESULTS / f"rarity_seed{args.seed}.json"
    outp.write_text(json.dumps(out, indent=2))

    # ---- validation against phase2q seed-0 (only when seed==0) ----------------
    if args.seed == 0:
        ref = json.loads((PHASE2Q / "results" / "biological_seed0.json").read_text())
        checks = []
        for arm in ("poisson", "nb_global", "nb_conditional", "real_ceiling"):
            mine = res_orig[arm]["rare_macro_f1"]
            theirs = ref["tstr"][arm]["rare_macro_f1"]
            checks.append({"arm": arm, "phase2x": mine, "phase2q": theirs,
                           "abs_diff": abs(mine - theirs)})
            L.log(f"  [validate] {arm:15s} rare-F1 phase2x={mine:.4f} phase2q={theirs:.4f}")
        out["validation_vs_phase2q_seed0"] = checks
        outp.write_text(json.dumps(out, indent=2))

    for defname, res in (("original", res_orig), ("abundance", res_abund)):
        r = ratios(res)
        L.log(f"  [{defname:9s}] rare-F1 conditional={r['conditional']:.4f} "
              f"global={r['global']:.4f} ratio={r['global_over_conditional']:.3f}")
    L.log(f"=== done -> {outp} ({time.time()-t0:.0f}s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
