"""Phase 2p (Tier 3) -- dataset-level BOOTSTRAP CIs for the central F4 claim (CPU-only).

Limitation being closed (stated in the manuscript): our three seeds vary only the decoder
INITIALISATION, so they understate generalisation uncertainty across the corpus. This adds a
proper DATA-resampled interval: resample the 50 source datasets with replacement (a cluster
bootstrap that respects the pseudoreplication -- cell types within a dataset share a decoder,
mean, and library-size distribution), and recompute the headline statistic each time.

The headline statistic is the per-gene variance-calibration contest of F4's primary claim: on
the expression-restricted |log2 r_var|, the dispersion-free Poisson draw versus the learned-theta
negative binomial, holding the NB's own fitted mean fixed (pca_512, Arm A). diff = NB - Poisson
(both lower-is-better), so diff > 0 => Poisson better; poisson_win = mean(diff > 0).

Re-analysis only: reads the per-group arrays already stored in the e5 result JSON (314 groups x
3 seeds, with (dataset, cell_type) keys). No GPU, no re-decoding.

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "phase2e_expressed_rvar_2026-07-09" / "results" / "e5_expressed_rvar.json"
OUT = Path(__file__).resolve().parent / "results" / "bootstrap_ci.json"

METRIC = "r_var_abs_log2_expr"   # the F3-mandated expression-restricted statistic
ENC = "pca_512"
B = 10000
RNG_SEED = 20260710


def per_group_diffs(seed_block):
    """Return (datasets[314], diffs[314]) with diff = NB - Poisson on |log2 r_var| (expr)."""
    keys = seed_block["group_keys"]
    nb = np.asarray(seed_block["variants"]["nb_stochastic"][METRIC], dtype=np.float64)
    po = np.asarray(seed_block["variants"]["poisson_on_nb_mean"][METRIC], dtype=np.float64)
    ds = np.array([k[0] for k in keys])
    diff = nb - po
    m = np.isfinite(diff)
    return ds[m], diff[m]


def cluster_bootstrap(ds, diff, rng, b=B):
    """Resample the unique datasets with replacement; recompute win-fraction + median diff."""
    uniq = np.unique(ds)
    # index groups by dataset once
    by_ds = {d: diff[ds == d] for d in uniq}
    wins = np.empty(b)
    meds = np.empty(b)
    n = uniq.size
    for i in range(b):
        pick = uniq[rng.integers(0, n, size=n)]
        pool = np.concatenate([by_ds[d] for d in pick])
        wins[i] = np.mean(pool > 0)          # Poisson beats NB on this group
        meds[i] = np.median(pool)
    return wins, meds


def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def main() -> int:
    d = json.loads(SRC.read_text())
    pg = d["per_group"]["nb_arm"][ENC]
    seeds = sorted(pg)
    rng = np.random.default_rng(RNG_SEED)

    per_seed = {}
    all_ds, all_diff = [], []
    for s in seeds:
        ds, diff = per_group_diffs(pg[s])
        all_ds.append(ds)
        all_diff.append(diff)
        wins, meds = cluster_bootstrap(ds, diff, rng)
        pw = float(np.mean(diff > 0))
        per_seed[s] = {
            "n_groups": int(diff.size),
            "poisson_win_point": pw,
            "poisson_win_ci95": ci(wins),
            "median_diff_point": float(np.median(diff)),
            "median_diff_ci95": ci(meds),
            "ci_excludes_half": bool(ci(wins)[0] > 0.5),   # Poisson-wins majority robust to resampling
        }

    # pooled across the 3 seeds (concatenate; datasets repeat 3x -> still a valid cluster bootstrap
    # over dataset x seed blocks, which is the more conservative reading)
    ds_all = np.concatenate(all_ds)
    diff_all = np.concatenate(all_diff)
    wins, meds = cluster_bootstrap(ds_all, diff_all, rng)
    pooled = {
        "n_groups": int(diff_all.size),
        "poisson_win_point": float(np.mean(diff_all > 0)),
        "poisson_win_ci95": ci(wins),
        "median_diff_point": float(np.median(diff_all)),
        "median_diff_ci95": ci(meds),
        "ci_excludes_half": bool(ci(wins)[0] > 0.5),
    }

    out = {
        "purpose": "Dataset-level cluster bootstrap CI for F4's primary per-gene calibration claim "
                   "(Poisson beats learned-theta NB on expression-restricted |log2 r_var|, pca_512, "
                   "Arm A). Closes the 'seeds vary init only' limitation with a data-resampled interval.",
        "metric": METRIC, "encoder": ENC, "n_bootstrap": B, "convention":
            "diff = NB - Poisson (lower better); diff>0 => Poisson wins; win = mean(diff>0)",
        "per_seed": per_seed, "pooled_over_seeds": pooled,
        "robust": bool(all(v["ci_excludes_half"] for v in per_seed.values()) and pooled["ci_excludes_half"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"F4 primary claim -- dataset cluster bootstrap ({B} resamples), pca_512, {METRIC}")
    for s in seeds:
        v = per_seed[s]
        print(f"  seed {s}: Poisson wins {v['poisson_win_point']:.3f} "
              f"(95% CI {v['poisson_win_ci95'][0]:.3f}-{v['poisson_win_ci95'][1]:.3f}), "
              f"CI>0.5? {v['ci_excludes_half']}")
    v = pooled
    print(f"  pooled : Poisson wins {v['poisson_win_point']:.3f} "
          f"(95% CI {v['poisson_win_ci95'][0]:.3f}-{v['poisson_win_ci95'][1]:.3f})")
    print(f"ROBUST to dataset resampling (all CIs exclude 0.5): {out['robust']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
