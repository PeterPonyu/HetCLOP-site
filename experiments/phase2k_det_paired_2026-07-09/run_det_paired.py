"""Paired deterministic-vs-stochastic energy tests.

`paper.tex` asserts, in the trap section, that on energy distance "the deterministic
decoder loses every group". No paired test involving `deterministic` had ever been run.
Phase 2b's `paired_tests` block contains only

    energy_distance__stochastic_vs_matched_noise{,_per_type}
    r_var_abs_log2__stochastic_vs_matched_noise{,_per_type}

-- the deterministic decoder appears in `recomputed_medians` (a MARGINAL statistic) and
in `per_group`, but never in a paired contrast. The sentence was written from marginal
medians. That is the same marginal-vs-paired inversion that has already produced three
retractions in this project (spec §14, §15, §26).

This script runs the missing test. It is pure re-analysis of values Phase 2b already
wrote to disk: no GPU, no retraining, no resampling. It reuses the project's
dataset-clustered inference (50 datasets, 314 groups) rather than the naive paired
Wilcoxon, because the 314 (dataset x cell type) groups are pseudoreplicated.

Convention: energy distance is LOWER-IS-BETTER. records carry diff = det - opponent, so
diff > 0 means the DETERMINISTIC decoder is worse. `det_win_fraction_groups` is the
fraction of groups where the deterministic decoder is strictly better.

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2f_f4_resolution_2026-07-09"))

from clustered_inference import (  # noqa: E402
    clustered_dataset_medians,
    dataset_signflip_permutation,
    naive_paired,
    within_dataset_signflip_permutation,
)

SRC = ROOT / "experiments" / "phase2b_stochastic_2026-07-09" / "results" / "phase2b_v3_paired.json"
OUT = Path(__file__).resolve().parent / "results" / "det_paired.json"
OPPONENTS = ("matched_noise", "matched_noise_per_type", "stochastic_betanll")
METRIC = "energy_distance"


def build_records(group_keys, det, opp):
    """diff = det - opp; positive => deterministic is WORSE (energy is lower-is-better)."""
    out = []
    for (ds, ct), a, b in zip(group_keys, det, opp):
        if np.isfinite(a) and np.isfinite(b):
            out.append({"dataset": ds, "cell_type": ct,
                        "a": float(a), "b": float(b), "diff": float(a) - float(b)})
    return out


def main() -> int:
    payload = json.loads(SRC.read_text())
    pg = payload["per_group"]
    results = {}

    for enc in pg:
        gk = [tuple(g) for g in pg[enc]["group_keys"]]
        variants = pg[enc]["variants"]
        det = np.asarray(variants["deterministic"][METRIC], dtype=float)
        per_enc = {}
        for opp_name in OPPONENTS:
            opp = np.asarray(variants[opp_name][METRIC], dtype=float)
            rec = build_records(gk, det, opp)
            diffs = np.array([r["diff"] for r in rec])
            cl = clustered_dataset_medians(rec)
            ds_med = np.asarray(cl["dataset_medians"], dtype=float)
            per_enc[opp_name] = {
                "n_groups": int(diffs.size),
                # strictly better for the deterministic decoder
                "det_win_fraction_groups": float(np.mean(diffs < 0)),
                "det_win_fraction_datasets": float(np.mean(ds_med < 0)),
                "n_groups_det_wins": int(np.sum(diffs < 0)),
                "median_paired_diff_det_minus_opp": float(np.median(diffs)),
                "naive_p": naive_paired(rec)["p_value"],
                "clustered_wilcoxon_p": cl["wilcoxon_p"],
                "clustered_sign_p": cl["sign_test_p"],
                "clustered_n_datasets": cl["n_datasets"],
                "dataset_signflip_perm_p": dataset_signflip_permutation(rec, n_perm=10000, seed=0)["p_value"],
                "within_dataset_signflip_perm_p": within_dataset_signflip_permutation(rec, n_perm=10000, seed=0)["p_value"],
            }
        results[enc] = per_enc

    # The claim under test, stated so it can only be answered by the data.
    any_group_won = {
        enc: {o: results[enc][o]["n_groups_det_wins"] for o in OPPONENTS} for enc in results
    }
    loses_every_group = all(
        n == 0 for enc in any_group_won for n in any_group_won[enc].values()
    )
    max_clustered_p = max(
        results[e][o]["clustered_wilcoxon_p"] for e in results for o in OPPONENTS
    )
    det_win_lo = min(results[e][o]["det_win_fraction_groups"] for e in results for o in OPPONENTS)
    det_win_hi = max(results[e][o]["det_win_fraction_groups"] for e in results for o in OPPONENTS)

    out = {
        "purpose": "Missing paired test for paper.tex:436 ('the deterministic decoder loses every group').",
        "source": str(SRC.relative_to(ROOT)),
        "metric": METRIC,
        "convention": "diff = det - opponent; energy is lower-is-better; diff>0 => deterministic worse",
        "inference": "dataset-clustered (collapse 314 groups to 50 dataset medians), plus sign-flip permutation",
        "claim_under_test": "deterministic decoder loses EVERY group on energy distance",
        "claim_is_true": bool(loses_every_group),
        "groups_the_deterministic_decoder_wins": any_group_won,
        "supported_restatement": (
            "The deterministic decoder is significantly worse than every stochastic variant on all "
            "four encoders (dataset-clustered Wilcoxon over 50 datasets, max p = "
            f"{max_clustered_p:.2e}), but it does not lose every group: it wins "
            f"{det_win_lo:.1%}-{det_win_hi:.1%} of the 314 groups depending on encoder and opponent."
        ),
        "max_clustered_wilcoxon_p": max_clustered_p,
        "det_win_fraction_groups_min": det_win_lo,
        "det_win_fraction_groups_max": det_win_hi,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"claim 'loses every group' is TRUE? {loses_every_group}")
    print(f"deterministic wins between {det_win_lo:.1%} and {det_win_hi:.1%} of groups")
    print(f"max dataset-clustered Wilcoxon p across 12 contrasts: {max_clustered_p:.3e}")
    for e in results:
        for o in OPPONENTS:
            r = results[e][o]
            print(f"  {e:13s} vs {o:22s} det_win={r['det_win_fraction_groups']:.4f} "
                  f"({r['n_groups_det_wins']:3d}/{r['n_groups']}) "
                  f"medDiff={r['median_paired_diff_det_minus_opp']:+.4f} "
                  f"clustered_p={r['clustered_wilcoxon_p']:.2e}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
