"""Recompute the E7 permutation p-values with the ROBUST signed-rank statistic
and patch them into results/phase2f.json, reusing the per-group arrays already
persisted there (no GPU / no rebuild).

The first run's permutation used a mean-of-dataset-medians statistic, which
energy's heavy-tailed between-dataset magnitudes dominate (one dataset with
|median|~9 vs next ~1), giving an uninformative p~0.5 even when 90% of datasets
agree in sign. ``clustered_inference`` now uses a Wilcoxon signed-rank statistic
(robust to that magnitude heterogeneity); this script recomputes both permutation
fields from the persisted ``per_group`` diffs and rewrites them in place. All
other E7 fields (naive, clustered Wilcoxon, sign test) are unchanged.

"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"

from clustered_inference import (  # noqa: E402
    dataset_signflip_permutation, within_dataset_signflip_permutation,
)

N_PERM = 10000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    path = RESULTS / "phase2f.json"
    out = json.loads(path.read_text())
    comps = out["E7_clustered_inference"]["per_comparison"]
    for c in comps:
        recs = c["per_group"]  # persisted [{dataset,cell_type,a,b,diff}, ...]
        perm = dataset_signflip_permutation(recs, n_perm=N_PERM, seed=0)
        perm_w = within_dataset_signflip_permutation(recs, n_perm=N_PERM, seed=0)
        c["permutation_dataset_signflip"] = {
            "p": perm["p_value"], "statistic": perm["statistic"],
            "observed_stat": perm["observed_stat"],
            "significant_0.05": bool(perm["p_value"] < 0.05)}
        c["permutation_within_dataset_signflip"] = {
            "p": perm_w["p_value"], "statistic": perm_w["statistic"],
            "observed_stat": perm_w["observed_stat"],
            "significant_0.05": bool(perm_w["p_value"] < 0.05)}
        log(f"  [{c['encoder']:12s} s{c['seed']} {c['metric']:20s}] "
            f"permDS_p={perm['p_value']:.4f} permWD_p={perm_w['p_value']:.4f} "
            f"clWil_p={c['clustered_50dataset']['wilcoxon_p']:.3f} "
            f"favors={c['clustered_50dataset']['favors']}")
    out["E7_clustered_inference"]["permutation_statistic"] = (
        "wilcoxon_signed_rank_sum_of_dataset_medians (robust to energy's "
        "between-dataset magnitude outliers; a mean statistic is not)")
    path.write_text(json.dumps(out, indent=2))
    log(f"=== E7 permutations patched in {time.time()-t0:.0f}s -> {path} ===")


if __name__ == "__main__":
    main()
