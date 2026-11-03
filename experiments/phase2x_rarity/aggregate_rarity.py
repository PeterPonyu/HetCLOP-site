"""Phase 2x -- aggregate the abundance-based rarity runs across seeds.

Mirrors the mean/sd convention of phase2q's aggregate.py: per-arm mean and sd
across seeds for the abundance-based rare-type TSTR F1, alongside the original
<50-test-cells definition recomputed on the SAME draws (in-run anchor).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ["poisson", "nb_global", "nb_conditional", "real_ceiling"]


def main() -> int:
    per_seed = {}
    for f in sorted(RESULTS.glob("rarity_seed*.json")):
        d = json.loads(f.read_text())
        per_seed[d["seed"]] = d
    seeds = sorted(per_seed)
    assert seeds, "no rarity_seed*.json found"

    def ms(get):
        vals = np.array([get(per_seed[s]) for s in seeds], dtype=np.float64)
        return [float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0]

    out = {
        "purpose": "seed-aggregate of the abundance-based rarity robustness arm",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "rarity_definitions": per_seed[seeds[0]]["rarity_definitions"],
        "n_rare_types_abundance": len(per_seed[seeds[0]]["rare_types_abundance"]),
        "n_rare_types_original": len(per_seed[seeds[0]]["rare_types_original"]),
        "rare_set_overlap_n": len(per_seed[seeds[0]]["rare_set_overlap"]),
        "abundance_q1_cells": per_seed[seeds[0]]["rarity_definitions"]["abundance"],
        "abundance_rare_support": per_seed[seeds[0]]["abundance_rare_support"],
        "abundance": {a: {"rare_f1_mean_sd": ms(lambda d, a=a: d["tstr_abundance_definition"][a]["rare_macro_f1"]),
                          "rare_f1_unmasked_mean_sd": ms(lambda d, a=a: d["tstr_abundance_definition"][a]["rare_macro_f1_unmasked"]),
                          "macro_f1_mean_sd": ms(lambda d, a=a: d["tstr_abundance_definition"][a]["macro_f1"])}
                      for a in ARMS},
        "original_on_same_draws": {a: {"rare_f1_mean_sd": ms(lambda d, a=a: d["tstr_original_definition"][a]["rare_macro_f1"])}
                                   for a in ARMS},
        "headline_ratio_abundance_mean_sd": ms(
            lambda d: d["headline_ratio_abundance"]["global_over_conditional"]),
        "headline_ratio_original_mean_sd": ms(
            lambda d: d["headline_ratio_original"]["global_over_conditional"]),
    }
    g = out["abundance"]["nb_global"]["rare_f1_mean_sd"][0]
    c = out["abundance"]["nb_conditional"]["rare_f1_mean_sd"][0]
    out["claims"] = {
        "global_more_than_halves_conditional_under_abundance_rarity": bool(g < 0.5 * c),
    }
    outp = RESULTS / "rarity_agg.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"seeds={seeds} abundance rare-F1 global={g:.4f} cond={c:.4f} "
          f"ratio={g/c:.3f} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
