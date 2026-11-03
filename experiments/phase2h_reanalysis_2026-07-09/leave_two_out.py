"""Leave-two-out recomputation for Phase 2h (T1).

Two datasets in the corpus are species-misnamed (named "Hm"/human but are mouse,
verified against GEO): ``GSE222002_TcellsHmCancer`` (MC38 syngeneic, C57BL/6) and
``GSE168181_BreastHmCancer`` (4T1 mammary, Balb/c). Our gene-symbol species
detector identified both correctly and mapped them via MGI orthologs, so they are
processed correctly -- but a referee will ask whether the load-bearing F4
conclusions depend on them.

This module recomputes F4's paired inference from the persisted per-group arrays
in ``phase2f.json`` (E7 ``per_comparison``), with and without those two datasets,
for both metrics (``energy_distance`` and expression-restricted ``|log2 r_var|``),
both gate-passing encoders (``pca_512``, ``randproj_512``), all three seeds.

All inference reuses the Phase 2f machinery verbatim (import, do not reimplement):
naive paired Wilcoxon, dataset-clustered Wilcoxon + sign test, and the two
sign-flip permutations. The permutation statistic is the Wilcoxon signed-rank sum
of per-dataset median diffs (heavy-tailed-robust), as mandated by the standing
rules.

Holm is applied WITHIN the family of the two gate-passing encoders, within each
(metric, seed) -- the family reported in Table 2 (tab:nb). The reported per-leg
number is the worst (largest) Holm p across the three seeds, matching the
manuscript's convention.

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHASE2F = REPO / "experiments" / "phase2f_f4_resolution_2026-07-09"
sys.path.insert(0, str(PHASE2F))

from clustered_inference import (  # noqa: E402
    clustered_dataset_medians,
    dataset_signflip_permutation,
    naive_paired,
    within_dataset_signflip_permutation,
)

# The two species-misnamed (actually mouse) datasets, verified against GEO.
MISNAMED_MOUSE_DATASETS = (
    "GSE222002_TcellsHmCancer",
    "GSE168181_BreastHmCancer",
)

GATE_ENCODERS = ("pca_512", "randproj_512")
METRICS = ("energy_distance", "r_var_abs_log2_expr")
SEEDS = (42, 43, 44)

# permutation settings that reproduce phase2f.json bit-for-bit (defaults there)
N_PERM = 10000
PERM_SEED = 0


def load_per_comparison(phase2f_json):
    """Index E7 per_comparison by (encoder, seed, metric) -> per_group record list.

    Each per_group entry is {dataset, cell_type, a, b, diff} -- exactly the
    ``records`` shape the clustered_inference functions consume."""
    import json

    with open(phase2f_json) as fh:
        d = json.load(fh)
    idx = {}
    stored = {}
    for c in d["E7_clustered_inference"]["per_comparison"]:
        key = (c["encoder"], int(c["seed"]), c["metric"])
        idx[key] = c["per_group"]
        stored[key] = c
    return idx, stored


def drop_datasets(records, drop):
    """Return records with any group whose dataset is in ``drop`` removed."""
    drop = set(drop)
    return [r for r in records if r["dataset"] not in drop]


def recompute_leg(records, n_perm=N_PERM, perm_seed=PERM_SEED):
    """Run all four inferential views on one (encoder, seed, metric) record list."""
    return {
        "naive_paired": naive_paired(records),
        "clustered_dataset_medians": clustered_dataset_medians(records),
        "permutation_dataset_signflip": dataset_signflip_permutation(
            records, n_perm=n_perm, seed=perm_seed
        ),
        "permutation_within_dataset_signflip": within_dataset_signflip_permutation(
            records, n_perm=n_perm, seed=perm_seed
        ),
    }


def holm_family_over_encoders(raw_p_by_encoder):
    """Holm-correct across the encoder family {pca_512, randproj_512}.

    ``raw_p_by_encoder`` maps encoder -> raw p. Returns encoder -> holm p using
    the standard 2-test step-down (smallest * 2 with cumulative max, clip 1)."""
    from multiplicity import holm

    encs = list(raw_p_by_encoder.keys())
    p = [raw_p_by_encoder[e] for e in encs]
    hp = holm(p)
    return {e: float(v) for e, v in zip(encs, hp)}
