"""Quick probe: assemble the 50-dataset corpus, build the shared 2500 panel,
report cell counts, species, panel coverage, and the G_in/G_out split sizes.
De-risks the data layer before the expensive GPU runs."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.data.real_corpus import assemble_real_corpus
from src.data.shared_gene_panel import canonicalize_symbols, select_shared_panel
from src.orthologs.mapping import load_ortholog_table
from src.utils.paths import CORPUS_SOURCE_ROOT, ORTHOLOG_TABLE_PATH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine_datasets import ALL_DATASETS  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    corpus = assemble_real_corpus(ALL_DATASETS, CORPUS_SOURCE_ROOT)
    ordered = list(corpus["adata_by_dataset"].keys())
    log(f"Assembled {len(ordered)} datasets, {len(corpus['skipped'])} skipped in {time.time()-t0:.1f}s")
    for s in corpus["skipped"]:
        log(f"  SKIP {s['dataset_name']}: {s['reason']}")

    ot = load_ortholog_table(ORTHOLOG_TABLE_PATH)
    human_syms = {}
    per_ds = {}
    for name in ordered:
        ad = corpus["adata_by_dataset"][name]
        sp = corpus["species_by_dataset"][name]
        syms = canonicalize_symbols(list(ad.var_names), sp, ot)
        human_syms[name] = syms
        per_ds[name] = {
            "n_cells": int(ad.n_obs),
            "n_types": int(len(np.unique(corpus["labels"][corpus["dataset_ids"] == name]))),
            "species": sp,
            "n_native_genes": int(ad.n_vars),
            "n_mapped_human": int(sum(1 for s in syms if s is not None)),
        }

    panel, stats = select_shared_panel(human_syms, target_size=2500)
    log(f"Panel size {len(panel)}, coverage stats: {json.dumps(stats['vocab_freq_histogram'])[:200]}")
    log(f"panel min/median/max dataset freq: {stats['panel_min_dataset_freq']}/"
        f"{stats['panel_median_dataset_freq']}/{stats['panel_max_dataset_freq']}")

    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(panel))
    g_in = sorted(np.array(panel)[perm[: len(panel) // 2]].tolist())
    g_out = sorted(np.array(panel)[perm[len(panel) // 2 :]].tolist())
    log(f"G_in={len(g_in)} G_out={len(g_out)}")

    total_cells = sum(v["n_cells"] for v in per_ds.values())
    species_ct = {}
    for v in per_ds.values():
        species_ct[v["species"]] = species_ct.get(v["species"], 0) + 1
    log(f"TOTAL cells: {total_cells}  species datasets: {species_ct}")

    out = {
        "ordered_datasets": ordered,
        "skipped": corpus["skipped"],
        "per_dataset": per_ds,
        "total_cells": total_cells,
        "panel_stats": {k: v for k, v in stats.items() if k != "vocab_freq_histogram"},
        "g_in_size": len(g_in),
        "g_out_size": len(g_out),
    }
    outp = Path(__file__).resolve().parent / "cache" / "probe_summary.json"
    outp.write_text(json.dumps(out, indent=2))
    log(f"Wrote {outp}")


if __name__ == "__main__":
    main()
