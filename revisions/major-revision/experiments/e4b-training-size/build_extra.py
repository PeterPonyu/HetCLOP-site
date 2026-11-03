"""E4b: collect the extra training cells (10,000-cell sample minus the original 3000-cell sample).

The analysed corpus (cells, labels, split) is used unchanged for validation and test. Extra cells are
added to training only. A uniform RandomState(seed).choice(N, k) without replacement is a prefix of the
same seed's permutation, so the 3000-cell sample is the first 3000 entries of the 10,000-cell sample when N > 10,000; for smaller
datasets the larger sample is every cell, and extra cells are those outside the original sample.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
E4 = HERE.parent / 'e4-sampling-sensitivity'
spec = importlib.util.spec_from_file_location('bc', E4 / 'build_corpus.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    meta = np.load(bc.CACHE / 'meta.npz', allow_pickle=True)
    ordered = list(meta['ordered'])
    panel = json.loads((bc.CACHE / 'panel.json').read_text())['panel']
    pindex = {s: j for j, s in enumerate(panel)}
    ot = bc.load_ortholog_table(bc.ORTHOLOG_TABLE_PATH)
    info = json.loads((bc.CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000/metadata_structured.json').read_text())
    blocks, dsids, halves, rows = [], [], [], []
    for name in ordered:
        t0 = time.time()
        proc = ad.read_h5ad(bc.CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000' / f'{name}_processed.h5ad', backed='r')
        plabels = bc.build_cell_type_labels(name, proc, bc.CORPUS_SOURCE_ROOT)
        syms = bc.canonicalize_symbols(list(proc.var_names), bc.detect_species(proc), ot)
        X, obs, var, batch = bc.source_names(info[name]['source_file'])
        labels, ref, _ = bc.transfer_labels(name, X, obs, var, proc, plabels)
        n = len(obs)
        big = bc.uniform(n, 10000, 0)
        small = bc.uniform(n, 3000, 0)
        if n > 10000:
            assert np.array_equal(big[:len(small)], small), f'{name}: prefix property failed'
        # Extra cells: the 10,000-cell sample minus the original 3000-cell sample, in sample order.
        extra = big[~np.isin(big, small)]
        mito = np.char.startswith(var.astype(str), 'MT-') | np.char.startswith(var.astype(str), 'mt-')
        if len(extra):
            extra = extra[bc.fixed_qc(X[extra], mito)]
            extra = extra[labels[extra] != bc.UNLABELED]
        gcol = pd.Index(var).get_indexer(proc.var_names.astype(str))
        P = np.zeros((len(gcol), len(panel)), dtype=np.float32)
        for k, s in enumerate(syms):
            if s is not None and s in pindex:
                P[k, pindex[s]] = 1.0
        if len(extra):
            blocks.append((X[extra][:, gcol] @ sp.csr_matrix(P)).astype(np.float32).tocsr())
            dsids.append(np.full(len(extra), name, dtype=object))
            halves.append(np.random.RandomState(0).permutation(len(extra)) < (len(extra) + 1) // 2)
        n_train = int(((meta['dataset_ids'] == name) & (meta['split'] == 0)).sum())
        rows.append({'dataset': name, 'source_cells': n, 'analysed_train_cells': n_train, 'extra_cells': int(len(extra))})
        print(time.strftime('%H:%M:%S'), name, 'extra', len(extra), f'{time.time() - t0:.0f}s', flush=True)
        del X
    sp.save_npz(HERE / 'extra_counts.npz', sp.vstack(blocks).tocsr())
    np.savez(HERE / 'extra_meta.npz', dataset_ids=np.concatenate(dsids), half=np.concatenate(halves))
    pd.DataFrame(rows).to_csv(HERE / 'training-sizes.csv', index=False)
    print('extra cells', sum(r['extra_cells'] for r in rows))


if __name__ == '__main__':
    main()
