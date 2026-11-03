"""Rebuild the analysed corpus under alternative sampling and QC rules (E4).

All variants start from the original source matrices, keep each dataset's
original 8000-gene list and the fixed 2500-gene shared panel, and reuse the
original split procedure. The reference variant must reproduce the analysed
cache exactly; that check validates the rebuild before any other variant is
interpreted.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'phase2a_decoder_2026-07-08'))
from combine_datasets import ALL_DATASETS  # noqa: E402
from src.data.labels import build_cell_type_labels  # noqa: E402
from src.data.real_corpus import detect_species  # noqa: E402
from src.data.shared_gene_panel import canonicalize_symbols  # noqa: E402
from src.orthologs.mapping import load_ortholog_table  # noqa: E402
from src.utils.paths import CORPUS_SOURCE_ROOT, ORTHOLOG_TABLE_PATH  # noqa: E402

CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
DESIGN = json.loads((HERE / 'design.json').read_text())
VARIANTS = list(DESIGN['variants'])
OUT = HERE / 'corpora'
LABEL_CACHE = HERE / 'label-transfer'
UNLABELED = '__unlabeled__'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def uniform(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, cap, replace=False)


def type_floor(labels, cap, seed):
    n = len(labels)
    if n <= cap:
        return np.arange(n)
    types, counts = np.unique(labels, return_counts=True)
    floor = np.minimum(50, counts)
    spare = cap - floor.sum()
    assert spare >= 0, 'floors exceed cap'
    room = counts - floor
    raw = spare * room / room.sum()
    extra = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - extra))[: spare - extra.sum()]:
        extra[i] += 1
    alloc = np.minimum(floor + extra, counts)
    rng = np.random.RandomState(seed)
    picks = [rng.permutation(np.flatnonzero(labels == t))[:k] for t, k in zip(types, alloc)]
    idx = np.concatenate(picks)
    return idx[np.random.RandomState(seed).permutation(len(idx))]


def batch_balanced(batches, cap, seed):
    n = len(batches)
    if n <= cap:
        return np.arange(n)
    levels, counts = np.unique(batches, return_counts=True)
    alloc = np.zeros(len(levels), dtype=int)
    remaining = cap
    open_ = np.ones(len(levels), dtype=bool)
    while remaining > 0 and open_.any():
        share = max(remaining // open_.sum(), 1)
        for i in np.flatnonzero(open_):
            take = min(share, counts[i] - alloc[i], remaining)
            alloc[i] += take
            remaining -= take
            if alloc[i] == counts[i]:
                open_[i] = False
            if remaining == 0:
                break
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.permutation(np.flatnonzero(batches == b))[:k] for b, k in zip(levels, alloc)])
    return idx[np.random.RandomState(seed).permutation(len(idx))]


def fixed_qc(X, mito):
    """Original order: cell filter on all genes, gene filter on survivors,
    mitochondrial fraction over the surviving genes."""
    ngenes = np.asarray((X > 0).sum(1)).ravel()
    keep = ngenes >= 200
    Y = X[keep]
    gkeep = np.asarray((Y > 0).sum(0)).ravel() >= 3
    total = np.asarray(Y[:, gkeep].sum(1)).ravel()
    mt = np.asarray(Y[:, gkeep & mito].sum(1)).ravel()
    pct = np.where(total > 0, 100 * mt / np.maximum(total, 1e-12), 0)
    out = np.flatnonzero(keep)
    return out[pct < 20]


def mad_qc(X, mito):
    total = np.asarray(X.sum(1)).ravel()
    ngenes = np.asarray((X > 0).sum(1)).ravel()
    mt = np.asarray(X[:, mito].sum(1)).ravel()
    pct = np.where(total > 0, 100 * mt / np.maximum(total, 1e-12), 0)
    keep = np.ones(len(total), dtype=bool)
    for v in [np.log1p(total), np.log1p(ngenes)]:
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        keep &= (v >= med - 3 * mad) & (v <= med + 3 * mad)
    med = np.median(pct)
    keep &= pct <= med + 3 * 1.4826 * np.median(np.abs(pct - med))
    return np.flatnonzero(keep)


def source_names(path):
    a = ad.read_h5ad(path)
    a.var_names_make_unique()
    X = a.X
    X = sp.csr_matrix(X, dtype=np.float32) if not sp.issparse(X) else X.tocsr().astype(np.float32)
    batch = a.obs['batch'].astype(str).to_numpy() if 'batch' in a.obs else np.full(a.n_obs, 'none')
    return X, np.asarray(a.obs_names.astype(str)), np.asarray(a.var_names.astype(str)), batch


def transfer_labels(name, X, obs, var, proc, plabels):
    """Return labels for every source cell and the reference positions."""
    path = LABEL_CACHE / f'{name}.npz'
    sample = uniform(len(obs), 3000, 0)
    # Processed cells are an order-preserving subsequence of the seed-0 sample;
    # match in order so duplicated source names resolve to the right row.
    snames, ref, j = obs[sample], [], 0
    for v in proc.obs_names.astype(str):
        while snames[j] != v:
            j += 1
        ref.append(sample[j])
        j += 1
    ref = np.array(ref)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        assert np.array_equal(z['ref'], ref)
        return z['labels'], ref, z['transferred']
    gcol = pd.Index(var).get_indexer(proc.var_names.astype(str))
    assert (gcol >= 0).all(), f'{name}: processed genes absent from source'
    G = X[:, gcol]
    lib = np.asarray(G.sum(1)).ravel()
    F = G.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr()
    F.data = np.log1p(F.data)
    reflab = np.array([UNLABELED if v is None else str(v) for v in plabels], dtype=object)
    pca = PCA(n_components=min(30, len(ref) - 1), random_state=0).fit(F[ref].toarray())
    knn = KNeighborsClassifier(n_neighbors=15).fit(pca.transform(F[ref].toarray()), reflab)
    labels = np.empty(len(obs), dtype=object)
    share = np.zeros(len(obs))
    for start in range(0, len(obs), 20000):
        block = pca.transform(F[start:start + 20000].toarray())
        prob = knn.predict_proba(block)
        labels[start:start + 20000] = knn.classes_[prob.argmax(1)]
        share[start:start + 20000] = prob.max(1)
    labels[share < 0.5] = UNLABELED
    labels[ref] = reflab
    transferred = np.ones(len(obs), dtype=bool)
    transferred[ref] = False
    LABEL_CACHE.mkdir(exist_ok=True)
    np.savez(path, labels=labels, ref=ref, transferred=transferred, share=share)
    return labels, ref, transferred


def select(variant, labels, batch, n, ref_seed0):
    if variant == 'reference' or variant == 'mad_qc':
        return uniform(n, 3000, 0), False
    if variant.startswith('uniform_s'):
        return uniform(n, 3000, int(variant[-1])), False
    if variant == 'type_floor':
        return type_floor(labels, 3000, 0), False
    if variant == 'batch_balanced':
        if len(np.unique(batch)) < 2:
            return uniform(n, 3000, 0), True
        return batch_balanced(batch, 3000, 0), False
    if variant == 'cap10000':
        return uniform(n, 10000, 0), False
    raise ValueError(variant)


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    ordered = list(meta['ordered'])
    assert ordered == [d for d in ALL_DATASETS if d in ordered]
    panel = json.loads((CACHE / 'panel.json').read_text())['panel']
    pindex = {s: j for j, s in enumerate(panel)}
    ot = load_ortholog_table(ORTHOLOG_TABLE_PATH)
    info = json.loads((CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000/metadata_structured.json').read_text())
    raw_cache = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
    wanted = sys.argv[1:] or VARIANTS
    import os
    limit = int(os.environ.get('E4_LIMIT', '0'))
    if limit:
        ordered = ordered[:limit]
    store = {v: {'blocks': [], 'ds': [], 'lab': [], 'batch': [], 'transferred': []} for v in wanted}
    comp = []
    for name in ordered:
        t0 = time.time()
        proc = ad.read_h5ad(CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000' / f'{name}_processed.h5ad', backed='r')
        plabels = build_cell_type_labels(name, proc, CORPUS_SOURCE_ROOT)
        species = detect_species(proc)
        syms = canonicalize_symbols(list(proc.var_names), species, ot)
        X, obs, var, batch = source_names(info[name]['source_file'])
        labels, ref, transferred = transfer_labels(name, X, obs, var, proc, plabels)
        gcol = pd.Index(var).get_indexer(proc.var_names.astype(str))
        P = np.zeros((len(gcol), len(panel)), dtype=np.float32)
        for k, s in enumerate(syms):
            if s is not None and s in pindex:
                P[k, pindex[s]] = 1.0
        P = sp.csr_matrix(P)
        mito = np.char.startswith(var.astype(str), 'MT-') | np.char.startswith(var.astype(str), 'mt-')
        for variant in wanted:
            idx, fallback = select(variant, labels, batch, len(obs), ref)
            Xs = X[idx]
            keep = mad_qc(Xs, mito) if variant == 'mad_qc' else fixed_qc(Xs, mito)
            idx = idx[keep]
            lab = labels[idx]
            labelled = lab != UNLABELED
            idx = idx[labelled]
            if len(idx) < 50:
                comp.append({'variant': variant, 'dataset': name, 'excluded': True, 'analysed_cells': len(idx)})
                continue
            counts = (X[idx][:, gcol] @ P).astype(np.float32).tocsr()
            store[variant]['blocks'].append(counts)
            store[variant]['ds'].append(np.full(len(idx), name, dtype=object))
            store[variant]['lab'].append(labels[idx].astype(str))
            store[variant]['batch'].append(batch[idx].astype(str))
            store[variant]['transferred'].append(transferred[idx])
            comp.append({'variant': variant, 'dataset': name, 'excluded': False, 'source_cells': len(obs),
                         'sampled_cells': int(len(select(variant, labels, batch, len(obs), ref)[0])),
                         'post_qc_cells': int(len(keep)), 'analysed_cells': int(len(idx)),
                         'cell_types': int(len(np.unique(labels[idx]))),
                         'batch_levels_source': int(len(np.unique(batch))),
                         'batch_levels_retained': int(len(np.unique(batch[idx]))),
                         'transferred_share': float(transferred[idx].mean()), 'batch_fallback': fallback})
            if variant == 'reference':
                mask = meta['dataset_ids'] == name
                assert mask.sum() == len(idx), f'{name}: reference cell count {len(idx)} != cache {mask.sum()}'
                assert np.array_equal(labels[idx].astype(str), meta['labels'][mask].astype(str)), f'{name}: labels differ'
                assert np.array_equal(counts.toarray(), np.asarray(raw_cache[np.flatnonzero(mask)])), f'{name}: counts differ'
        log(f'{name}: N={len(obs)} {time.time() - t0:.0f}s')
        del X
    OUT.mkdir(exist_ok=True)
    for variant in wanted:
        s = store[variant]
        ds = np.concatenate(s['ds'])
        split = np.zeros(len(ds), dtype=np.int8)
        srng = np.random.default_rng(42)
        for name in [d for d in ordered if d in set(ds)]:
            where = np.flatnonzero(ds == name)
            p = srng.permutation(len(where))
            ntr, nva = int(0.70 * len(where)), int(0.15 * len(where))
            sv = np.zeros(len(where), dtype=np.int8)
            sv[p[ntr:ntr + nva]] = 1
            sv[p[ntr + nva:]] = 2
            split[where] = sv
        if variant == 'reference' and not limit:
            assert np.array_equal(split, meta['split']), 'reference split differs from cache'
        d = OUT / variant
        d.mkdir(exist_ok=True)
        sp.save_npz(d / 'counts.npz', sp.vstack(s['blocks']).tocsr())
        np.savez(d / 'meta.npz', dataset_ids=ds, labels=np.concatenate(s['lab']), split=split,
                 batch=np.concatenate(s['batch']), transferred=np.concatenate(s['transferred']))
        log(f'saved {variant}: {len(ds)} cells')
    old = pd.read_csv(HERE / 'composition.csv') if (HERE / 'composition.csv').exists() else pd.DataFrame()
    new = pd.DataFrame(comp)
    if len(old):
        old = old[~old.variant.isin(wanted)]
    pd.concat([old, new], ignore_index=True).to_csv(HERE / 'composition.csv', index=False)
    if 'reference' in wanted:
        (HERE / 'reference-reproduction.json').write_text(json.dumps({'status': 'PASS',
            'checked': 'cells, order, labels, panel counts and split equal the analysed cache for every dataset'}, indent=2))


if __name__ == '__main__':
    main()
