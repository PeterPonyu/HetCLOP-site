"""E4b: fit the E4 decoder recipe with larger training sets while validation and test cells stay fixed.

The x1 configuration is identical to the E4 reference corpus (same cells, split, labels, recipe and fit
seeds), so its group tables are taken from e4-sampling-sensitivity/runs/reference_*.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb, train_nb_decoder  # noqa: E402
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

E4 = HERE.parent / 'e4-sampling-sensitivity'
spec = importlib.util.spec_from_file_location('e4run', E4 / 'run.py')
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)
CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
DESIGN = json.loads((HERE / 'design.json').read_text())


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    torch.set_num_threads(4)
    sizes = sys.argv[1].split(',') if len(sys.argv) > 1 else ['x2', 'x3']
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    panel = json.loads((CACHE / 'panel.json').read_text())
    gin, gout = np.array(panel['gin_idx']), np.array(panel['gout_idx'])
    base = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
    labels, datasets, split = meta['labels'].astype(str), meta['dataset_ids'].astype(str), meta['split']
    va, te = split == 1, split == 2
    extra = sp.load_npz(HERE / 'extra_counts.npz').tocsr()
    emeta = np.load(HERE / 'extra_meta.npz', allow_pickle=True)
    cfg = e4.E2['training'].copy()
    cfg.pop('implementation'); cfg.pop('selection')
    for size in sizes:
        pick = emeta['half'] if size == 'x2' else np.ones(extra.shape[0], dtype=bool)
        raw_tr = np.vstack([np.asarray(base[split == 0], dtype=np.float32), extra[pick].toarray().astype(np.float32)])
        raw_va, raw_te = np.asarray(base[va], dtype=np.float32), np.asarray(base[te], dtype=np.float32)
        for arm in ['A', 'B']:
            tgt = (lambda r: r) if arm == 'A' else (lambda r: r[:, gout])
            libf = (lambda r: r.sum(1)) if arm == 'A' else (lambda r: r[:, gin].sum(1))
            inp = (lambda r: np.log1p(r)) if arm == 'A' else (lambda r: np.log1p(r[:, gin]))
            pca = PCA(n_components=512, random_state=0).fit(inp(raw_tr))
            Z_tr, Z_va, Z_te = (pca.transform(inp(r)).astype(np.float32) for r in (raw_tr, raw_va, raw_te))
            real_val, real_te = np.log1p(tgt(raw_va)), np.log1p(tgt(raw_te))
            marker_cols = gout if arm == 'A' else np.arange(len(gout))
            for seed in DESIGN['fit_seeds']:
                out = HERE / 'runs' / f'{size}_{arm}_{seed}'
                if (out / 'complete.json').exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                lib_tr = libf(raw_tr)
                mu, theta, loss, history, nparams = train_nb_decoder(
                    Z_tr, tgt(raw_tr), lib_tr, Z_va, tgt(raw_va), libf(raw_va),
                    np.vstack([Z_va, Z_te]), np.concatenate([libf(raw_va), libf(raw_te)]),
                    mean_libsize=float(lib_tr.mean()), seed=seed, **cfg)
                nv = int(va.sum())
                pval = np.log1p(np.random.default_rng((seed, 202)).poisson(mu[:nv]).astype(np.float32))
                vrows = {}
                for d in np.unique(datasets[va]):
                    for t in np.unique(labels[va][datasets[va] == d]):
                        m = (datasets[va] == d) & (labels[va] == t)
                        if m.sum() >= 20:
                            vrows[(d, t)] = -r_var_per_gene_expressed(real_val[m], pval[m])['expressed']['mean_log2_ratio']
                mte, thte = mu[nv:], theta[nv:]
                shared = np.broadcast_to(np.median(theta[:nv], axis=0), mte.shape)
                draws = {'poisson': np.log1p(np.random.default_rng((seed, 2)).poisson(mte).astype(np.float32)),
                         'shared': np.log1p(sample_nb(mte, shared, seed=seed + 1)),
                         'cell': np.log1p(sample_nb(mte, thte, seed=seed))}
                rows = e4.score_groups(real_te, draws, marker_cols, labels[te], datasets[te])
                for r in rows:
                    r.update({'size': size, 'arm': arm, 'seed': seed,
                              'deficit': vrows.get((r['dataset'], r['cell_type']), np.nan)})
                pd.DataFrame(rows).to_csv(out / 'groups.csv', index=False)
                (out / 'fit.json').write_text(json.dumps({'size': size, 'arm': arm, 'seed': seed, 'n_train': int(len(raw_tr)),
                    'best_validation_nll': loss, 'seconds': time.time() - t0}, indent=2))
                (out / 'complete.json').write_text(json.dumps({'groups': len(rows)}))
                print(time.strftime('%H:%M:%S'), size, arm, seed, 'n_train', len(raw_tr), f'{time.time() - t0:.0f}s', flush=True)
                del mu, theta, draws
            del Z_tr, Z_va, Z_te


if __name__ == '__main__':
    main()
