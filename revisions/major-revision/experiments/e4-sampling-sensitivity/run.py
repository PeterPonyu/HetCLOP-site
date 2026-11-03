"""Fit the E2 decoder recipe on each E4 corpus variant and score the
fixed-mean dispersion contrasts per (dataset, cell type) group."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb, train_nb_decoder  # noqa: E402
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
DESIGN = json.loads((HERE / 'design.json').read_text())
E2 = json.loads((HERE.parent / 'e2-dose-response' / 'design.json').read_text())


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def welch(X, inside):
    a, b = X[inside], X[~inside]
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + 1e-8) / inside.sum() + (b.var(0) + 1e-8) / (~inside).sum())


def score_groups(real, draws, marker_cols, labels, datasets, min_cells=20):
    rows = []
    for ds in np.unique(datasets):
        m = datasets == ds
        labs = labels[m]
        types, counts = np.unique(labs, return_counts=True)
        r = real[m]
        dd = {k: v[m] for k, v in draws.items()}
        for t, c in zip(types, counts):
            if c < min_cells:
                continue
            inside = labs == t
            row = {'dataset': ds, 'cell_type': t, 'n_test': int(c)}
            top_real = set(np.argsort(-welch(r[:, marker_cols], inside))[:50]) if len(types) > 1 else None
            for k, d in dd.items():
                row[f'cal_{k}'] = r_var_per_gene_expressed(r[inside], d[inside])['expressed']['median_abs_log2_ratio']
                row[f'welch_{k}'] = (len(top_real & set(np.argsort(-welch(d[:, marker_cols], inside))[:50])) / 50
                                     if top_real is not None else np.nan)
            rows.append(row)
    return rows


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    variant = sys.argv[1]
    arms = sys.argv[2].split(',') if len(sys.argv) > 2 else ['A', 'B']
    panel = json.loads((CACHE / 'panel.json').read_text())
    gin, gout = np.array(panel['gin_idx']), np.array(panel['gout_idx'])
    d = HERE / 'corpora' / variant
    meta = np.load(d / 'meta.npz', allow_pickle=True)
    raw = sp.load_npz(d / 'counts.npz').toarray().astype(np.float32)
    labels, datasets, split = meta['labels'].astype(str), meta['dataset_ids'].astype(str), meta['split']
    tr, va, te = split == 0, split == 1, split == 2
    logX = np.log1p(raw)
    cfg = E2['training'].copy()
    cfg.pop('implementation'); cfg.pop('selection')
    for arm in arms:
        target = raw if arm == 'A' else raw[:, gout]
        marker_cols = gout if arm == 'A' else np.arange(len(gout))
        lib = raw.sum(1) if arm == 'A' else raw[:, gin].sum(1)
        Xin = logX if arm == 'A' else logX[:, gin]
        Z = PCA(n_components=512, random_state=0).fit(Xin[tr]).transform(Xin).astype(np.float32)
        real_val, real_te = np.log1p(target[va]), np.log1p(target[te])
        for seed in DESIGN['fit_seeds']:
            out = HERE / 'runs' / f'{variant}_{arm}_{seed}'
            if (out / 'complete.json').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            mu, theta, loss, history, nparams = train_nb_decoder(
                Z[tr], target[tr], lib[tr], Z[va], target[va], lib[va],
                np.vstack([Z[va], Z[te]]), np.concatenate([lib[va], lib[te]]),
                mean_libsize=float(lib[tr].mean()), seed=seed, **cfg)
            assert np.isfinite(mu).all() and np.isfinite(theta).all()
            nv = int(va.sum())
            # Validation-only deficit (E2 definition).
            pval = np.log1p(np.random.default_rng((seed, 202)).poisson(mu[:nv]).astype(np.float32))
            vrows = {}
            for ds in np.unique(datasets[va]):
                for t in np.unique(labels[va][datasets[va] == ds]):
                    m = (datasets[va] == ds) & (labels[va] == t)
                    if m.sum() >= 20:
                        vrows[(ds, t)] = -r_var_per_gene_expressed(real_val[m], pval[m])['expressed']['mean_log2_ratio']
            mte, thte = mu[nv:], theta[nv:]
            shared = np.broadcast_to(np.median(theta[:nv], axis=0), mte.shape)
            draws = {
                'poisson': np.log1p(np.random.default_rng((seed, 2)).poisson(mte).astype(np.float32)),
                'shared': np.log1p(sample_nb(mte, shared, seed=seed + 1)),
                'cell': np.log1p(sample_nb(mte, thte, seed=seed)),
            }
            rows = score_groups(real_te, draws, marker_cols, labels[te], datasets[te])
            for r in rows:
                r.update({'variant': variant, 'arm': arm, 'seed': seed,
                          'deficit': vrows.get((r['dataset'], r['cell_type']), np.nan)})
            pd.DataFrame(rows).to_csv(out / 'groups.csv', index=False)
            (out / 'fit.json').write_text(json.dumps({'variant': variant, 'arm': arm, 'seed': seed,
                'best_validation_nll': loss, 'n_parameters': nparams, 'n_train': int(tr.sum()),
                'n_validation': nv, 'n_test': int(te.sum()), 'seconds': time.time() - t0,
                'device': torch.cuda.get_device_name(), 'torch': torch.__version__}, indent=2))
            (out / 'complete.json').write_text(json.dumps({'groups': len(rows)}))
            log(f'{variant} {arm} seed={seed} groups={len(rows)} {time.time() - t0:.0f}s')
            del mu, theta, draws
        del Z


if __name__ == '__main__':
    main()
