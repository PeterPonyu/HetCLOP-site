from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
for name in ['phase2a_decoder_2026-07-08', 'phase2b_stochastic_2026-07-09']:
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from run_phase2b_v2 import load_geneformer_armA
from src.architecture.stochastic_decoder import sample_nb, train_nb_decoder
from src.evaluation.distributional import r_var_per_gene_expressed

CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
DESIGN = json.loads((HERE / 'design.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S'), message, flush=True)


def group_metrics(real, draw, labels, datasets):
    rows = {}
    for ds in np.unique(datasets):
        for label in np.unique(labels[datasets == ds]):
            mask = (datasets == ds) & (labels == label)
            n = int(mask.sum())
            if n < 20:
                continue
            val = r_var_per_gene_expressed(real[mask], draw[mask])['expressed']
            rows[(str(ds), str(label))] = {'n': n, **val}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['A', 'B'])
    ap.add_argument('--encoder', choices=DESIGN['encoders'])
    ap.add_argument('--seed', type=int, choices=DESIGN['seeds'])
    args = ap.parse_args()
    assert digest(HERE / 'design.json') == (HERE / 'design.sha256').read_text().strip()
    assert torch.cuda.is_available(), 'GPU required by the run design'
    torch.set_num_threads(4)
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    split = meta['split']
    tr, va, te = split == 0, split == 1, split == 2
    assert not np.any(va & te)
    labels = meta['labels'].astype(str)
    datasets = meta['dataset_ids'].astype(str)
    panel = json.loads((CACHE / 'panel.json').read_text())
    gin, gout = np.array(panel['gin_idx']), np.array(panel['gout_idx'])
    assert not set(gin) & set(gout)
    raw = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
    logX = np.load(CACHE / 'logX_full.npy', mmap_mode='r')
    arms = [args.arm] if args.arm else DESIGN['arms']
    encoders = [args.encoder] if args.encoder else DESIGN['encoders']
    seeds = [args.seed] if args.seed is not None else DESIGN['seeds']
    inputs = ['raw_full.npy', 'logX_full.npy', 'meta.npz', 'panel.json', 'enc_A_scgpt.npy', 'enc_B_scgpt.npy', 'enc_B_geneformer.npy']
    if not (HERE / 'input-data-manifest.json').exists():
        (HERE / 'input-data-manifest.json').write_text(json.dumps({f: digest(CACHE / f) for f in inputs}, indent=2))
    for arm in arms:
        target = np.asarray(raw if arm == 'A' else raw[:, gout], dtype=np.float32)
        real_val, real_test = np.log1p(target[va]), np.log1p(target[te])
        lib = np.asarray(raw.sum(axis=1) if arm == 'A' else raw[:, gin].sum(axis=1), dtype=np.float32)
        Xin = np.asarray(logX if arm == 'A' else logX[:, gin], dtype=np.float32)
        for encoder in encoders:
            paths = [HERE / 'runs' / f'{arm}_{encoder}_{seed}' for seed in seeds]
            if all((p / 'complete.json').exists() for p in paths):
                log(f'skip verified completed {arm} {encoder}')
                continue
            log(f'prepare {arm} {encoder}')
            if encoder == 'pca_512':
                Z = PCA(n_components=512, random_state=0).fit(Xin[tr]).transform(Xin).astype(np.float32)
            elif encoder == 'randproj_512':
                W = np.random.default_rng(0).normal(0, 1 / np.sqrt(512), size=(Xin.shape[1], 512)).astype(np.float32)
                Z = (Xin @ W).astype(np.float32)
            elif encoder == 'geneformer' and arm == 'A':
                Z = load_geneformer_armA(list(meta['ordered'].astype(str)))
                assert Z is not None
            else:
                fm = 'scgpt' if encoder == 'scgpt_512' else 'geneformer'
                Z = np.load(CACHE / f'enc_{arm}_{fm}.npy')
            assert len(Z) == len(raw)
            for seed, out in zip(seeds, paths):
                out.mkdir(parents=True, exist_ok=True)
                if (out / 'complete.json').exists():
                    continue
                log(f'fit {arm} {encoder} seed={seed}')
                start = time.time()
                config = DESIGN['training'].copy()
                config.pop('implementation'); config.pop('selection')
                mu, theta, loss, history, nparams = train_nb_decoder(
                    Z[tr], target[tr], lib[tr], Z[va], target[va], lib[va],
                    np.vstack([Z[va], Z[te]]), np.concatenate([lib[va], lib[te]]),
                    mean_libsize=float(lib[tr].mean()), seed=seed, **config)
                assert np.isfinite(mu).all() and np.isfinite(theta).all()
                assert (mu >= 0).all() and (theta > 0).all()
                np.save(out / 'mu.npy', mu)
                np.save(out / 'theta.npy', theta)
                (out / 'fit.json').write_text(json.dumps({'arm': arm, 'encoder': encoder, 'seed': seed,
                    'best_validation_nll': loss, 'history': history, 'n_parameters': nparams,
                    'n_validation': int(va.sum()), 'n_test': int(te.sum()), 'n_genes': target.shape[1],
                    'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
                    'seconds': time.time() - start}, indent=2))
                nv = int(va.sum())
                pval = np.log1p(np.random.default_rng((seed, 202)).poisson(mu[:nv]).astype(np.float32))
                ptest = np.log1p(np.random.default_rng((seed, 2)).poisson(mu[nv:]).astype(np.float32))
                nbtest = np.log1p(sample_nb(mu[nv:], theta[nv:], seed=seed))
                vm = group_metrics(real_val, pval, labels[va], datasets[va])
                pm = group_metrics(real_test, ptest, labels[te], datasets[te])
                nm = group_metrics(real_test, nbtest, labels[te], datasets[te])
                rows, excluded = [], []
                for key in sorted(set(vm) | set(pm)):
                    if key not in vm or key not in pm:
                        excluded.append({'dataset': key[0], 'cell_type': key[1], 'reason': 'group has fewer than20cells in validation or test'})
                        continue
                    v, p, n = vm[key], pm[key], nm[key]
                    values = [v['mean_log2_ratio'], p['median_abs_log2_ratio'], n['median_abs_log2_ratio']]
                    if not np.isfinite(values).all():
                        excluded.append({'dataset': key[0], 'cell_type': key[1], 'reason': 'no expressed variable genes'})
                        continue
                    rows.append({'arm': arm, 'encoder': encoder, 'seed': seed, 'dataset': key[0], 'cell_type': key[1],
                        'n_validation': v['n'], 'n_test': p['n'], 'validation_genes': v['n_genes_scored'],
                        'test_genes': p['n_genes_scored'], 'deficit': -v['mean_log2_ratio'],
                        'poisson_calibration': p['median_abs_log2_ratio'], 'nb_calibration': n['median_abs_log2_ratio'],
                        'effect': n['median_abs_log2_ratio'] - p['median_abs_log2_ratio']})
                pd.DataFrame(rows).to_csv(out / 'groups.csv', index=False)
                (out / 'excluded.json').write_text(json.dumps(excluded, indent=2))
                (out / 'complete.json').write_text(json.dumps({'design_sha256': digest(HERE / 'design.json'),
                    'files': {name: digest(out / name) for name in ['mu.npy', 'theta.npy', 'fit.json', 'groups.csv', 'excluded.json']}}, indent=2))
                log(f'done {arm} {encoder} seed={seed} groups={len(rows)} elapsed={time.time()-start:.1f}s')
                del mu, theta, pval, ptest, nbtest
            del Z
    files = sorted((HERE / 'runs').glob('*/groups.csv'))
    pd.concat([pd.read_csv(p) for p in files], ignore_index=True).to_csv(HERE / 'groups.csv', index=False)
    log(f'complete {len(files)} fit records')


if __name__ == '__main__':
    main()
