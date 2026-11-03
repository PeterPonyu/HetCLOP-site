from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb
from src.evaluation.distributional import r_var_per_gene_expressed

E2 = HERE.parent / 'e2-dose-response'
CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
DESIGN = json.loads((HERE / 'design.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def scores(X, inside, method, ranks=None):
    ni, no = int(inside.sum()), int((~inside).sum())
    if method == 'auroc':
        return (ranks[inside].sum(0) - ni * (ni + 1) / 2) / (ni * no)
    a, b = X[inside], X[~inside]
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + 1e-8) / ni + (b.var(0) + 1e-8) / no)


def group_rows(real, draw, labels, datasets):
    rows = []
    for ds in np.unique(datasets):
        mask = datasets == ds
        labs, r, s = labels[mask], real[mask], draw[mask]
        types, counts = np.unique(labs, return_counts=True)
        rr, sr = rankdata(r, axis=0), rankdata(s, axis=0)
        for label, count in zip(types, counts):
            if count < 20:
                continue
            inside = labs == label
            cal = r_var_per_gene_expressed(r[inside], s[inside])['expressed']
            row = {'dataset': ds, 'cell_type': label, 'n_test': int(count),
                   'calibration': cal['median_abs_log2_ratio']}
            for method in ['welch', 'auroc']:
                if len(types) < 2:
                    row[method] = np.nan
                    continue
                a = set(np.argsort(-scores(r, inside, method, rr))[:50])
                b = set(np.argsort(-scores(s, inside, method, sr))[:50])
                row[method] = len(a & b) / 50
            rows.append(row)
    return rows


def classification_scores(truth, pred, rare):
    rare = sorted(set(rare) & set(truth))
    mask = np.isin(truth, rare)
    return {'macro_f1': float(f1_score(truth, pred, labels=np.unique(truth), average='macro', zero_division=0)),
            'rare_f1_full': float(f1_score(truth, pred, labels=rare, average='macro', zero_division=0)) if rare else np.nan,
            'rare_f1_masked': float(f1_score(truth[mask], pred[mask], labels=rare, average='macro', zero_division=0)) if rare else np.nan,
            'n_eval': len(truth), 'n_rare_eval': int(mask.sum()), 'n_rare_types': len(rare)}


def main():
    assert digest(HERE / 'design.json') == (HERE / 'design.sha256').read_text().strip()
    assert json.loads((E2 / 'verification.json').read_text())['status'] == 'PASS'
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    va, te = meta['split'] == 1, meta['split'] == 2
    labels = meta['labels'].astype(str)
    ds = meta['dataset_ids'].astype(str)[te]
    lab, labv = labels[te], labels[va]
    panel = json.loads((CACHE / 'panel.json').read_text())
    raw = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
    rawtest = np.asarray(raw[te])[:, panel['gout_idx']]
    real_val = np.log1p(np.asarray(raw[va])[:, panel['gout_idx']])
    dsval = meta['dataset_ids'].astype(str)[va]
    u, c = np.unique(lab, return_counts=True)
    rare = set(u[c < 50])
    source_manifest = {}
    for arm in DESIGN['arms']:
        real = np.log1p(rawtest)
        for seed in DESIGN['seeds']:
            out = HERE / 'runs' / f'{arm}_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            source = E2 / 'runs' / f'{arm}_pca_512_{seed}'
            for name in ['mu.npy', 'theta.npy', 'complete.json']:
                source_manifest[str((source / name).relative_to(ROOT))] = digest(source / name)
            (HERE / 'source-manifest.json').write_text(json.dumps(source_manifest, indent=2))
            if (out / 'complete.json').exists():
                continue
            muall = np.load(source / 'mu.npy', mmap_mode='r')
            thetaall = np.load(source / 'theta.npy', mmap_mode='r')
            if arm == 'A':
                muall = muall[:, panel['gout_idx']]
                thetaall = thetaall[:, panel['gout_idx']]
            nv = int(va.sum())
            valdraw = np.log1p(np.random.default_rng((seed, 202)).poisson(muall[:nv]).astype(np.float32))
            valrows = []
            for dataset in np.unique(dsval):
                for label in np.unique(labv[dsval == dataset]):
                    mask = (dsval == dataset) & (labv == label)
                    if mask.sum() < 20:
                        continue
                    metric = r_var_per_gene_expressed(real_val[mask], valdraw[mask])['expressed']
                    valrows.append({'arm': arm, 'seed': seed, 'encoder': 'pca_512', 'dataset': dataset,
                        'cell_type': label, 'deficit': -metric['mean_log2_ratio'], 'n_validation': int(mask.sum())})
            pd.DataFrame(valrows).to_csv(out / 'validation-deficits.csv', index=False)
            mu = muall[nv:]
            tg = np.median(thetaall[:nv], axis=0)
            by_type = {t: np.median(thetaall[:nv][labv == t], axis=0) if (labv == t).sum() >= 20 else tg for t in np.unique(lab)}
            fallback = sorted(t for t in by_type if (labv == t).sum() < 20)
            split = np.random.default_rng(seed).permutation(len(lab))
            a, b = split[:len(split)//2], split[len(split)//2:]
            assert not set(a) & set(b)
            np.savez(out / 'evaluation-split.npz', train=a, evaluation=b)
            pooled, per_ds, markers, predictions, fit_logs = [], [], [], [], []
            for variant in DESIGN['variants'] + ['real_reference']:
                print(time.strftime('%H:%M:%S'), arm, seed, variant, flush=True)
                if variant == 'poisson':
                    draw = np.log1p(np.random.default_rng(seed + 1).poisson(mu).astype(np.float32))
                elif variant == 'real_reference':
                    draw = real
                else:
                    theta = (np.broadcast_to(tg, mu.shape) if variant == 'shared_per_gene' else
                             np.stack([by_type[t] for t in lab]) if variant == 'per_cell_type' else thetaall[nv:])
                    draw = np.log1p(sample_nb(mu, theta, seed=seed + 1))
                assert draw.shape == real.shape and np.isfinite(draw).all()
                if variant != 'real_reference':
                    markers.extend([{'arm': arm, 'seed': seed, 'variant': variant, **r} for r in group_rows(real, draw, lab, ds)])
                clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', random_state=seed)
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
                    warnings.simplefilter('always', ConvergenceWarning)
                    clf.fit(draw[a], lab[a])
                    pred = clf.predict(real[b])
                fit_logs.append({'variant': variant, 'iterations': clf.n_iter_.tolist(), 'warnings': [str(w.message) for w in caught]})
                pooled.append({'arm': arm, 'seed': seed, 'variant': variant, **classification_scores(lab[b], pred, rare)})
                predictions.extend({'arm': arm, 'seed': seed, 'variant': variant, 'test_index': int(i),
                    'dataset': str(ds[i]), 'truth': str(lab[i]), 'prediction': str(p)} for i, p in zip(b, pred))
                for dataset in np.unique(ds[b]):
                    keep = ds[b] == dataset
                    per_ds.append({'arm': arm, 'seed': seed, 'variant': variant, 'dataset': dataset,
                        **classification_scores(lab[b][keep], pred[keep], rare)})
                del draw
            for name, records in [('groups', markers), ('dataset-classification', per_ds), ('pooled-classification', pooled), ('predictions', predictions)]:
                pd.DataFrame(records).to_csv(out / f'{name}.csv', index=False)
            (out / 'fit-log.json').write_text(json.dumps({'classifier': fit_logs, 'theta_fallback_types': fallback,
                'mu_sha256': digest(source / 'mu.npy'), 'all_variants_share_mu': True,
                'rare_types': sorted(rare), 'n_genes': int(real.shape[1])}, indent=2))
            files = ['validation-deficits.csv', 'groups.csv', 'dataset-classification.csv', 'pooled-classification.csv', 'predictions.csv', 'fit-log.json', 'evaluation-split.npz']
            (out / 'complete.json').write_text(json.dumps({'design_sha256': digest(HERE / 'design.json'),
                'files': {f: digest(out / f) for f in files}}, indent=2))
    for name in ['validation-deficits', 'groups', 'dataset-classification', 'pooled-classification']:
        pd.concat([pd.read_csv(p) for p in sorted((HERE / 'runs').glob(f'*/{name}.csv'))], ignore_index=True).to_csv(HERE / f'{name}.csv', index=False)


if __name__ == '__main__':
    main()
