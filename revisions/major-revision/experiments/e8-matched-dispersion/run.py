"""E8: shared-dispersion controls matched in the amount of extra variance (design.json, fixed before execution).

    python run.py original    # reuse the saved E2 PCA-512 fits, arms A and B (CPU)
    python run.py largest     # refit Arm B PCA-512 on the largest training set of E4b, then score (GPU)

Every shared dispersion is computed from validation cells only. All variants of a fit use the same draw seed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb  # noqa: E402

CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
DESIGN = json.loads((HERE / 'design.json').read_text())
LOG_THETA = (-4.0, 8.0)
VARIANTS = ['poisson', 'per_cell', 'shared_median', 'shared_matched', 'shared_fitted']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E1 = load('e1run', EXP / 'e1-armb-biology' / 'run.py')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()

def design_ok() -> bool:
    """The design hash recorded before execution, or, in a released copy whose descriptive note was reworded, the
    released digest that ../released-source-digests.json pairs with that recorded hash."""
    recorded = (HERE / 'design.sha256').read_text().strip()
    now = digest(HERE / 'design.json')
    if now == recorded:
        return True
    rec = HERE.parent / 'released-source-digests.json'
    alt = json.loads(rec.read_text())['files'].get('revisions/major-revision/experiments/e8-matched-dispersion/design.json', {}) \
        if rec.exists() else {}
    return alt.get('analysed') == recorded and alt.get('released') == now



def pooled_loglik(y, mu, log_theta):
    """Negative-binomial log-likelihood summed over cells, per gene, at one theta per gene (terms free of theta dropped)."""
    th = np.exp(log_theta)[None, :]
    return (gammaln(y + th) - gammaln(th) + th * np.log(th) - (th + y) * np.log(th + mu)).sum(0)


def fitted_theta(y, mu):
    """Per-gene maximum-likelihood theta given the fitted means: a coarse grid on log theta, then golden-section search
    within one grid step of the best grid point."""
    y, mu = y.astype(np.float64), np.maximum(mu.astype(np.float64), 1e-8)
    grid = np.arange(LOG_THETA[0], LOG_THETA[1] + 1e-9, 0.5)
    ll = np.stack([pooled_loglik(y, mu, np.full(y.shape[1], g)) for g in grid])
    best = grid[ll.argmax(0)]
    lo, hi = np.maximum(best - 0.5, LOG_THETA[0]), np.minimum(best + 0.5, LOG_THETA[1])
    phi = (np.sqrt(5) - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = pooled_loglik(y, mu, c), pooled_loglik(y, mu, d)
    for _ in range(25):
        left = fc > fd                       # the maximum lies in [lo, d]
        lo, hi = np.where(left, lo, c), np.where(left, d, hi)
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fc, fd = pooled_loglik(y, mu, c), pooled_loglik(y, mu, d)
    return np.exp((lo + hi) / 2)


def shared_thetas(mu_val, theta_val, y_val):
    lo, hi = np.exp(LOG_THETA[0]), np.exp(LOG_THETA[1])
    extra = (mu_val.astype(np.float64) ** 2 / theta_val).sum(0)
    matched = np.clip((mu_val.astype(np.float64) ** 2).sum(0) / np.maximum(extra, 1e-300), lo, hi)
    return {'shared_median': np.median(theta_val, axis=0),
            'shared_matched': matched,
            'shared_fitted': fitted_theta(y_val, mu_val)}


def score_fit(out, mu_te, theta_te, shared, real, lab, ds, rare, seed):
    """Draw every variant with the same seed and score markers, calibration and TSTR as E1."""
    split = np.random.default_rng(seed).permutation(len(lab))
    a, b = split[:len(split) // 2], split[len(split) // 2:]
    markers, pooled, per_ds, variance = [], [], [], {}
    extra_cell = (mu_te.astype(np.float64) ** 2 / theta_te).sum(0)
    for variant in VARIANTS:
        if variant == 'poisson':
            draw = np.log1p(np.random.default_rng(seed + 1).poisson(mu_te).astype(np.float32))
        else:
            theta = theta_te if variant == 'per_cell' else np.broadcast_to(shared[variant], mu_te.shape)
            draw = np.log1p(sample_nb(mu_te, theta, seed=seed + 1))
            if variant != 'per_cell':
                ratio = (mu_te.astype(np.float64) ** 2 / shared[variant]).sum(0) / np.maximum(extra_cell, 1e-300)
                variance[variant] = {'median': float(np.median(ratio)), 'q25': float(np.quantile(ratio, .25)),
                                     'q75': float(np.quantile(ratio, .75))}
        assert draw.shape == real.shape and np.isfinite(draw).all()
        markers.extend({'variant': variant, **r} for r in E1.group_rows(real, draw, lab, ds))
        clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', random_state=seed)
        with warnings.catch_warnings(), threadpool_limits(limits=4):
            warnings.simplefilter('ignore', ConvergenceWarning)
            clf.fit(draw[a], lab[a])
            pred = clf.predict(real[b])
        pooled.append({'variant': variant, **E1.classification_scores(lab[b], pred, rare)})
        for dataset in np.unique(ds[b]):
            keep = ds[b] == dataset
            per_ds.append({'variant': variant, 'dataset': dataset, **E1.classification_scores(lab[b][keep], pred[keep], rare)})
        del draw
    pd.DataFrame(markers).to_csv(out / 'groups.csv', index=False)
    pd.DataFrame(per_ds).to_csv(out / 'dataset-classification.csv', index=False)
    pd.DataFrame(pooled).to_csv(out / 'pooled-classification.csv', index=False)
    np.savez(out / 'shared-theta.npz', **{k: np.asarray(v, dtype=np.float64) for k, v in shared.items()})
    (out / 'achieved-variance.json').write_text(json.dumps(variance, indent=2) + '\n')


def main():
    assert design_ok()
    stage = sys.argv[1]
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    panel = json.loads((CACHE / 'panel.json').read_text())
    gin, gout = np.array(panel['gin_idx']), np.array(panel['gout_idx'])
    split, labels, datasets = meta['split'], meta['labels'].astype(str), meta['dataset_ids'].astype(str)
    va, te = split == 1, split == 2
    nv = int(va.sum())
    raw = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
    y_val = np.asarray(raw[va])[:, gout]
    real = np.log1p(np.asarray(raw[te])[:, gout])
    lab, ds = labels[te], datasets[te]
    u, c = np.unique(lab, return_counts=True)
    rare = set(u[c < 50])
    manifest = {}
    if stage == 'original':
        for arm in ['A', 'B']:
            for seed in [42, 43, 44]:
                out = HERE / 'runs' / f'x1_{arm}_{seed}'
                out.mkdir(parents=True, exist_ok=True)
                src = EXP / 'e2-dose-response' / 'runs' / f'{arm}_pca_512_{seed}'
                for name in ['mu.npy', 'theta.npy']:
                    manifest[str((src / name).relative_to(ROOT))] = digest(src / name)
                if (out / 'complete.json').exists():
                    continue
                t0 = time.time()
                mu, theta = np.load(src / 'mu.npy'), np.load(src / 'theta.npy')
                if arm == 'A':
                    mu, theta = mu[:, gout], theta[:, gout]
                shared = shared_thetas(mu[:nv], theta[:nv], y_val)
                score_fit(out, mu[nv:], theta[nv:], shared, real, lab, ds, rare, seed)
                (out / 'complete.json').write_text(json.dumps({'size': 'x1', 'arm': arm, 'seed': seed,
                                                               'seconds': time.time() - t0}) + '\n')
                print(time.strftime('%H:%M:%S'), 'x1', arm, seed, f'{time.time() - t0:.0f}s', flush=True)
        (HERE / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    elif stage == 'largest':
        import scipy.sparse as sp
        import torch
        from sklearn.decomposition import PCA
        from src.architecture.stochastic_decoder import train_nb_decoder
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        e4 = load('e4run', EXP / 'e4-sampling-sensitivity' / 'run.py')
        e4b = EXP / 'e4b-training-size'
        cfg = e4.E2['training'].copy()
        cfg.pop('implementation'); cfg.pop('selection')
        extra = sp.load_npz(e4b / 'extra_counts.npz').tocsr()
        raw_tr = np.vstack([np.asarray(raw[split == 0], dtype=np.float32), extra.toarray().astype(np.float32)])
        raw_va, raw_te = np.asarray(raw[va], dtype=np.float32), np.asarray(raw[te], dtype=np.float32)
        inp = lambda r: np.log1p(r[:, gin])
        lib = lambda r: r[:, gin].sum(1)
        pca = PCA(n_components=512, random_state=0).fit(inp(raw_tr))
        Z_tr, Z_va, Z_te = (pca.transform(inp(r)).astype(np.float32) for r in (raw_tr, raw_va, raw_te))
        for seed in [42, 43, 44]:
            out = HERE / 'runs' / f'x3_B_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'complete.json').exists():
                continue
            t0 = time.time()
            lib_tr = lib(raw_tr)
            mu, theta, loss, history, nparams = train_nb_decoder(
                Z_tr, raw_tr[:, gout], lib_tr, Z_va, raw_va[:, gout], lib(raw_va),
                np.vstack([Z_va, Z_te]), np.concatenate([lib(raw_va), lib(raw_te)]),
                mean_libsize=float(lib_tr.mean()), seed=seed, **cfg)
            np.save(out / 'mu.npy', mu.astype(np.float32))
            np.save(out / 'theta.npy', theta.astype(np.float32))
            shared = shared_thetas(mu[:nv], theta[:nv], y_val)
            score_fit(out, mu[nv:], theta[nv:], shared, real, lab, ds, rare, seed)
            (out / 'complete.json').write_text(json.dumps({'size': 'x3', 'arm': 'B', 'seed': seed, 'n_train': int(len(raw_tr)),
                'best_validation_nll': float(loss), 'seconds': time.time() - t0}) + '\n')
            print(time.strftime('%H:%M:%S'), 'x3 B', seed, 'n_train', len(raw_tr), f'{time.time() - t0:.0f}s', flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
