"""E6: real-data gene-gene correlation error of fixed-mean draws (saved E2 PCA-512 fits)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb  # noqa: E402
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
E2 = HERE.parent / 'e2-dose-response'
CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
panel = json.loads((CACHE / 'panel.json').read_text())
gout = np.array(panel['gout_idx'])
raw = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
split, lab, ds = meta['split'], meta['labels'].astype(str), meta['dataset_ids'].astype(str)
tr, va, te = split == 0, split == 1, split == 2


def corr_err(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):
        ca, cb = np.corrcoef(a, rowvar=False), np.corrcoef(b, rowvar=False)
    iu = np.triu_indices(a.shape[1], 1)
    d = np.abs(ca[iu] - cb[iu])
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else np.nan


rows = []
for arm in ['A', 'B']:
    cols = np.arange(raw.shape[1]) if arm == 'A' else gout
    target = np.asarray(raw[:, cols], dtype=np.float32)
    real_va, real_te = np.log1p(target[va]), np.log1p(target[te])
    for seed in [42, 43, 44]:
        src = E2 / 'runs' / f'{arm}_pca_512_{seed}'
        mu = np.load(src / 'mu.npy', mmap_mode='r'); theta = np.load(src / 'theta.npy', mmap_mode='r')
        nv = int(va.sum())
        mte = np.asarray(mu[nv:]); thte = np.asarray(theta[nv:])
        shared = np.broadcast_to(np.median(np.asarray(theta[:nv]), axis=0), mte.shape)
        draws = {'poisson': np.log1p(np.random.default_rng(seed + 1).poisson(mte).astype(np.float32)),
                 'shared': np.log1p(sample_nb(mte, shared, seed=seed + 1)),
                 'cell': np.log1p(sample_nb(mte, thte, seed=seed + 1))}
        for d in np.unique(ds[te]):
            for t in np.unique(lab[te][ds[te] == d]):
                mt = (ds[te] == d) & (lab[te] == t)
                mv = (ds[va] == d) & (lab[va] == t)
                mtr = tr & (ds == d) & (lab == t)
                if mt.sum() < 20 or mv.sum() < 20 or mtr.sum() < 2:
                    continue
                genes = np.argsort(-target[mtr].mean(0))[:30]
                r = {'arm': arm, 'seed': seed, 'dataset': d, 'cell_type': t, 'n_test': int(mt.sum()),
                     'corr_reference': corr_err(real_va[mv][:, genes], real_te[mt][:, genes])}
                for k, x in draws.items():
                    r[f'corr_{k}'] = corr_err(x[mt][:, genes], real_te[mt][:, genes])
                    r[f'cal_{k}'] = r_var_per_gene_expressed(real_te[mt], x[mt])['expressed']['median_abs_log2_ratio']
                rows.append(r)
        print(arm, seed, len(rows), flush=True)
        del mte, thte, draws

g = pd.DataFrame(rows)
g.to_csv(HERE / 'groups.csv', index=False)
rng = np.random.default_rng(20261110)
out = []
for arm in ['A', 'B']:
    s = g[g.arm == arm]
    gm = s.groupby(['dataset', 'cell_type']).mean(numeric_only=True)
    for a, b, metric in [('cell', 'poisson', 'corr'), ('shared', 'poisson', 'corr'), ('cell', 'shared', 'corr'),
                         ('cell', 'poisson', 'cal'), ('shared', 'poisson', 'cal')]:
        diff = (gm[f'{metric}_{a}'] - gm[f'{metric}_{b}']).dropna()
        per = diff.groupby(level=0).mean().to_numpy()
        bs = rng.choice(per, size=(2000, len(per))).mean(1)
        out.append({'arm': arm, 'metric': metric, 'a': a, 'b': b, 'effect': float(per.mean()),
                    'ci_low': float(np.quantile(bs, .025)), 'ci_high': float(np.quantile(bs, .975)),
                    'group_fraction_positive': float((diff > 0).mean()), 'n_groups': int(diff.size), 'n_datasets': int(per.size)})
    both = ((gm.cal_cell < gm.cal_poisson) & (gm.corr_cell > gm.corr_poisson)).mean()
    out.append({'arm': arm, 'metric': 'tradeoff', 'a': 'cell', 'b': 'poisson',
                'effect': float(both), 'ci_low': np.nan, 'ci_high': np.nan,
                'group_fraction_positive': np.nan, 'n_groups': int(len(gm)), 'n_datasets': int(gm.index.get_level_values(0).nunique())})
    out.append({'arm': arm, 'metric': 'levels', 'a': 'medians', 'b': '',
                'effect': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'group_fraction_positive': np.nan,
                'n_groups': int(len(gm)), 'n_datasets': 0,
                **{f'median_{c}': float(gm[c].median()) for c in ['corr_reference', 'corr_poisson', 'corr_shared', 'corr_cell']}})
pd.DataFrame(out).to_csv(HERE / 'summary.csv', index=False)
print(pd.DataFrame(out).to_string())
