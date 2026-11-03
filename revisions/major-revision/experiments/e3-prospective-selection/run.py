"""E3: validation-only dispersion selection on the saved E2 fits."""
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb  # noqa: E402
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

E2 = HERE.parent / 'e2-dose-response'
CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
OPTIONS = ['poisson', 'shared', 'per_type', 'per_cell']


def one(arg):
    arm, enc, seed = arg
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    panel = json.loads((CACHE / 'panel.json').read_text())
    raw = np.load(CACHE / 'raw_full.npy', mmap_mode='r')
    split, lab, ds = meta['split'], meta['labels'].astype(str), meta['dataset_ids'].astype(str)
    va, te = split == 1, split == 2
    cols = np.arange(raw.shape[1]) if arm == 'A' else np.array(panel['gout_idx'])
    src = E2 / 'runs' / f'{arm}_{enc}_{seed}'
    mu = np.load(src / 'mu.npy'); theta = np.load(src / 'theta.npy')
    nv = int(va.sum())
    tg = np.median(theta[:nv], axis=0)
    labv = lab[va]
    by_type = {t: (np.median(theta[:nv][labv == t], axis=0) if (labv == t).sum() >= 20 else tg) for t in np.unique(lab)}
    rows = []
    for split_name, sel, m, th, offset in [('validation', va, mu[:nv], theta[:nv], 0), ('test', te, mu[nv:], theta[nv:], 1)]:
        real = np.log1p(np.asarray(raw[sel][:, cols], dtype=np.float32))
        labs, dss = lab[sel], ds[sel]
        thetas = {'shared': np.broadcast_to(tg, m.shape), 'per_type': np.stack([by_type[t] for t in labs]), 'per_cell': th}
        draws = {'poisson': np.log1p(np.random.default_rng((seed, 30 + offset)).poisson(m).astype(np.float32))}
        for k, v in thetas.items():
            draws[k] = np.log1p(sample_nb(m, v, seed=seed + 31 + offset))
        for d in np.unique(dss):
            for t in np.unique(labs[dss == d]):
                g = (dss == d) & (labs == t)
                if g.sum() < 20:
                    continue
                r = {'arm': arm, 'encoder': enc, 'seed': seed, 'split': split_name, 'dataset': d, 'cell_type': t}
                for k, x in draws.items():
                    r[k] = r_var_per_gene_expressed(real[g], x[g])['expressed']['median_abs_log2_ratio']
                rows.append(r)
    print(arm, enc, seed, len(rows), flush=True)
    return rows


if __name__ == '__main__':
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    jobs = [(a, e, s) for a in ['A', 'B'] for e in ['pca_512', 'randproj_512', 'scgpt_512', 'geneformer'] for s in [42, 43, 44]]
    with Pool(4) as p:
        res = p.map(one, jobs)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(HERE / 'groups.csv', index=False)
    print('done')
