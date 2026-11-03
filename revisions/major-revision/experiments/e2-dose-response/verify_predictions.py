from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb
from src.evaluation.distributional import r_var_per_gene_expressed


def main():
    cache = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
    meta = np.load(cache / 'meta.npz', allow_pickle=True)
    raw = np.load(cache / 'raw_full.npy', mmap_mode='r')
    panel = json.loads((cache / 'panel.json').read_text())
    va, te = meta['split'] == 1, meta['split'] == 2
    vlab, vds = meta['labels'][va].astype(str), meta['dataset_ids'][va].astype(str)
    tlab, tds = meta['labels'][te].astype(str), meta['dataset_ids'][te].astype(str)
    allrows = pd.read_csv(HERE / 'groups.csv')
    n_validation, n_effects = 0, 0
    records = []
    for (arm, encoder, seed), group in allrows.groupby(['arm', 'encoder', 'seed']):
        row = group.iloc[0]
        out = HERE / 'runs' / f'{arm}_{encoder}_{seed}'
        mu = np.load(out / 'mu.npy', mmap_mode='r')
        vi = np.flatnonzero((vlab == row.cell_type) & (vds == row.dataset))
        prefix = int(vi.max() + 1)
        target = raw[np.flatnonzero(va)[vi]]
        if arm == 'B':
            target = target[:, panel['gout_idx']]
        draw = np.random.default_rng((seed, 202)).poisson(mu[:prefix]).astype(np.float32)
        observed = r_var_per_gene_expressed(np.log1p(target), np.log1p(draw[vi]))['expressed']
        assert abs(-observed['mean_log2_ratio'] - row.deficit) < 1e-10
        n_validation += 1
        if encoder == 'pca_512' and seed == 42:
            theta = np.load(out / 'theta.npy', mmap_mode='r')
            nv = int(va.sum())
            pmu, ptheta = mu[nv:], theta[nv:]
            nb = sample_nb(pmu, ptheta, seed=seed)
            poisson = np.random.default_rng((seed, 2)).poisson(pmu).astype(np.float32)
            ti = np.flatnonzero((tlab == row.cell_type) & (tds == row.dataset))
            real = raw[np.flatnonzero(te)[ti]]
            if arm == 'B':
                real = real[:, panel['gout_idx']]
            a = r_var_per_gene_expressed(np.log1p(real), np.log1p(nb[ti]))['expressed']['median_abs_log2_ratio']
            b = r_var_per_gene_expressed(np.log1p(real), np.log1p(poisson[ti]))['expressed']['median_abs_log2_ratio']
            assert abs(a - b - row.effect) < 1e-10
            n_effects += 1
        records.append({'arm': arm, 'encoder': encoder, 'seed': int(seed), 'dataset': row.dataset, 'cell_type': row.cell_type, 'validation_deficit_reproduced': True})
    report = {'status': 'PASS', 'fit_validation_predictors_replayed': n_validation,
              'test_effects_replayed_from_full_draws': n_effects, 'records': records}
    (HERE / 'prediction-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k,v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
