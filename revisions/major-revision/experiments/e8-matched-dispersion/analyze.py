"""E8 analysis (design.json): paired contrasts between dispersion variants at the same fitted means.

Seeds are averaged within group, groups within dataset, and datasets weighted equally; 95% percentile intervals come
from 10,000 whole-dataset bootstrap resamples (seed 20261110, the same resamples for every contrast of a setting) and
p-values from 10,000 sign flips of the dataset means. Outputs: summary.csv, dataset-effects.csv, joint-direction.csv,
variance.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRASTS = [('per_cell', 'shared_matched'), ('per_cell', 'shared_fitted'), ('per_cell', 'shared_median'),
             ('shared_matched', 'shared_median'), ('per_cell', 'poisson'), ('shared_matched', 'poisson'),
             ('shared_fitted', 'poisson')]
TIE = 1e-12


def runs():
    for d in sorted((HERE / 'runs').glob('x*_*_*')):
        if (d / 'complete.json').exists():
            size, arm, seed = d.name.split('_')
            yield d, size, arm, int(seed)


def load_groups():
    frames = []
    for d, size, arm, seed in runs():
        g = pd.read_csv(d / 'groups.csv')
        g['size'], g['arm'], g['seed'] = size, arm, seed
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def load_classification():
    frames = []
    for d, size, arm, seed in runs():
        g = pd.read_csv(d / 'dataset-classification.csv')
        g['size'], g['arm'], g['seed'] = size, arm, seed
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def dataset_effects(frame, keys, endpoint, a, b):
    """Dataset-level mean of paired a-minus-b differences (pairs share group/dataset and seed)."""
    w = frame.pivot_table(index=keys + ['seed'], columns='variant', values=endpoint)
    diff = (w[a] - w[b]).dropna().rename('d').reset_index()
    inner = [k for k in keys if k != 'dataset']
    if inner:
        diff = diff.groupby(['dataset'] + inner).d.mean().reset_index()
    return diff.groupby('dataset').d.mean()


def summarise(effects, idx, flips):
    x = effects.to_numpy()
    boot = x[idx].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    t = x.mean()
    p = (1 + (np.abs((flips[:, :len(x)] * x).mean(1)) >= abs(t) - 1e-15).sum()) / (len(flips) + 1)
    return {'effect': t, 'ci_low': lo, 'ci_high': hi, 'p_signflip': p, 'n_datasets': len(x)}


def main():
    groups, cls = load_groups(), load_classification()
    rows, eff_rows, joint_rows = [], [], []
    for (size, arm), sub in groups.groupby(['size', 'arm']):
        datasets = np.array(sorted(sub.dataset.unique()))
        rng = np.random.default_rng(20261110)
        idx_all = rng.integers(len(datasets), size=(10000, len(datasets)))
        flips = np.random.default_rng(20261111).choice([-1.0, 1.0], size=(10000, len(datasets)))
        csub = cls[(cls['size'] == size) & (cls.arm == arm)]
        effects = {}
        for a, b in CONTRASTS:
            for endpoint, frame, keys in [('calibration', sub, ['dataset', 'cell_type']),
                                          ('welch', sub, ['dataset', 'cell_type']),
                                          ('auroc', sub, ['dataset', 'cell_type']),
                                          ('macro_f1', csub, ['dataset']),
                                          ('rare_f1_full', csub, ['dataset']),
                                          ('rare_f1_masked', csub, ['dataset'])]:
                e = dataset_effects(frame, keys, endpoint, a, b)
                if endpoint.startswith('rare'):
                    # Datasets with rare types in every seed (as E1).
                    present = frame.dropna(subset=[endpoint]).groupby('dataset').seed.nunique()
                    e = e[e.index.isin(present[present == frame.seed.nunique()].index)]
                # Endpoints scored on fewer datasets (rare types) get their own resamples from the same seed.
                idx = idx_all if len(e) == len(datasets) else \
                    np.random.default_rng(20261110).integers(len(e), size=(10000, len(e)))
                s = summarise(e, idx, flips)
                rows.append({'size': size, 'arm': arm, 'contrast': f'{a} minus {b}', 'endpoint': endpoint, **s})
                effects[(a, b, endpoint)] = e
                eff_rows.extend({'size': size, 'arm': arm, 'contrast': f'{a} minus {b}', 'endpoint': endpoint,
                                 'dataset': k, 'effect': v} for k, v in e.items())
        for a, b in CONTRASTS:
            cal, wel = effects[(a, b, 'calibration')], effects[(a, b, 'welch')]
            j = pd.concat([cal.rename('cal'), wel.rename('wel')], axis=1).dropna()
            better_cal = j.cal < -TIE
            joint_rows.append({'size': size, 'arm': arm, 'contrast': f'{a} minus {b}', 'n_datasets': len(j),
                               'calibration_lower': int(better_cal.sum()),
                               'calibration_lower_recall_higher': int((better_cal & (j.wel > TIE)).sum()),
                               'calibration_lower_recall_lower': int((better_cal & (j.wel < -TIE)).sum()),
                               'calibration_lower_recall_tied': int((better_cal & (j.wel.abs() <= TIE)).sum()),
                               'recall_higher': int((j.wel > TIE).sum())})
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(eff_rows).to_csv(HERE / 'dataset-effects.csv', index=False)
    pd.DataFrame(joint_rows).to_csv(HERE / 'joint-direction.csv', index=False)
    var_rows = []
    for d, size, arm, seed in runs():
        for variant, v in json.loads((d / 'achieved-variance.json').read_text()).items():
            var_rows.append({'size': size, 'arm': arm, 'seed': seed, 'variant': variant, **v})
    pd.DataFrame(var_rows).to_csv(HERE / 'variance.csv', index=False)
    out = pd.DataFrame(rows)
    show = out[out.endpoint.isin(['calibration', 'welch'])]
    print(show.round(4).to_string(index=False))
    print(pd.DataFrame(joint_rows).to_string(index=False))
    print(pd.DataFrame(var_rows).groupby(['size', 'arm', 'variant'])[['median', 'q25', 'q75']].mean().round(3))


if __name__ == '__main__':
    main()
