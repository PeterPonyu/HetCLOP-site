"""E4b analysis: E4 contrasts at training sizes x1 (the E4 reference fits), x2 and x3, with validation and test
cells fixed. Follows design.json: 2000-resample dataset-cluster bootstrap (seed 20261107); the trend is the paired
x3-minus-x1 difference of dataset effects with its own dataset bootstrap interval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E4 = HERE.parent / 'e4-sampling-sensitivity'
CONTRASTS = [
    ('A', 'cal_cell', 'cal_poisson', 'Arm A per-cell minus Poisson calibration error'),
    ('B', 'cal_cell', 'cal_poisson', 'Arm B per-cell minus Poisson calibration error'),
    ('B', 'cal_cell', 'cal_shared', 'Arm B per-cell minus shared calibration error'),
    ('B', 'welch_cell', 'welch_poisson', 'Arm B per-cell minus Poisson Welch recall'),
    ('B', 'welch_cell', 'welch_shared', 'Arm B per-cell minus shared Welch recall'),
]


def classify(lo, hi):
    return 'below zero' if hi < 0 else ('above zero' if lo > 0 else 'spans zero')


def dataset_effects(frame, a, b):
    g = frame.assign(d=frame[a] - frame[b]).groupby(['dataset', 'cell_type']).d.mean()
    return g.groupby(level=0).mean()


def boot(values, rng):
    x = np.asarray(values, dtype=float)
    b = rng.choice(x, size=(2000, len(x)), replace=True).mean(1)
    lo, hi = np.quantile(b, [.025, .975])
    return float(x.mean()), float(lo), float(hi)


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    x1 = pd.concat([pd.read_csv(p) for p in sorted((E4 / 'runs').glob('reference_*/groups.csv'))], ignore_index=True)
    x1['size'] = 'x1'
    new = pd.concat([pd.read_csv(p.parent / 'groups.csv') for p in sorted((HERE / 'runs').glob('*/complete.json'))], ignore_index=True)
    frame = pd.concat([x1, new], ignore_index=True)
    meta = np.load(HERE.parents[3] / 'experiments/phase2a_decoder_2026-07-08/cache/meta.npz', allow_pickle=True)
    ntrain = {'x1': int((meta['split'] == 0).sum())}
    for p in sorted((HERE / 'runs').glob('*/fit.json')):
        f = json.loads(p.read_text())
        ntrain[f['size']] = f['n_train']
    rows, effects = [], {}
    for size in ['x1', 'x2', 'x3']:
        for arm, a, b, label in CONTRASTS:
            sub = frame[(frame['size'] == size) & (frame.arm == arm)].dropna(subset=[a, b])
            if sub.seed.nunique() < 3:
                continue
            d = dataset_effects(sub, a, b)
            effects[(size, label)] = d
            e, lo, hi = boot(d, np.random.default_rng(20261107))
            rows.append({'size': size, 'contrast': label, 'effect': e, 'ci_low': lo, 'ci_high': hi,
                         'classification': classify(lo, hi), 'n_datasets': len(d), 'n_train': ntrain.get(size)})
    for arm, a, b, label in CONTRASTS:
        if ('x3', label) in effects and ('x1', label) in effects:
            j = pd.concat([effects[('x3', label)], effects[('x1', label)]], axis=1, join='inner').dropna()
            e, lo, hi = boot(j.iloc[:, 0] - j.iloc[:, 1], np.random.default_rng(20261107))
            rows.append({'size': 'x3 minus x1', 'contrast': label, 'effect': e, 'ci_low': lo, 'ci_high': hi,
                         'classification': classify(lo, hi), 'n_datasets': len(j), 'n_train': None})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / 'summary.csv', index=False)
    print(out.round(4).to_string())


if __name__ == '__main__':
    main()
