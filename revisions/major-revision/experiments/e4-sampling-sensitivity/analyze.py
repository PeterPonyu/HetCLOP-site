"""Summarise E4: contrast effects per corpus variant with dataset-cluster
bootstrap intervals, stability against the reference variant, the E2 slope,
and corpus composition."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'design.json').read_text())
spec = importlib.util.spec_from_file_location('e2analyze', HERE.parent / 'e2-dose-response' / 'analyze.py')
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)

CONTRASTS = [
    ('A', 'calibration', 'cal_cell', 'cal_poisson', 'Arm A per-cell minus Poisson calibration error'),
    ('B', 'calibration', 'cal_cell', 'cal_poisson', 'Arm B per-cell minus Poisson calibration error'),
    ('B', 'calibration', 'cal_cell', 'cal_shared', 'Arm B per-cell minus shared calibration error'),
    ('B', 'marker', 'welch_cell', 'welch_poisson', 'Arm B per-cell minus Poisson Welch recall'),
    ('B', 'marker', 'welch_cell', 'welch_shared', 'Arm B per-cell minus shared Welch recall'),
]


def classify(lo, hi):
    return 'below zero' if hi < 0 else ('above zero' if lo > 0 else 'spans zero')


def cluster_effect(frame, a, b, rng, n_boot):
    g = frame.assign(d=frame[a] - frame[b]).groupby(['dataset', 'cell_type'], as_index=False).d.mean()
    per_ds = g.groupby('dataset').d.mean().to_numpy()
    boot = rng.choice(per_ds, size=(n_boot, len(per_ds)), replace=True).mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {'effect': float(per_ds.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'group_fraction_positive': float((g.d > 0).mean()), 'n_groups': len(g), 'n_datasets': len(per_ds)}


def main():
    frame = pd.concat([pd.read_csv(p) for p in sorted((HERE / 'runs').glob('*/groups.csv'))], ignore_index=True)
    variants = [v for v in DESIGN['variants'] if v in set(frame.variant)]
    rows = []
    for v in variants:
        for arm, kind, a, b, label in CONTRASTS:
            sub = frame[(frame.variant == v) & (frame.arm == arm)].dropna(subset=[a, b])
            rng = np.random.default_rng(20261105)
            r = cluster_effect(sub, a, b, rng, 2000)
            rows.append({'variant': v, 'contrast': label, 'arm': arm, 'kind': kind,
                         'n_seeds': sub.seed.nunique(), **r, 'classification': classify(r['ci_low'], r['ci_high'])})
        sub = frame[frame.variant == v].dropna(subset=['deficit']).copy()
        sub['effect'] = sub.cal_cell - sub.cal_poisson
        sub['encoder'] = 'pca_512'
        row, _ = e2.analyze_frame(sub, 'pooled', 1000, 20261105)
        rows.append({'variant': v, 'contrast': 'Deficit-effect slope (pooled arms)', 'arm': 'both', 'kind': 'slope',
                     'n_seeds': sub.seed.nunique(), 'effect': row['slope'], 'ci_low': row['ci_low'],
                     'ci_high': row['ci_high'], 'n_groups': row['n_groups'], 'n_datasets': row['n_datasets'],
                     'group_fraction_positive': np.nan, 'classification': classify(row['ci_low'], row['ci_high'])})
    out = pd.DataFrame(rows)
    ref = out[out.variant == 'reference'].set_index('contrast')
    out['reference_effect'] = out.contrast.map(ref.effect)
    out['difference_from_reference'] = out.effect - out.reference_effect
    out['stable'] = out.classification == out.contrast.map(ref.classification)
    out.to_csv(HERE / 'summary.csv', index=False)
    comp = pd.read_csv(HERE / 'composition.csv')
    test = frame.groupby(['variant', 'dataset']).agg(scored_groups=('cell_type', 'nunique')).reset_index()
    cs = comp[~comp.excluded].groupby('variant').agg(datasets=('dataset', 'size'), cells=('analysed_cells', 'sum'),
        median_types=('cell_types', 'median'), median_batches=('batch_levels_retained', 'median'),
        transferred_share=('transferred_share', 'mean'), batch_fallback=('batch_fallback', 'sum'))
    cs['scored_groups'] = test.groupby('variant').scored_groups.sum()
    cs.to_csv(HERE / 'composition-summary.csv')
    stable = out.groupby('variant').stable.agg(['sum', 'size'])
    ledger = {'design_sha256': (HERE / 'design.sha256').read_text().strip(),
              'variants': variants, 'stability': {v: f'{int(s)}/{int(n)}' for v, (s, n) in stable.iterrows()},
              'summary_rows': len(out)}
    (HERE / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(out[['variant', 'contrast', 'effect', 'ci_low', 'ci_high', 'classification', 'stable']].to_string())
    print(cs.to_string())


if __name__ == '__main__':
    main()
