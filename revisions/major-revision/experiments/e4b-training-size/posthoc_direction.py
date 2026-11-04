"""Post hoc descriptive check added after the second review round (not part of design.json), from the saved group
tables of the largest training set in Arm B: (1) in how many datasets per-cell dispersion lowered calibration error
and, among those, how marker recall moved, against Poisson and against the shared per-gene value; (2) how much of the
net per-cell-minus-Poisson calibration change one dataset contributes, and the contrast without it (dataset bootstrap,
10,000 resamples, seed 20261113). Seeds are averaged within group and groups within dataset; ties are differences
within 1e-12."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIE = 1e-12


def main():
    g = pd.concat([pd.read_csv(f) for f in sorted((HERE / 'runs').glob('x3_B_*/groups.csv'))])
    out = {'n_groups': int(g[['dataset', 'cell_type']].drop_duplicates().shape[0]), 'n_datasets': int(g.dataset.nunique())}
    for comp in ['poisson', 'shared']:
        g['cal'] = g.cal_cell - g[f'cal_{comp}']
        g['wel'] = g.welch_cell - g[f'welch_{comp}']
        d = g.groupby(['dataset', 'cell_type'])[['cal', 'wel']].mean().groupby('dataset').mean()
        lower = d.cal < -TIE
        out[comp] = {'calibration_lower': int(lower.sum()),
                     'recall_lower': int((lower & (d.wel < -TIE)).sum()),
                     'recall_higher': int((lower & (d.wel > TIE)).sum()),
                     'recall_tied': int((lower & (d.wel.abs() <= TIE)).sum()),
                     'mean_calibration': float(d.cal.mean()), 'mean_recall': float(d.wel.mean()),
                     'markers_of_50': float(50 * d.wel.mean())}
        if comp == 'poisson':
            share = d.cal / d.cal.sum()
            top = share.idxmax()
            rest = d.drop(top).cal.to_numpy()
            rng = np.random.default_rng(20261113)
            boot = rest[rng.integers(len(rest), size=(10000, len(rest)))].mean(1)
            out['influence'] = {'dataset': top, 'share_of_net_calibration_change': float(share.max()),
                                'without': float(rest.mean()), 'without_ci': [float(x) for x in np.quantile(boot, [.025, .975])]}
    (HERE / 'posthoc-direction.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
