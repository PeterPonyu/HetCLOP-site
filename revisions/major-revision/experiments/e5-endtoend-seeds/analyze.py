"""E5 analysis: ten-seed end-to-end outcomes (archived seeds 1-3 plus new seeds 4-10). Follows design.json:
per arm mean, standard deviation and range; per-seed ordering; paired per-seed differences with the number of
seeds below zero and a seed-level 95% percentile bootstrap (10,000 resamples, seed 20261108)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / 'experiments' / 'phase2v_e2e_2026-07-19' / 'results'
DESIGN = json.loads((HERE / 'design.json').read_text())
ARMS = {'e2e_poisson': 'poisson', 'e2e_nb_gene': 'shared', 'e2e_nb_gene-cell': 'per_cell'}
ENDPOINTS = ['marker_recall_mean', 'tstr_macro_f1', 'tstr_rare_f1', 'tstr_rare_f1_unmasked']


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    seeds = DESIGN['seeds']['archived'] + DESIGN['seeds']['new']
    recs = []
    for s in seeds:
        path = (ARCHIVE if s in DESIGN['seeds']['archived'] else HERE / 'results') / f'e2e_generation_seed{s}.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        for arm, name in ARMS.items():
            recs.append({'seed': s, 'arm': name, 'source': 'archived' if s in DESIGN['seeds']['archived'] else 'new',
                         'n_scoreable_types': d['arms'][arm]['n_scoreable_types'],
                         **{e: d['arms'][arm][e] for e in ENDPOINTS}})
    per_seed = pd.DataFrame(recs)
    per_seed.to_csv(HERE / 'per-seed.csv', index=False)
    n_seeds = per_seed.seed.nunique()
    rows = []
    for e in ENDPOINTS:
        w = per_seed.pivot(index='seed', columns='arm', values=e)
        for arm in ['poisson', 'shared', 'per_cell']:
            rows.append({'endpoint': e, 'quantity': arm, 'mean': w[arm].mean(), 'sd': w[arm].std(ddof=1),
                         'min': w[arm].min(), 'max': w[arm].max(), 'n_seeds': len(w)})
        rng = np.random.default_rng(20261108)
        for a, b in [('per_cell', 'poisson'), ('shared', 'poisson'), ('per_cell', 'shared')]:
            d = (w[a] - w[b]).to_numpy()
            bs = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
            lo, hi = np.quantile(bs, [.025, .975])
            rows.append({'endpoint': e, 'quantity': f'{a} minus {b}', 'mean': d.mean(), 'sd': d.std(ddof=1),
                         'min': d.min(), 'max': d.max(), 'n_seeds': len(d), 'ci_low': lo, 'ci_high': hi,
                         'seeds_below_zero': int((d < 0).sum())})
        order = w[['poisson', 'shared', 'per_cell']].apply(lambda r: ' > '.join(r.sort_values(ascending=False).index), axis=1)
        for o, n in order.value_counts().items():
            rows.append({'endpoint': e, 'quantity': f'ordering {o}', 'mean': np.nan, 'n_seeds': int(n)})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / 'summary.csv', index=False)
    print('seeds available', n_seeds)
    print(out.round(4).to_string())


if __name__ == '__main__':
    main()
