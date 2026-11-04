"""E5 scoring-support check, post hoc (not part of design.json). Uses only the saved per-type marker recall.

The three draws of a seed can score different cell types (69-72), so per-seed marker means may compare different
populations. This recomputes the paired marker differences on (a) the cell types scoreable in all 30 runs (ten seeds
x three draws) and (b) within each seed, the types scoreable in both draws being compared. Inference is the same
seed-level percentile bootstrap as the main analysis (10,000 resamples, seed 20261108); seeds are refits on one corpus,
so the intervals describe seed repeatability, not variation across datasets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / 'experiments' / 'phase2v_e2e_2026-07-19' / 'results'
DESIGN = json.loads((HERE / 'design.json').read_text())
ARMS = {'e2e_poisson': 'poisson', 'e2e_nb_gene': 'shared', 'e2e_nb_gene-cell': 'per_cell'}


def load():
    rec = {}
    for s in DESIGN['seeds']['archived'] + DESIGN['seeds']['new']:
        path = (ARCHIVE if s in DESIGN['seeds']['archived'] else HERE / 'results') / f'e2e_generation_seed{s}.json'
        d = json.loads(path.read_text())
        for arm, name in ARMS.items():
            per = d['arms'][arm]['marker_recall_per_type']
            # Consistency with the saved per-seed mean that the main analysis reports.
            assert abs(np.mean(list(per.values())) - d['arms'][arm]['marker_recall_mean']) < 1e-12, (s, arm)
            rec[s, name] = per
    return rec


def boot(d):
    rng = np.random.default_rng(20261108)
    bs = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
    return np.quantile(bs, [.025, .975])


def main():
    rec = load()
    seeds = sorted({s for s, _ in rec})
    common = set.intersection(*(set(v) for v in rec.values()))
    rows = []
    for a in ['shared', 'per_cell']:
        for support in ['all-runs common', 'per-seed pairwise']:
            diffs, n_types = [], []
            for s in seeds:
                keys = common if support == 'all-runs common' else set(rec[s, a]) & set(rec[s, 'poisson'])
                diffs.append(np.mean([rec[s, a][k] - rec[s, 'poisson'][k] for k in keys]))
                n_types.append(len(keys))
            d = np.array(diffs)
            lo, hi = boot(d)
            rows.append({'comparison': f'{a} minus poisson', 'support': support, 'n_types_min': min(n_types),
                         'n_types_max': max(n_types), 'n_seeds': len(d), 'mean': d.mean(), 'ci_low': lo, 'ci_high': hi,
                         'seeds_below_zero': int((d < 0).sum()), 'max_seed_difference': d.max()})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / 'common-support.csv', index=False)
    (HERE / 'common-support.json').write_text(json.dumps({
        'status': 'post hoc check; uses saved per-type marker recall only; no refit',
        'n_common_types_all_runs': len(common), 'seeds': seeds}, indent=2) + '\n')
    print('types scoreable in all 30 runs:', len(common))
    print(out.round(4).to_string())


if __name__ == '__main__':
    main()
