"""E3 analysis: per-dataset validation-only choice, scored on test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OPT = ['poisson', 'shared', 'per_type', 'per_cell']
g = pd.read_csv(HERE / 'groups.csv')
key = ['arm', 'encoder', 'seed', 'dataset']
val = g[g.split == 'validation'].groupby(key)[OPT].median()
test = g[g.split == 'test'].groupby(key)[OPT].median()
both = val.join(test, lsuffix='_val', rsuffix='_test', how='inner')
# Ties prefer Poisson, then shared, then per-type, then per-cell (np.argmin returns the first minimum).
choice = np.array(OPT)[np.argmin(both[[f'{o}_val' for o in OPT]].to_numpy(), axis=1)]
both['choice'] = choice
both['chosen_test'] = [both.loc[i, f'{c}_test'] for i, c in zip(both.index, choice)]
both['best_test'] = both[[f'{o}_test' for o in OPT]].min(axis=1)
both['regret'] = both.chosen_test - both.best_test
both['vs_poisson'] = both.chosen_test - both.poisson_test
both['vs_per_cell'] = both.chosen_test - both.per_cell_test
both = both.reset_index()
both.to_csv(HERE / 'choices.csv', index=False)

# Welch marker recall for PCA fits from the matched E1 tables.
e1 = pd.read_csv(HERE.parent / 'e1-armb-biology' / 'groups.csv')
vmap = {'poisson': 'poisson', 'shared': 'shared_per_gene', 'per_type': 'per_cell_type', 'per_cell': 'per_cell'}
w = e1.groupby(['arm', 'seed', 'dataset', 'variant']).welch.mean().unstack('variant')
pca = both[both.encoder == 'pca_512'].copy()
pca['welch_chosen'] = [w.loc[(a, s, d), vmap[c]] if (a, s, d) in w.index else np.nan
                       for a, s, d, c in zip(pca.arm, pca.seed, pca.dataset, pca.choice)]
pca['welch_poisson'] = [w.loc[(a, s, d), 'poisson'] if (a, s, d) in w.index else np.nan for a, s, d in zip(pca.arm, pca.seed, pca.dataset)]
pca['welch_vs_poisson'] = pca.welch_chosen - pca.welch_poisson

rng = np.random.default_rng(20261111)
def boot(series):
    per = series.groupby(level=0).mean().dropna().to_numpy()
    b = rng.choice(per, size=(2000, len(per))).mean(1)
    return {'effect': float(per.mean()), 'ci_low': float(np.quantile(b, .025)), 'ci_high': float(np.quantile(b, .975)), 'n_datasets': int(len(per))}

rows = []
for arm in ['A', 'B']:
    s = both[both.arm == arm].set_index('dataset')
    for col in ['regret', 'vs_poisson', 'vs_per_cell']:
        rows.append({'arm': arm, 'scope': 'all encoders', 'quantity': col, **boot(s[col])})
    freq = s.choice.value_counts(normalize=True).reindex(OPT).fillna(0)
    for o in OPT:
        rows.append({'arm': arm, 'scope': 'all encoders', 'quantity': f'chosen_{o}', 'effect': float(freq[o]),
                     'ci_low': np.nan, 'ci_high': np.nan, 'n_datasets': int(s.index.nunique())})
    rows.append({'arm': arm, 'scope': 'all encoders', 'quantity': 'share_best_option', 'effect': float((s.regret == 0).mean()),
                 'ci_low': np.nan, 'ci_high': np.nan, 'n_datasets': int(s.index.nunique())})
    for col in ['chosen_test', 'poisson_test', 'per_cell_test', 'best_test']:
        rows.append({'arm': arm, 'scope': 'all encoders', 'quantity': col, **boot(s[col])})
    # Descriptive breakdown by encoder (the design pools encoders within arm; this split is reported as descriptive).
    for enc in sorted(s.encoder.unique()):
        e = s[s.encoder == enc]
        for col in ['regret', 'vs_poisson', 'vs_per_cell']:
            rows.append({'arm': arm, 'scope': enc, 'quantity': col, **boot(e[col])})
        fe = e.choice.value_counts(normalize=True).reindex(OPT).fillna(0)
        for o in OPT:
            rows.append({'arm': arm, 'scope': enc, 'quantity': f'chosen_{o}', 'effect': float(fe[o]),
                         'ci_low': np.nan, 'ci_high': np.nan, 'n_datasets': int(e.index.nunique())})
    p = pca[pca.arm == arm].set_index('dataset')
    rows.append({'arm': arm, 'scope': 'PCA-512', 'quantity': 'welch_vs_poisson', **boot(p.welch_vs_poisson)})
    rows.append({'arm': arm, 'scope': 'PCA-512', 'quantity': 'vs_poisson', **boot(p.vs_poisson)})
out = pd.DataFrame(rows)
out.to_csv(HERE / 'summary.csv', index=False)
print(out[out.scope.isin(['all encoders', 'PCA-512'])].to_string())
print('PCA rows without an E1 match:', int(pca.welch_chosen.isna().sum()), 'of', len(pca))
