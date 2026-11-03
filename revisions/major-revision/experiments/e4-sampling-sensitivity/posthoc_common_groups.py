"""POST HOC (not in the pre-execution design): restrict each variant to the
dataset-cell-type groups scored in the reference variant, to separate a change
in training size from a change in the evaluated population."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import importlib.util
H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a', H / 'analyze.py'); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
g = pd.concat([pd.read_csv(p) for p in sorted((H / 'runs').glob('*/groups.csv'))], ignore_index=True)
ref = set(map(tuple, g[g.variant == 'reference'][['arm', 'dataset', 'cell_type']].drop_duplicates().to_numpy()))
keep = np.array([t in ref for t in map(tuple, g[['arm', 'dataset', 'cell_type']].to_numpy())])
rows = []
for v in ['reference', 'cap10000', 'type_floor']:
    for arm, kind, x, y, label in a.CONTRASTS:
        sub = g[keep & (g.variant == v) & (g.arm == arm)].dropna(subset=[x, y])
        r = a.cluster_effect(sub, x, y, np.random.default_rng(20261105), 2000)
        rows.append({'variant': v, 'contrast': label, **r, 'classification': a.classify(r['ci_low'], r['ci_high'])})
out = pd.DataFrame(rows)
out.to_csv(H / 'posthoc-common-groups.csv', index=False)
print(out[['variant', 'contrast', 'n_groups', 'effect', 'ci_low', 'ci_high', 'classification']].to_string())
