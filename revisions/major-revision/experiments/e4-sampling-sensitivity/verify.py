"""E4 verification: design integrity, completeness, reference reproduction,
finite metrics, and an independent recomputation of the summary effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
design = json.loads((HERE / 'design.json').read_text())
checks = {}
checks['design_hash'] = hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
checks['reference_reproduces_cache'] = json.loads((HERE / 'reference-reproduction.json').read_text())['status'] == 'PASS'
want = {(v, a, s) for v in design['variants'] for a in 'AB' for s in design['fit_seeds']}
have = {tuple(p.parent.name.rsplit('_', 2)) for p in (HERE / 'runs').glob('*/complete.json')}
have = {(v, a, int(s)) for v, a, s in have}
checks['all_fits_complete'] = have == want
frames = [pd.read_csv(p) for p in sorted((HERE / 'runs').glob('*/groups.csv'))]
g = pd.concat(frames, ignore_index=True)
checks['finite_primary_metrics'] = bool(np.isfinite(g[['cal_poisson', 'cal_shared', 'cal_cell']].to_numpy()).all())
checks['reference_groups_match_original_314'] = int(g[(g.variant == 'reference') & (g.seed == 42) & (g.arm == 'A')].shape[0]) == 314
# Independent recomputation of the dataset-level point estimates in summary.csv.
s = pd.read_csv(HERE / 'summary.csv')
pairs = {'Arm A per-cell minus Poisson calibration error': ('A', 'cal_cell', 'cal_poisson'),
         'Arm B per-cell minus Poisson calibration error': ('B', 'cal_cell', 'cal_poisson'),
         'Arm B per-cell minus shared calibration error': ('B', 'cal_cell', 'cal_shared'),
         'Arm B per-cell minus Poisson Welch recall': ('B', 'welch_cell', 'welch_poisson'),
         'Arm B per-cell minus shared Welch recall': ('B', 'welch_cell', 'welch_shared')}
worst = 0.0
for _, r in s.iterrows():
    if r.contrast not in pairs:
        continue
    arm, a, b = pairs[r.contrast]
    sub = g[(g.variant == r.variant) & (g.arm == arm)].dropna(subset=[a, b])
    per_group = (sub[a] - sub[b]).groupby([sub.dataset, sub.cell_type]).mean()
    est = per_group.groupby(level=0).mean().mean()
    worst = max(worst, abs(est - r.effect))
checks['summary_effects_recomputed'] = worst < 1e-12
checks['interval_contains_estimate'] = bool(((s.ci_low <= s.effect) & (s.effect <= s.ci_high)).all())
checks = {k: bool(v) for k, v in checks.items()}
status = 'PASS' if all(checks.values()) else 'FAIL'
out = {'status': status, 'checks': checks, 'fits': len(have), 'group_rows': len(g), 'max_recompute_error': float(worst)}
(HERE / 'verification.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
assert status == 'PASS'
