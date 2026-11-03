from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'design.json').read_text())


def sufficient_statistics(frame):
    cats = ['encoder', 'seed'] + (['arm'] if frame.arm.nunique() > 1 else [])
    columns = pd.get_dummies(frame[cats].astype(str), drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(frame)), frame.deficit.to_numpy(), columns.to_numpy()])
    y = frame.effect.to_numpy()
    stats = []
    for ds in sorted(frame.dataset.unique()):
        keep = (frame.dataset == ds).to_numpy()
        A, b = X[keep], y[keep]
        stats.append((len(b), A.T @ A, A.T @ b, b @ b, A.sum(0), b.sum()))
    arrays = tuple(np.array([row[i] for row in stats]) for i in range(6))
    return arrays, X, y


def fit_random_intercept(stats):
    n, xx, xy, yy, sx, sy = stats
    N, p = n.sum(), xy.shape[1]
    xx0, xy0, yy0 = xx.sum(0), xy.sum(0), yy.sum()

    def solve(rho, values=False):
        w = rho / (1 + n * rho)
        A = xx0 - np.einsum('i,ij,ik->jk', w, sx, sx)
        b = xy0 - np.einsum('i,ij,i->j', w, sx, sy)
        c = yy0 - np.sum(w * sy * sy)
        sign, ld = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf
        beta = np.linalg.solve(A, b)
        rss = max(float(c - b @ beta), np.finfo(float).tiny)
        objective = (N - p) * np.log(rss / (N - p)) + np.log1p(n * rho).sum() + ld
        if values:
            return beta, rss / (N - p), rho
        return objective

    opt = minimize_scalar(lambda t: solve(np.exp(t)), bounds=(-20, 20), method='bounded', options={'xatol': 1e-7})
    assert opt.success
    rho = float(np.exp(opt.x)) if opt.fun < solve(0) else 0.0
    return solve(rho, values=True)


def analyze_frame(frame, label, n_boot, seed):
    stats, X, y = sufficient_statistics(frame)
    beta, residual, rho = fit_random_intercept(stats)
    import statsmodels.api as sm
    with warnings.catch_warnings(record=True) as caught:
        reference = sm.MixedLM(y, X, groups=frame.dataset).fit(reml=True, method=['powell', 'lbfgs'], disp=False)
    discrepancy = float(np.max(np.abs(reference.fe_params - beta)))
    assert discrepancy < 1e-3, (label, discrepancy)
    rng = np.random.default_rng(seed)
    n_ds = len(stats[0])
    boot = []
    for i in range(n_boot):
        idx = rng.integers(n_ds, size=n_ds)
        bb, _, br = fit_random_intercept(tuple(a[idx] for a in stats))
        boot.append({'analysis': label, 'replicate': i, 'slope': float(bb[1]), 'variance_ratio': float(br)})
    ci = np.quantile([r['slope'] for r in boot], [.025, .975])
    row = {'analysis': label, 'n_rows': len(frame), 'n_datasets': n_ds,
           'n_groups': len(frame[['dataset', 'cell_type']].drop_duplicates()),
           'slope': float(beta[1]), 'ci_low': float(ci[0]), 'ci_high': float(ci[1]),
           'residual_variance': float(residual), 'dataset_variance': float(residual * rho),
           'reference_max_coefficient_error': discrepancy, 'n_bootstrap': n_boot,
           'direction': 'supports negative association' if ci[1] < 0 else ('positive association' if ci[0] > 0 else 'inconclusive'),
           'reference_converged': bool(reference.converged),
           'reference_warnings': ' | '.join(str(w.message) for w in caught)}
    return row, boot


def main():
    frame = pd.read_csv(HERE / 'groups.csv')
    wanted = {(a, e, s) for a in DESIGN['arms'] for e in DESIGN['encoders'] for s in DESIGN['seeds']}
    assert set(frame[['arm', 'encoder', 'seed']].itertuples(index=False, name=None)) == wanted
    rows, draws = [], []
    for i, (label, subset) in enumerate([('pooled', frame), ('Arm_A', frame[frame.arm == 'A']), ('Arm_B', frame[frame.arm == 'B'])]):
        row, boot = analyze_frame(subset, label, DESIGN['inference']['resamples'], DESIGN['inference']['seed'] + i)
        rows.append(row); draws.extend(boot)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(draws).to_csv(HERE / 'bootstrap.csv', index=False)
    frame.groupby(['arm', 'encoder', 'seed']).agg(n_groups=('effect', 'size'),
        median_validation_deficit=('deficit', 'median'), mean_effect=('effect', 'mean'),
        median_effect=('effect', 'median')).to_csv(HERE / 'seed-summary.csv')
    ledger = {'design_sha256': (HERE / 'design.sha256').read_text().strip(),
              'new_replication': True, 'source_table': 'groups.csv', 'inference': rows,
              'fit_seeds': DESIGN['seeds'], 'fit_count': len(wanted),
              'primary_slope_interpretation': 'association of independently measured validation deficit with test calibration effect; not biological validation or a sufficient rule'}
    (HERE / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')


if __name__ == '__main__':
    main()
