"""E7: accession-level clustering sensitivity and an independent replay of the E4 summaries.

Every analysis keeps its original dataset-level aggregation; only the resampling or permutation unit changes
from dataset to GEO accession. Outputs: accession-summary.csv, replay-e4.csv, summary.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
MS = HERE.parents[1] / 'manuscript'
SEED = 20261112
sys.path.insert(0, str(ROOT / 'experiments' / 'phase2f_f4_resolution_2026-07-09'))
from clustered_inference import clustered_dataset_medians, dataset_signflip_permutation  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E2 = load('e2analyze', EXP / 'e2-dose-response' / 'analyze.py')
ACC = pd.read_csv(MS / 'figs' / 'dataset_table.csv').set_index('dataset_id').accession.to_dict()


def classify(lo, hi):
    return 'below zero' if hi < 0 else ('above zero' if lo > 0 else 'spans zero')


def boot_mean(values, n_boot, rng):
    x = np.asarray(values, dtype=float)
    b = x[rng.integers(len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(b, [.025, .975])
    return float(x.mean()), float(lo), float(hi)


def nested(frame, value, unit):
    """Seed-average within group, groups within dataset, then datasets within accession when unit='accession'."""
    g = frame.groupby(['dataset', 'cell_type'])[value].mean()
    d = g.groupby(level=0).mean()
    return d if unit == 'dataset' else d.groupby(d.index.map(ACC)).mean()


def row(analysis, contrast, unit, effect, lo, hi, n, extra=None):
    r = {'analysis': analysis, 'contrast': contrast, 'unit': unit, 'effect': effect, 'ci_low': lo, 'ci_high': hi,
         'classification': classify(lo, hi) if np.isfinite(lo) else '', 'n_clusters': n}
    r.update(extra or {})
    return r


def armb_energy():
    p = json.loads((ROOT / 'experiments/phase2g_armb_f4_2026-07-09/results/phase2g.json').read_text())
    out = []
    by = {}
    for c in p['E7_clustered_inference']['per_comparison']:
        if c['metric'] != 'energy_distance':
            continue
        recs = c['per_group']
        dres = clustered_dataset_medians(recs)
        assert abs(dres['wilcoxon_p'] - c['clustered_50dataset']['wilcoxon_p']) < 1e-12, 'dataset-level p not reproduced'
        arecs = [{**r, 'dataset': ACC[r['dataset']]} for r in recs]
        ares = clustered_dataset_medians(arecs)
        aperm = dataset_signflip_permutation(arecs, n_perm=10000, seed=0)
        by.setdefault(c['encoder'], []).append((dres, ares, aperm, c['permutation_dataset_signflip']['p']))
    for enc, items in by.items():
        out.append(row('armb_energy', f'Arm B per-cell minus Poisson energy, {enc}', 'dataset', np.nan, np.nan, np.nan, 50,
                       {'wilcoxon_p_worst': max(i[0]['wilcoxon_p'] for i in items),
                        'sign_p_worst': max(i[0]['sign_test_p'] for i in items),
                        'signflip_p_worst': max(i[3] for i in items)}))
        out.append(row('armb_energy', f'Arm B per-cell minus Poisson energy, {enc}', 'accession', np.nan, np.nan, np.nan,
                       items[0][1]['n_datasets'],
                       {'wilcoxon_p_worst': max(i[1]['wilcoxon_p'] for i in items),
                        'sign_p_worst': max(i[1]['sign_test_p'] for i in items),
                        'signflip_p_worst': max(i[2]['p_value'] for i in items)}))
    return out


def e1_matched():
    pe = pd.read_csv(EXP / 'e1-armb-biology' / 'paired-dataset-effects.csv')
    ref = pd.read_csv(EXP / 'e1-armb-biology' / 'summary.csv')
    rng = np.random.default_rng(SEED)
    out = []
    for (arm, metric, a, b), part in pe.groupby(['arm', 'metric', 'a', 'b'], sort=True):
        ms = part.groupby('dataset').delta.agg(['mean', 'size'])
        d = ms[ms['size'] == part.seed.nunique()]['mean']
        acc = d.groupby(d.index.map(ACC)).mean()
        r0 = ref[(ref.arm == arm) & (ref.metric == metric) & (ref.a == a) & (ref.b == b)].iloc[0]
        assert abs(r0.effect - d.mean()) < 1e-12
        e, lo, hi = boot_mean(acc, 10000, rng)
        null = (rng.choice([-1, 1], size=(10000, len(acc))) * acc.to_numpy()).mean(1)
        pval = (1 + np.sum(np.abs(null) >= abs(acc.mean()) - 1e-15)) / 10001
        label = f'Arm {arm} {a} minus {b}: {metric}'
        out.append(row('e1_matched', label, 'dataset', float(r0.effect), float(r0.ci_low), float(r0.ci_high), int(r0.n_datasets),
                       {'signflip_p': float(r0.signflip_p_two_sided)}))
        out.append(row('e1_matched', label, 'accession', e, lo, hi, len(acc), {'signflip_p': float(pval)}))
    return out


def slope_rows(analysis, frame, labels, n_boot, ref_rows):
    out = []
    for i, (label, sub) in enumerate(labels(frame)):
        r0 = ref_rows[label]
        out.append(row(analysis, f'deficit-effect slope, {label}', 'dataset', r0['slope'], r0['ci_low'], r0['ci_high'], r0['n_datasets']))
        acc = sub.assign(dataset=sub.dataset.map(ACC))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r, _ = E2.analyze_frame(acc, label, n_boot, SEED + i)
        out.append(row(analysis, f'deficit-effect slope, {label}', 'accession', r['slope'], r['ci_low'], r['ci_high'], r['n_datasets']))
    return out


def e2_slope():
    frame = pd.read_csv(EXP / 'e2-dose-response' / 'groups.csv')
    ref = pd.read_csv(EXP / 'e2-dose-response' / 'summary.csv').set_index('analysis').to_dict('index')
    ref = {k: {**v, 'n_datasets': int(v['n_datasets'])} for k, v in ref.items()}
    return slope_rows('e2_slope', frame,
                      lambda f: [('pooled', f), ('Arm_A', f[f.arm == 'A']), ('Arm_B', f[f.arm == 'B'])], 2000, ref)


E4_CONTRASTS = [
    ('A', 'cal_cell', 'cal_poisson', 'Arm A per-cell minus Poisson calibration error'),
    ('B', 'cal_cell', 'cal_poisson', 'Arm B per-cell minus Poisson calibration error'),
    ('B', 'cal_cell', 'cal_shared', 'Arm B per-cell minus shared calibration error'),
    ('B', 'welch_cell', 'welch_poisson', 'Arm B per-cell minus Poisson Welch recall'),
    ('B', 'welch_cell', 'welch_shared', 'Arm B per-cell minus shared Welch recall'),
]


def e4_frame():
    return pd.concat([pd.read_csv(p) for p in sorted((EXP / 'e4-sampling-sensitivity' / 'runs').glob('*/groups.csv'))], ignore_index=True)


def e4_reference():
    frame = e4_frame()
    s = pd.read_csv(EXP / 'e4-sampling-sensitivity' / 'summary.csv')
    s = s[s.variant == 'reference'].set_index('contrast')
    rng = np.random.default_rng(SEED)
    out = []
    for arm, a, b, label in E4_CONTRASTS:
        sub = frame[(frame.variant == 'reference') & (frame.arm == arm)].dropna(subset=[a, b])
        sub = sub.assign(d=sub[a] - sub[b])
        acc = nested(sub, 'd', 'accession')
        r0 = s.loc[label]
        out.append(row('e4_reference', label, 'dataset', float(r0.effect), float(r0.ci_low), float(r0.ci_high), int(r0.n_datasets)))
        out.append(row('e4_reference', label, 'accession', *boot_mean(acc, 2000, rng), len(acc)))
    sub = frame[frame.variant == 'reference'].dropna(subset=['deficit']).copy()
    sub['effect'] = sub.cal_cell - sub.cal_poisson
    sub['encoder'] = 'pca_512'
    r0 = s.loc['Deficit-effect slope (pooled arms)']
    ref = {'pooled': {'slope': r0.effect, 'ci_low': r0.ci_low, 'ci_high': r0.ci_high, 'n_datasets': int(r0.n_datasets)}}
    out += slope_rows('e4_reference', sub, lambda f: [('pooled', f)], 1000, ref)
    return out


def e4_scvi_reference():
    runs = sorted((EXP / 'e4-scvi-sensitivity' / 'runs').glob('reference_*/complete.json'))
    if len(runs) < 6:
        return [row('e4_scvi_reference', 'not available', 'dataset', np.nan, np.nan, np.nan, 0)]
    frame = pd.concat([pd.read_csv(p.parent / 'groups.csv') for p in runs], ignore_index=True)
    rng = np.random.default_rng(SEED)
    out = []
    for mode, name in [('gene', 'shared per-gene'), ('gene-cell', 'per-cell')]:
        sub = frame[frame['mode'] == mode].assign(d=lambda x: x.cal_nb - x.cal_poisson)
        for unit in ['dataset', 'accession']:
            v = nested(sub, 'd', unit)
            out.append(row('e4_scvi_reference', f'scVI {name} NB minus Poisson calibration error', unit, *boot_mean(v, 2000, rng), len(v)))
    return out


def e3_selection():
    c = pd.read_csv(EXP / 'e3-prospective-selection' / 'choices.csv')
    rng = np.random.default_rng(SEED)
    out = []
    for arm in ['A', 'B']:
        for col, name in [('regret', 'regret relative to the best option on test'), ('vs_poisson', 'selected minus always-Poisson'),
                          ('vs_per_cell', 'selected minus always-per-cell')]:
            d = c[c.arm == arm].groupby('dataset')[col].mean()
            for unit, v in [('dataset', d), ('accession', d.groupby(d.index.map(ACC)).mean())]:
                out.append(row('e3_selection', f'Arm {arm} {name}', unit, *boot_mean(v, 2000, rng), len(v)))
    return out


def e6_dependence():
    g = pd.read_csv(EXP / 'e6-gene-dependence' / 'groups.csv')
    rng = np.random.default_rng(SEED)
    out = []
    for arm in ['A', 'B']:
        s = g[g.arm == arm]
        for metric, a, b in [('corr', 'cell', 'poisson'), ('corr', 'shared', 'poisson'), ('corr', 'cell', 'shared'),
                             ('cal', 'cell', 'poisson'), ('cal', 'shared', 'poisson')]:
            sub = s.assign(d=s[f'{metric}_{a}'] - s[f'{metric}_{b}'])
            name = 'correlation error' if metric == 'corr' else 'calibration error'
            for unit in ['dataset', 'accession']:
                v = nested(sub, 'd', unit)
                out.append(row('e6_dependence', f'Arm {arm} {a} minus {b} {name}', unit, *boot_mean(v, 2000, rng), len(v)))
    return out


def replay_slope(args):
    """statsmodels REML random-intercept slope and a 1000-resample dataset bootstrap of that model."""
    import statsmodels.api as sm
    variant, sub = args
    sub = sub.reset_index(drop=True)
    dummies = pd.get_dummies(sub[['seed', 'arm']].astype(str), drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(sub)), sub.deficit.to_numpy(), dummies.to_numpy()])
    y = sub.effect.to_numpy()

    def fit(Xm, ym, groups):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return sm.MixedLM(ym, Xm, groups=groups).fit(reml=True, method=['powell', 'lbfgs'], disp=False).fe_params[1]

    point = fit(X, y, sub.dataset.to_numpy())
    rng = np.random.default_rng(SEED + 7)
    ds = sub.dataset.unique()
    where = {d: np.flatnonzero(sub.dataset.to_numpy() == d) for d in ds}
    boots = []
    for _ in range(1000):
        pick = rng.choice(ds, size=len(ds), replace=True)
        idx = np.concatenate([where[d] for d in pick])
        grp = np.concatenate([np.full(len(where[d]), k) for k, d in enumerate(pick)])
        boots.append(fit(X[idx], y[idx], grp))
    lo, hi = np.quantile(boots, [.025, .975])
    return variant, float(point), float(lo), float(hi)


def replay_e4():
    frame = e4_frame()
    s = pd.read_csv(EXP / 'e4-sampling-sensitivity' / 'summary.csv')
    rng = np.random.default_rng(SEED + 1)
    rows, jobs = [], []
    for v in s.variant.unique():
        for arm, a, b, label in E4_CONTRASTS:
            sub = frame[(frame.variant == v) & (frame.arm == arm)].dropna(subset=[a, b])
            # Independent path: explicit per-group seed means, then dataset means, then index-based bootstrap.
            per_group = {}
            for (d, t), part in sub.groupby(['dataset', 'cell_type']):
                per_group.setdefault(d, []).append(float(np.mean(part[a].to_numpy() - part[b].to_numpy())))
            per_ds = np.array([np.mean(per_group[d]) for d in sorted(per_group)])
            e, lo, hi = boot_mean(per_ds, 2000, rng)
            r0 = s[(s.variant == v) & (s.contrast == label)].iloc[0]
            tol = 0.25 * (r0.ci_high - r0.ci_low)
            rows.append({'variant': v, 'contrast': label, 'effect_summary': r0.effect, 'effect_replay': e,
                         'effect_match': abs(e - r0.effect) < 1e-9, 'ci_summary': f'[{r0.ci_low:.4f}, {r0.ci_high:.4f}]',
                         'ci_replay': f'[{lo:.4f}, {hi:.4f}]',
                         'ci_within_tolerance': abs(lo - r0.ci_low) <= tol and abs(hi - r0.ci_high) <= tol,
                         'classification_match': classify(lo, hi) == r0.classification})
        sub = frame[frame.variant == v].dropna(subset=['deficit']).copy()
        sub['effect'] = sub.cal_cell - sub.cal_poisson
        jobs.append((v, sub))
    with Pool(8) as pool:
        slopes = pool.map(replay_slope, jobs)
    for v, point, lo, hi in slopes:
        r0 = s[(s.variant == v) & (s.contrast == 'Deficit-effect slope (pooled arms)')].iloc[0]
        tol = 0.25 * (r0.ci_high - r0.ci_low)
        rows.append({'variant': v, 'contrast': 'Deficit-effect slope (pooled arms)', 'effect_summary': r0.effect,
                     'effect_replay': point, 'effect_match': abs(point - r0.effect) < 1e-3,
                     'ci_summary': f'[{r0.ci_low:.4f}, {r0.ci_high:.4f}]', 'ci_replay': f'[{lo:.4f}, {hi:.4f}]',
                     'ci_within_tolerance': abs(lo - r0.ci_low) <= tol and abs(hi - r0.ci_high) <= tol,
                     'classification_match': classify(lo, hi) == r0.classification})
    return pd.DataFrame(rows)


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    assert len(set(ACC.values())) == 45
    rows = []
    for f in [armb_energy, e1_matched, e2_slope, e4_reference, e4_scvi_reference, e3_selection, e6_dependence]:
        rows += f()
        print(f.__name__, 'done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / 'accession-summary.csv', index=False)
    # Agreement between dataset and accession clustering for each analysis row pair.
    agree = []
    for (an, ct), part in out.groupby(['analysis', 'contrast'], sort=False):
        if set(part.unit) != {'dataset', 'accession'}:
            continue
        d, a = part[part.unit == 'dataset'].iloc[0], part[part.unit == 'accession'].iloc[0]
        if an == 'armb_energy':
            same = (d.wilcoxon_p_worst < .05) == (a.wilcoxon_p_worst < .05) and (d.signflip_p_worst < .05) == (a.signflip_p_worst < .05)
        else:
            same = d.classification == a.classification
        agree.append({'analysis': an, 'contrast': ct, 'agree': bool(same)})
    agree = pd.DataFrame(agree)
    agree.to_csv(HERE / 'agreement.csv', index=False)
    replay = replay_e4()
    replay.to_csv(HERE / 'replay-e4.csv', index=False)
    summary = {'design_sha256': (HERE / 'design.sha256').read_text().strip(), 'n_accessions': 45,
               'comparisons': int(len(agree)), 'agreeing': int(agree.agree.sum()),
               'disagreements': agree[~agree.agree].to_dict('records'),
               'replay_rows': int(len(replay)), 'replay_effects_match': int(replay.effect_match.sum()),
               'replay_ci_within_tolerance': int(replay.ci_within_tolerance.sum()),
               'replay_classification_match': int(replay.classification_match.sum())}
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
