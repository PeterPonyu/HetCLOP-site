"""Structural verifiers for the follow-up experiments (selection, label transfer, gene dependence, accession
clustering, scVI sensitivity, training size, ten-seed end-to-end). Each writes <experiment>/verification.json with a
PASS/FAIL status and the checks it ran; verify_current_manuscript.py reads those records.

The checks recompute what can be recomputed cheaply from saved outputs (choices, point estimates, completeness,
evaluation support); bootstrap intervals are reproduced separately by rerunning each analysis script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
EXP = REV / 'experiments'
ROOT = REV.parents[1]


def design_ok(d):
    return hashlib.sha256((d / 'design.json').read_bytes()).hexdigest() == (d / 'design.sha256').read_text().strip()


def write(d, checks):
    status = 'PASS' if all(c['pass'] for c in checks.values()) else 'FAIL'
    (d / 'verification.json').write_text(json.dumps({'status': status, 'checks': checks}, indent=2, default=str) + '\n')
    print(f'{d.name}: {status}', {k: v['pass'] for k, v in checks.items() if not v['pass']} or '')
    return status


def check(ok, detail=''):
    return {'pass': bool(ok), 'detail': detail}


def scvi():
    d = EXP / 'e4-scvi-sensitivity'
    design = json.loads((d / 'design.json').read_text())
    variants = ['reference', 'uniform_s1', 'uniform_s2', 'uniform_s3', 'type_floor', 'batch_balanced', 'cap10000', 'mad_qc']
    want = {f'{v}_{m}_{s}' for v in variants for m in ['gene', 'gene-cell'] for s in design['fit_seeds']}
    have = {p.parent.name for p in (d / 'runs').glob('*/complete.json')}
    groups = {r: pd.read_csv(d / 'runs' / r / 'groups.csv') for r in sorted(have)}
    empty = [r for r, g in groups.items() if g.empty or g[['cal_nb', 'cal_poisson']].isna().any().any()]
    # Same evaluation groups for both modes and all seeds of a corpus.
    support = []
    for v in variants:
        keys = {frozenset(map(tuple, groups[r][['dataset', 'cell_type']].values)) for r in groups if r.startswith(v + '_')}
        if len(keys) != 1:
            support.append(v)
    s = pd.read_csv(d / 'summary.csv')
    return write(d, {'design-hash': check(design_ok(d)),
                     'all-fits-present': check(want == have, f'{len(have)}/{len(want)}'),
                     'group-tables-complete': check(not empty, empty),
                     'identical-groups-within-corpus': check(not support, support),
                     'summary-finite': check(len(s) == 16 and np.isfinite(s[['effect', 'ci_low', 'ci_high']]).all().all())})


def training_size():
    d = EXP / 'e4b-training-size'
    design = json.loads((d / 'design.json').read_text())
    meta = np.load(ROOT / 'experiments/phase2a_decoder_2026-07-08/cache/meta.npz', allow_pickle=True)
    ex = np.load(d / 'extra_meta.npz', allow_pickle=True)
    n1 = int((meta['split'] == 0).sum())
    sizes = {'x2': n1 + int(ex['half'].sum()), 'x3': n1 + len(ex['half'])}
    want = {f'{k}_{a}_{s}' for k in sizes for a in 'AB' for s in design['fit_seeds']}
    have = {p.parent.name for p in (d / 'runs').glob('*/complete.json')}
    bad_n = [r for r in have if json.loads((d / 'runs' / r / 'fit.json').read_text())['n_train'] != sizes[r.split('_')[0]]]
    # Evaluation support: every enlarged fit scores exactly the reference corpus's groups of the same arm.
    ref = {}
    for p in (EXP / 'e4-sampling-sensitivity' / 'runs').glob('reference_*/groups.csv'):
        g = pd.read_csv(p)
        for arm, sub in g.groupby('arm'):
            ref.setdefault(arm, set()).update(map(tuple, sub[['dataset', 'cell_type']].values))
    diff = []
    for r in sorted(have):
        g = pd.read_csv(d / 'runs' / r / 'groups.csv')
        if set(map(tuple, g[['dataset', 'cell_type']].values)) != ref[r.split('_')[1]]:
            diff.append(r)
    extra_ds = set(ex['dataset_ids'].astype(str))
    return write(d, {'design-hash': check(design_ok(d)),
                     'all-fits-present': check(want == have, f'{len(have)}/{len(want)}'),
                     'training-sizes': check(not bad_n, f"x1 {n1}, x2 {sizes['x2']}, x3 {sizes['x3']}"),
                     'identical-evaluation-groups': check(not diff, diff),
                     'extra-cells-from-known-datasets': check(extra_ds <= set(meta['dataset_ids'].astype(str))),
                     'half-is-half': check(abs(int(ex['half'].sum()) - len(ex['half']) / 2) <= len(extra_ds))})


def ten_seeds():
    d = EXP / 'e5-endtoend-seeds'
    design = json.loads((d / 'design.json').read_text())
    archive = ROOT / 'experiments/phase2v_e2e_2026-07-19/results'
    per = pd.read_csv(d / 'per-seed.csv')
    mism = []
    for s in design['seeds']['archived'] + design['seeds']['new']:
        f = (archive if s in design['seeds']['archived'] else d / 'results') / f'e2e_generation_seed{s}.json'
        j = json.loads(f.read_text())
        for arm, name in {'e2e_poisson': 'poisson', 'e2e_nb_gene': 'shared', 'e2e_nb_gene-cell': 'per_cell'}.items():
            row = per[(per.seed == s) & (per.arm == name)].iloc[0]
            if abs(row.marker_recall_mean - j['arms'][arm]['marker_recall_mean']) > 1e-12:
                mism.append((s, name))
    return write(d, {'design-hash': check(design_ok(d)),
                     'ten-seeds-three-arms': check(len(per) == 30 and per.seed.nunique() == 10),
                     'archived-seeds-not-refitted': check(not any((d / 'results' / f'e2e_generation_seed{s}.json').exists()
                                                                for s in design['seeds']['archived'])),
                     'per-seed-table-matches-results': check(not mism, mism)})


def selection():
    d = EXP / 'e3-prospective-selection'
    g = pd.read_csv(d / 'groups.csv')
    c = pd.read_csv(d / 'choices.csv')
    opt = ['poisson', 'shared', 'per_type', 'per_cell']
    val = g[g.split == 'validation'].groupby(['arm', 'encoder', 'seed', 'dataset'])[opt].median()
    again = val.idxmin(axis=1)  # first minimum, i.e. the stated tie order
    merged = c.set_index(['arm', 'encoder', 'seed', 'dataset']).join(again.rename('again'))
    return write(d, {'design-hash': check(design_ok(d)),
                     'choices-from-validation-medians': check((merged.choice == merged.again).all()),
                     'both-splits-present': check(set(g.split) == {'validation', 'test'}),
                     'fits': check(len(g.groupby(['arm', 'encoder', 'seed'])) == 24)})


def label_transfer():
    d = EXP / 'e4c-label-transfer'
    ds = pd.read_csv(d / 'datasets.csv')
    s = json.loads((d / 'summary.json').read_text())
    ok = all(abs(ds[k].mean() - s[k]['mean']) < 1e-12 for k in ['coverage', 'accuracy_assigned', 'accuracy_overall', 'macro_f1'])
    t = pd.read_csv(d / 'types.csv')
    rr = t[t.rare].groupby('dataset').recall.mean().mean()
    return write(d, {'design-hash': check(design_ok(d)),
                     'fifty-datasets': check(len(ds) == 50),
                     'summary-means-recomputed': check(ok),
                     'rare-recall-recomputed': check(abs(rr - s['rare_type_recall']['mean_over_datasets']) < 1e-12)})


def dependence():
    d = EXP / 'e6-gene-dependence'
    g = pd.read_csv(d / 'groups.csv')
    s = pd.read_csv(d / 'summary.csv')
    bad = []
    for _, r in s[s.metric.isin(['corr', 'cal'])].iterrows():
        sub = g[g.arm == r.arm].groupby(['dataset', 'cell_type']).mean(numeric_only=True)
        eff = (sub[f'{r.metric}_{r.a}'] - sub[f'{r.metric}_{r.b}']).dropna().groupby(level=0).mean().mean()
        if abs(eff - r.effect) > 1e-12:
            bad.append((r.arm, r.metric, r.a, r.b))
    return write(d, {'design-hash': check(design_ok(d)),
                     'three-seeds-both-arms': check(set(g.seed) == {42, 43, 44} and set(g.arm) == {'A', 'B'}),
                     'point-estimates-recomputed': check(not bad, bad)})


def accession():
    d = EXP / 'e7-accession-clustering'
    a = pd.read_csv(d / 'agreement.csv')
    r = pd.read_csv(d / 'replay-e4.csv')
    s = json.loads((d / 'summary.json').read_text())
    return write(d, {'design-hash': check(design_ok(d)),
                     'all-comparisons-agree': check(bool(a.agree.all()) and len(a) == s['comparisons']),
                     'replay-matches': check(bool(r[['effect_match', 'ci_within_tolerance', 'classification_match']].all().all())
                                             and len(r) == s['replay_rows'])})


if __name__ == '__main__':
    results = [f() for f in [scvi, training_size, ten_seeds, selection, label_transfer, dependence, accession]]
    raise SystemExit(0 if all(x == 'PASS' for x in results) else 1)
