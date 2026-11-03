from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'design.json').read_text())


def paired_summary(deltas, n_boot, seed):
    x = np.asarray(deltas, dtype=float)
    assert len(x) > 1 and np.isfinite(x).all()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(x), size=(n_boot, len(x)))
    boot = x[sampled].mean(1)
    null = (rng.choice([-1, 1], size=(n_boot, len(x))) * x).mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    p = (1 + np.sum(np.abs(null) >= abs(x.mean()) - 1e-15)) / (n_boot + 1)
    return {'effect': float(x.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'signflip_p_two_sided': float(p), 'n_datasets': len(x)}, boot


def main():
    groups = pd.read_csv(HERE / 'groups.csv')
    classification = pd.read_csv(HERE / 'dataset-classification.csv')
    keys = ['arm', 'seed', 'variant', 'dataset']
    markers = groups.groupby(keys)[['welch', 'auroc', 'calibration']].mean().reset_index()
    dataset = markers.merge(classification, on=keys, how='outer', validate='one_to_one')
    dataset.to_csv(HERE / 'dataset-endpoints.csv', index=False)
    metrics = ['welch', 'auroc', 'calibration', 'macro_f1', 'rare_f1_full', 'rare_f1_masked']
    contrasts = [('shared_per_gene', 'poisson'), ('per_cell_type', 'poisson'), ('per_cell', 'poisson'),
                 ('per_cell', 'shared_per_gene'), ('per_cell_type', 'shared_per_gene')]
    summary, seed_rows, bootstrap, paired_rows = [], [], [], []
    counter = 0
    for arm in DESIGN['arms']:
        for a, b in contrasts:
            wide_a = dataset[(dataset.arm == arm) & (dataset.variant == a)]
            wide_b = dataset[(dataset.arm == arm) & (dataset.variant == b)]
            joined = wide_a.merge(wide_b, on=['arm', 'seed', 'dataset'], suffixes=('_a', '_b'), validate='one_to_one')
            for metric in metrics:
                valid = joined[[f'{metric}_a', f'{metric}_b']].notna().all(axis=1)
                pair = joined[valid].copy()
                pair['delta'] = pair[f'{metric}_a'] - pair[f'{metric}_b']
                for _, r in pair.iterrows():
                    paired_rows.append({'arm': arm, 'seed': int(r.seed), 'dataset': r.dataset,
                        'metric': metric, 'a': a, 'b': b, 'delta': float(r.delta)})
                mean_seed = pair.groupby('dataset').delta.agg(['mean', 'size'])
                complete = mean_seed[mean_seed['size'] == len(DESIGN['seeds'])]['mean']
                if len(complete) < 2:
                    raise ValueError(f'insufficient complete datasets for {arm} {a} {b} {metric}')
                result, boot = paired_summary(complete, DESIGN['inference']['resamples'], DESIGN['inference']['seed'] + counter)
                counter += 1
                entry = {'arm': arm, 'metric': metric, 'a': a, 'b': b, **result,
                         'n_datasets_excluded_incomplete_seeds': int(len(mean_seed) - len(complete)),
                         'n_seeds': len(DESIGN['seeds']), 'lower_is_better': metric == 'calibration'}
                summary.append(entry)
                for i, val in enumerate(boot):
                    bootstrap.append({'arm': arm, 'metric': metric, 'a': a, 'b': b, 'replicate': i, 'effect': val})
                for seed, part in pair.groupby('seed'):
                    sresult, _ = paired_summary(part.delta, DESIGN['inference']['resamples'], DESIGN['inference']['seed'] + counter)
                    counter += 1
                    seed_rows.append({'arm': arm, 'seed': seed, 'metric': metric, 'a': a, 'b': b, **sresult})
    pd.DataFrame(summary).to_csv(HERE / 'summary.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(HERE / 'seed-summary.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(HERE / 'paired-dataset-effects.csv', index=False)
    pd.DataFrame(bootstrap).to_csv(HERE / 'bootstrap.csv', index=False)
    report = pd.DataFrame(summary)
    print(report[(report.a == 'per_cell') & (report.b == 'poisson')].to_string(index=False), flush=True)
    ledger = {'design_sha256': (HERE / 'design.sha256').read_text().strip(), 'fit_seeds': DESIGN['seeds'],
              'source': 'new E2 PCA fits, paired draws at fixed means', 'contrasts': summary,
              'inference_scope': 'nominal endpoint-specific intervals; no claim of multiplicity-adjusted significance',
              'limitations': ['one encoder for matched biological comparison', 'cell-type dispersion uses labels and validation medians',
                              'both arms evaluated on the same held-out target gene panel', 'TSTR describes pooled training and within-dataset evaluation, not external-dataset generalisation']}
    paired = pd.DataFrame(paired_rows)
    interaction = []
    for metric in metrics:
        part = paired[(paired.a == 'per_cell') & (paired.b == 'poisson') & (paired.metric == metric)]
        means = part.groupby(['arm', 'dataset']).delta.agg(['mean', 'size']).reset_index()
        means = means[means['size'] == len(DESIGN['seeds'])]
        wide = means.pivot(index='dataset', columns='arm', values='mean').dropna()
        result, _ = paired_summary(wide['B'] - wide['A'], DESIGN['inference']['resamples'], DESIGN['inference']['seed'] + counter)
        counter += 1
        interaction.append({'metric': metric, 'contrast': 'Arm B minus Arm A, per-cell minus Poisson', **result})
    pd.DataFrame(interaction).to_csv(HERE / 'arm-interaction.csv', index=False)
    ledger['arm_interaction'] = interaction
    (HERE / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')


def marker_dose_response():
    import importlib.util
    path = HERE.parent / 'e2-dose-response/analyze.py'
    spec = importlib.util.spec_from_file_location('e2_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    group = pd.read_csv(HERE / 'groups.csv')
    e2 = pd.read_csv(HERE / 'validation-deficits.csv')
    keys = ['arm', 'seed', 'dataset', 'cell_type']
    p = group[group.variant == 'poisson']
    n = group[group.variant == 'per_cell']
    pairs = n.merge(p, on=keys, suffixes=('_nb', '_poisson'), validate='one_to_one')
    pairs = pairs.merge(e2[keys + ['deficit', 'encoder']], on=keys, validate='one_to_one')
    rows, all_boot = [], []
    for j, metric in enumerate(['welch', 'auroc']):
        frame = pairs.copy()
        frame['effect'] = frame[f'{metric}_nb'] - frame[f'{metric}_poisson']
        frame = frame.dropna(subset=['effect', 'deficit'])
        frame.to_csv(HERE / f'dose-response-{metric}-groups.csv', index=False)
        result, boot = module.analyze_frame(frame, metric, 2000, 20261108 + j)
        result['direction'] = 'positive association' if result['ci_low'] > 0 else ('negative association' if result['ci_high'] < 0 else 'inconclusive')
        rows.append(result); all_boot.extend(boot)
    pd.DataFrame(rows).to_csv(HERE / 'dose-response-marker-summary.csv', index=False)
    pd.DataFrame(all_boot).to_csv(HERE / 'dose-response-marker-bootstrap.csv', index=False)
    (HERE / 'dose-response-marker-ledger.json').write_text(json.dumps({'source': 'E1 validation deficit on the common target genes joined to E1 test marker effects',
        'seed': 20261108, 'inference': rows, 'primary': False}, indent=2))


if __name__ == '__main__':
    main()
    marker_dose_response()
