from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def main():
    design = json.loads((HERE / 'design.json').read_text())
    dh = (HERE / 'design.sha256').read_text().strip()
    assert digest(HERE / 'design.json') == dh
    for rel, h in json.loads((HERE / 'source-manifest.json').read_text()).items():
        assert digest(ROOT / rel) == h, rel
    n = 0
    for arm in design['arms']:
        for seed in design['seeds']:
            out = HERE / 'runs' / f'{arm}_{seed}'
            record = json.loads((out / 'complete.json').read_text())
            assert record['design_sha256'] == dh
            for name, h in record['files'].items():
                assert digest(out / name) == h, str(out / name)
            split = np.load(out / 'evaluation-split.npz')
            assert not set(split['train']) & set(split['evaluation'])
            fit = json.loads((out / 'fit-log.json').read_text())
            assert fit['all_variants_share_mu'] and fit['n_genes'] == 1250
            pred = pd.read_csv(out / 'predictions.csv')
            scores = pd.read_csv(out / 'dataset-classification.csv')
            assert set(pred.variant) == set(design['variants'] + ['real_reference'])
            for (variant, dataset), part in pred.groupby(['variant', 'dataset']):
                expected = scores[(scores.variant == variant) & (scores.dataset == dataset)].iloc[0]
                value = f1_score(part.truth, part.prediction, labels=np.unique(part.truth), average='macro', zero_division=0)
                assert abs(value - expected.macro_f1) < 1e-12
                rare = sorted(set(fit['rare_types']) & set(part.truth))
                if rare:
                    value = f1_score(part.truth, part.prediction, labels=rare, average='macro', zero_division=0)
                    assert abs(value - expected.rare_f1_full) < 1e-12
            n += 1
    summary = pd.read_csv(HERE / 'summary.csv')
    paired = pd.read_csv(HERE / 'paired-dataset-effects.csv')
    bootstrap = pd.read_csv(HERE / 'bootstrap.csv')
    for _, row in summary.iterrows():
        sel = (paired.arm == row.arm) & (paired.metric == row.metric) & (paired.a == row.a) & (paired.b == row.b)
        sub = paired[sel].groupby('dataset').delta.agg(['mean', 'size'])
        sub = sub[sub['size'] == 3]
        assert len(sub) == row.n_datasets and abs(sub['mean'].mean() - row.effect) < 1e-12
        b = bootstrap[(bootstrap.arm == row.arm) & (bootstrap.metric == row.metric) & (bootstrap.a == row.a) & (bootstrap.b == row.b)]
        assert len(b) == design['inference']['resamples']
        assert np.allclose(np.quantile(b.effect, [.025, .975]), [row.ci_low, row.ci_high])
    assert len(summary) == 60
    seed = pd.read_csv(HERE / 'seed-summary.csv')
    assert len(seed) == 180 and set(seed.seed) == set(design['seeds'])
    marker = pd.read_csv(HERE / 'dose-response-marker-summary.csv')
    assert len(marker) == 2 and (marker.reference_max_coefficient_error < .001).all()
    result = {'status': 'PASS', 'matched_mean_fits': n, 'variants': 4, 'seed_contrasts': len(seed),
              'dataset_clustered_contrasts': len(summary), 'marker_dose_response_models': len(marker)}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
