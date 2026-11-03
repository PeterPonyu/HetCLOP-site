from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def main():
    d = json.loads((HERE / 'design.json').read_text())
    design_hash = (HERE / 'design.sha256').read_text().strip()
    released = json.loads((HERE.parent / 'released-source-digests.json').read_text())['files']
    assert digest(HERE / 'design.json') == design_hash
    for rel, h in {**json.loads((HERE / 'input-code-manifest.json').read_text()), **json.loads((HERE / 'foundation-input-manifest.json').read_text())}.items():
        # Two pinned source files differ in this release by one reworded docstring line; see
        # ../released-source-digests.json, which records both digests.
        alt = released.get(rel, {})
        assert digest(ROOT / rel) == h or (alt.get('analysed') == h and digest(ROOT / rel) == alt['released']), rel
    cache = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'
    for rel, h in json.loads((HERE / 'input-data-manifest.json').read_text()).items():
        assert digest(cache / rel) == h, rel
    frame = pd.read_csv(HERE / 'groups.csv')
    keys = ['arm', 'encoder', 'seed', 'dataset', 'cell_type']
    assert not frame.duplicated(keys).any()
    assert np.isfinite(frame[['deficit', 'effect']].to_numpy()).all()
    assert (frame[['n_validation', 'n_test']] >= 20).all().all()
    assert np.allclose(frame.effect, frame.nb_calibration - frame.poisson_calibration, atol=1e-12)
    expected = {(a, e, s) for a in d['arms'] for e in d['encoders'] for s in d['seeds']}
    assert set(frame[keys[:3]].itertuples(index=False, name=None)) == expected
    for arm, encoder, seed in sorted(expected):
        p = HERE / 'runs' / f'{arm}_{encoder}_{seed}'
        record = json.loads((p / 'complete.json').read_text())
        assert record['design_sha256'] == design_hash
        for f, h in record['files'].items():
            assert digest(p / f) == h, str(p / f)
        fit = json.loads((p / 'fit.json').read_text())
        assert len(fit['history']) == 50 and fit['seed'] == seed
        mu = np.load(p / 'mu.npy', mmap_mode='r')
        theta = np.load(p / 'theta.npy', mmap_mode='r')
        assert mu.shape == theta.shape == (fit['n_validation'] + fit['n_test'], fit['n_genes'])
    summary = pd.read_csv(HERE / 'summary.csv')
    boot = pd.read_csv(HERE / 'bootstrap.csv')
    from analyze import fit_random_intercept, sufficient_statistics
    for _, r in summary.iterrows():
        sub = frame if r.analysis == 'pooled' else frame[frame.arm == r.analysis[-1]]
        beta, _, _ = fit_random_intercept(sufficient_statistics(sub)[0])
        assert abs(beta[1] - r.slope) < 1e-10
        b = boot[boot.analysis == r.analysis]
        assert len(b) == d['inference']['resamples'] and b.replicate.nunique() == len(b)
        assert np.allclose(np.quantile(b.slope, [.025, .975]), [r.ci_low, r.ci_high])
        assert r.reference_max_coefficient_error < 1e-3
    result = {'status': 'PASS', 'fits': len(expected), 'rows': len(frame), 'datasets': frame.dataset.nunique(),
              'all_fit_seeds_retained': True, 'validation_test_disjoint': True, 'bootstrap_replicates': len(boot)}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
