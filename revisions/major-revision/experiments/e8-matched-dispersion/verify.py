"""E8 verification: design hash, completeness, inputs, validation-only shared values, the matching property, and that
the analysis outputs regenerate exactly. Writes verification.json."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP = HERE.parent
CACHE = ROOT / 'experiments/phase2a_decoder_2026-07-08/cache'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def design_ok() -> bool:
    """The design hash recorded before execution, or, in a released copy whose descriptive note was reworded, the
    released digest that ../released-source-digests.json pairs with that recorded hash."""
    recorded = (HERE / 'design.sha256').read_text().strip()
    now = digest(HERE / 'design.json')
    if now == recorded:
        return True
    rec = HERE.parent / 'released-source-digests.json'
    alt = json.loads(rec.read_text())['files'].get('revisions/major-revision/experiments/e8-matched-dispersion/design.json', {}) \
        if rec.exists() else {}
    return alt.get('analysed') == recorded and alt.get('released') == now



def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    checks = {}
    checks['design-hash'] = design_ok()
    want = {f'x1_{a}_{s}' for a in 'AB' for s in (42, 43, 44)} | {f'x3_B_{s}' for s in (42, 43, 44)}
    have = {d.name for d in (HERE / 'runs').glob('*') if (d / 'complete.json').exists()}
    checks['all-fits-complete'] = want == have
    files_ok = all((HERE / 'runs' / r / f).exists() for r in want for f in
                   ['groups.csv', 'dataset-classification.csv', 'pooled-classification.csv', 'shared-theta.npz',
                    'achieved-variance.json'])
    checks['outputs-present'] = files_ok
    # Matching property and validation-only recomputation where the fitted arrays are available.
    meta = np.load(CACHE / 'meta.npz', allow_pickle=True)
    panel = json.loads((CACHE / 'panel.json').read_text())
    gout = np.array(panel['gout_idx'])
    nv = int((meta['split'] == 1).sum())
    matched, recomputed, sources = [], [], []
    for r in sorted(want):
        size, arm, seed = r.split('_')
        src = (EXP / 'e2-dose-response' / 'runs' / f'{arm}_pca_512_{seed}') if size == 'x1' else HERE / 'runs' / r
        if not (src / 'mu.npy').exists():
            sources.append(f'{r}: fitted arrays not distributed; recorded shared values used')
            continue
        mu, theta = np.load(src / 'mu.npy', mmap_mode='r'), np.load(src / 'theta.npy', mmap_mode='r')
        if size == 'x1' and arm == 'A':
            mu, theta = mu[:, gout], theta[:, gout]
        mv, tv = np.asarray(mu[:nv], dtype=np.float64), np.asarray(theta[:nv], dtype=np.float64)
        sh = np.load(HERE / 'runs' / r / 'shared-theta.npz')
        lo, hi = np.exp(-4.0), np.exp(8.0)
        target = (mv ** 2 / tv).sum(0)
        inside = (sh['shared_matched'] > lo) & (sh['shared_matched'] < hi)
        matched.append(bool(np.allclose((mv ** 2 / sh['shared_matched']).sum(0)[inside], target[inside], rtol=1e-6)))
        recomputed.append(bool(np.allclose(np.median(np.asarray(theta[:nv]), axis=0), sh['shared_median'])))
    # In a copy without the fitted arrays (the public archive) these two checks cannot run and are reported as such.
    na = 'not checkable: fitted arrays not in this copy'
    checks['matched-extra-variance-on-validation'] = bool(all(matched)) if matched else na
    checks['median-recomputed-from-validation'] = bool(all(recomputed)) if recomputed else na
    checks['arrays-checked'] = len(matched)
    checks['notes'] = sources
    if (HERE / 'input-manifest.json').exists():
        man = json.loads((HERE / 'input-manifest.json').read_text())
        present = {k: v for k, v in man.items() if (ROOT / k).exists()}
        checks['input-digests'] = (all(digest(ROOT / k) == v for k, v in present.items()) if present
                                   else 'not checkable: fitted arrays not in this copy')
    # The analysis regenerates its tables exactly.
    an = load('e8analyze', HERE / 'analyze.py')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(HERE / 'runs', tmp / 'runs', ignore=shutil.ignore_patterns('*.npy'))
        an.HERE = tmp
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            an.main()
        checks['analysis-regenerates'] = all((tmp / f).read_bytes() == (HERE / f).read_bytes()
                                             for f in ['summary.csv', 'dataset-effects.csv', 'joint-direction.csv', 'variance.csv'])
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    checks['status'] = 'PASS' if ok else 'FAIL'
    (HERE / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
