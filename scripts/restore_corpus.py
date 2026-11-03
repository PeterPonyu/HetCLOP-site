"""Place the deposited analysed corpus where the analysis scripts expect it, and verify every file.

Usage: python scripts/restore_corpus.py /path/to/hetclop-corpus-<version>

Writes experiments/phase2a_decoder_2026-07-08/cache/ (dense count and log-count matrices, split metadata and gene
panel) and the training-size analysis's extra-cell metadata. The dense matrices are rebuilt from their sparse copies
and must match the recorded SHA-256 digests byte for byte. The scGPT and Geneformer embeddings are not deposited;
recompute them with experiments/phase2a_decoder_2026-07-08/run_phase2a.py --stage encode (public model weights under
data/models/).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main(data: Path):
    sums = json.loads((data / 'SHA256SUMS.json').read_text())
    bad = [k for k, v in sums.items() if sha(data / k) != v]
    if bad:
        raise SystemExit(f'deposit files do not match SHA256SUMS.json: {bad}')
    manifest = json.loads((data / 'input-data-manifest.json').read_text())
    cache = REPO / 'experiments/phase2a_decoder_2026-07-08/cache'
    cache.mkdir(parents=True, exist_ok=True)
    for name in ['raw_full.npy', 'logX_full.npy']:
        m = sp.load_npz(data / name.replace('.npy', '.sparse.npz'))
        np.save(cache / name, m.toarray().astype(np.float32))
    for name in ['meta.npz', 'panel.json']:
        shutil.copy2(data / name, cache / name)
    shutil.copy2(data / 'training-size/extra_meta.npz',
                 REPO / 'revisions/major-revision/experiments/e4b-training-size/extra_meta.npz')
    wrong = [k for k, v in manifest.items() if sha(cache / k) != v]
    if wrong:
        raise SystemExit(f'restored cache differs from the analysed inputs: {wrong}')
    print('restored and verified', len(manifest), 'cache files in', cache.relative_to(REPO))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
