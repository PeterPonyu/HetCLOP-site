"""E5: run additional end-to-end seeds with the archived protocol, writing to this folder only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
E2E = ROOT / 'experiments' / 'phase2v_e2e_2026-07-19'
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(ROOT / 'experiments' / 'phase2j_external_2026-07-09'))
sys.path.insert(0, str(ROOT / 'experiments' / 'phase2q_biological_2026-07-10'))
sys.path.insert(0, str(ROOT))
import phase2j_lib as L  # noqa: E402
import run_e2e_multiseed as M  # noqa: E402
from run_biological import MIN_CELLS, RARE_MAX, marker_scores  # noqa: E402

assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
seeds = [int(s) for s in sys.argv[1].split(',')]
(HERE / 'results').mkdir(exist_ok=True)
data = L.load_corpus()
real = data['logX'][data['te']]
lab_te = data['labels'][data['te']]
u, c = np.unique(lab_te, return_counts=True)
types = [t for t, n in zip(u, c) if n >= MIN_CELLS]
rare = {t for t, n in zip(u, c) if n < RARE_MAX}
real_ms = marker_scores(real, lab_te, types)
for seed in seeds:
    out = HERE / 'results' / f'e2e_generation_seed{seed}.json'
    if out.exists():
        continue
    res = M.run_one_seed(data, seed, real, lab_te, real_ms, types, rare)
    out.write_text(json.dumps(res, indent=2))
print('E5 seeds done', seeds, flush=True)
