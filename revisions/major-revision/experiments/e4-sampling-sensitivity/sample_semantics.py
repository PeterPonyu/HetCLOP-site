"""Descriptive scan of the sample-level metadata in each source file (no inference).

For every dataset, records the obs columns that could identify donors or samples, the number of levels of the
harmonised 'batch' field used by the batch-balanced variant, and whether that field coincides with a
donor-like or sample-like column. Output: sample-semantics.csv.
"""
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.utils.paths import CORPUS_SOURCE_ROOT  # noqa: E402

DONOR = re.compile(r'donor|patient|individual|subject|person|participant|mouse_id|animal|^mouse$|^pt$|pid', re.I)
SAMPLE = re.compile(r'sample|orig\.ident|library|channel|^lib|gsm|replicate|^rep$|run|lane|10x|batch', re.I)


def same_partition(a, b):
    t = pd.crosstab(a, b)
    return bool(((t > 0).sum(1) == 1).all() and ((t > 0).sum(0) == 1).all())


def nested(a, b):
    """True when every level of a lies inside one level of b."""
    t = pd.crosstab(a, b)
    return bool(((t > 0).sum(1) == 1).all())


def scan(name):
    info = json.loads((CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000/metadata_structured.json').read_text())
    a = ad.read_h5ad(info[name]['source_file'], backed='r')
    obs = a.obs
    batch = obs['batch'].astype(str) if 'batch' in obs else None
    rec = {'dataset': name, 'n_obs': a.n_obs, 'has_batch': batch is not None,
           'batch_levels': int(batch.nunique()) if batch is not None else 0, 'columns': '|'.join(map(str, obs.columns))}
    donors, samples = [], []
    for c in obs.columns:
        if c == 'batch':
            continue
        s = obs[c]
        if not (s.dtype == object or str(s.dtype) == 'category' or pd.api.types.is_integer_dtype(s)):
            continue
        k = int(s.astype(str).nunique())
        if k < 2 or k > max(2000, a.n_obs // 5):
            continue
        if DONOR.search(str(c)):
            donors.append((c, k))
        elif SAMPLE.search(str(c)):
            samples.append((c, k))
    rec['donor_columns'] = '|'.join(f'{c}:{k}' for c, k in donors)
    rec['sample_columns'] = '|'.join(f'{c}:{k}' for c, k in samples)
    match = ''
    if batch is not None:
        for kind, cols in [('donor', donors), ('sample', samples)]:
            for c, _ in cols:
                if same_partition(batch, obs[c].astype(str)):
                    match = match or f'{kind}:{c}'
        rec['batch_matches'] = match
        rec['batch_nested_in_donor'] = any(nested(batch, obs[c].astype(str)) for c, _ in donors)
        rec['donor_nested_in_batch'] = any(nested(obs[c].astype(str), batch) for c, _ in donors)
    return rec


if __name__ == '__main__':
    meta = np.load(ROOT / 'experiments/phase2a_decoder_2026-07-08/cache/meta.npz', allow_pickle=True)
    with Pool(4) as p:
        rows = p.map(scan, list(meta['ordered']))
    pd.DataFrame(rows).to_csv(HERE / 'sample-semantics.csv', index=False)
    print(pd.DataFrame(rows)[['dataset', 'has_batch', 'batch_levels', 'donor_columns', 'sample_columns', 'batch_matches']].to_string())
