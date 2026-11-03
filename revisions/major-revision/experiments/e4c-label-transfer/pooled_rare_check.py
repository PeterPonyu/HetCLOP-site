"""Post hoc, descriptive: label-transfer recall for the rare types of the primary definition.

The fixed E4c design scores 'rare' as types with fewer than 50 test cells in their own dataset. The Methods'
primary rare-type set is narrower: labels with fewer than 50 cells in the pooled test split. Every pooled-rare
label is also rare within each dataset that contains it, so this check only restricts the saved per-type
recalls in types.csv to that set. No model is refitted and nothing in the fixed design is changed.
Output: pooled-rare-check.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
meta = np.load(ROOT / 'experiments/phase2a_decoder_2026-07-08/cache/meta.npz', allow_pickle=True)
lab, split = meta['labels'].astype(str), meta['split']
u, c = np.unique(lab[split == 2], return_counts=True)
pooled = set(u[c < 50])

t = pd.read_csv(HERE / 'types.csv')
sub = t[t.cell_type.isin(pooled)]
assert sub.rare.all(), 'a pooled-rare type is not rare within its dataset'
out = {
    'note': 'post hoc restriction of the saved per-type recalls; descriptive only',
    'pooled_rare_types': len(pooled),
    'pooled_rare_types_scored': int(sub.cell_type.nunique()),
    'type_dataset_pairs': int(len(sub)),
    'datasets': int(sub.dataset.nunique()),
    'mean_recall': float(sub.recall.mean()),
    'median_recall': float(sub.recall.median()),
    'min_recall': float(sub.recall.min()),
    'min_recall_pair': sub.loc[sub.recall.idxmin(), ['dataset', 'cell_type', 'n']].astype(str).to_dict(),
    'pairs_recall_ge_0.8': int((sub.recall >= 0.8).sum()),
}
(HERE / 'pooled-rare-check.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
