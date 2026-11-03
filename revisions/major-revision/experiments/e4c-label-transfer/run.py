"""E4c: cross-validated check of the E4 label-transfer rule on held-out original annotations."""
import hashlib
import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
E4 = HERE.parent / 'e4-sampling-sensitivity'
spec = importlib.util.spec_from_file_location('bc', E4 / 'build_corpus.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
U = bc.UNLABELED

meta = np.load(bc.CACHE / 'meta.npz', allow_pickle=True)
ds_all, lab_all, split_all = meta['dataset_ids'].astype(str), meta['labels'].astype(str), meta['split']
rows, types_rows = [], []
for name in list(meta['ordered']):
    proc = ad.read_h5ad(bc.CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000' / f'{name}_processed.h5ad')
    plabels = bc.build_cell_type_labels(name, proc, bc.CORPUS_SOURCE_ROOT)
    y = np.array([U if v is None else str(v) for v in plabels], dtype=object)
    import scipy.sparse as sp
    X = sp.csr_matrix(proc.layers['counts'], dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    F = X.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr()
    F.data = np.log1p(F.data)
    F = F.toarray()
    m = ds_all == name
    test_counts = pd.Series(lab_all[m & (split_all == 2)]).value_counts()
    rare = set(test_counts[test_counts < 50].index) | (set(lab_all[m]) - set(test_counts.index))
    pred = np.empty(len(y), dtype=object)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(F):
        pca = PCA(n_components=min(30, len(tr) - 1), random_state=0).fit(F[tr])
        knn = KNeighborsClassifier(n_neighbors=15).fit(pca.transform(F[tr]), y[tr])
        prob = knn.predict_proba(pca.transform(F[te]))
        p = knn.classes_[prob.argmax(1)].astype(object)
        p[prob.max(1) < 0.5] = U
        pred[te] = p
    lab = y != U
    yt, yp = y[lab], pred[lab]
    assigned = yp != U
    types = sorted(set(yt))
    rows.append({'dataset': name, 'n_labelled': int(lab.sum()), 'n_types': len(types),
                 'coverage': float(assigned.mean()),
                 'accuracy_assigned': float((yp[assigned] == yt[assigned]).mean()) if assigned.any() else np.nan,
                 'accuracy_overall': float((yp == yt).mean()),
                 'macro_f1': float(f1_score(yt, yp, labels=types, average='macro', zero_division=0)),
                 'n_rare_types': len(rare & set(types))})
    for t in types:
        inside = yt == t
        types_rows.append({'dataset': name, 'cell_type': t, 'n': int(inside.sum()), 'rare': t in rare,
                           'recall': float((yp[inside] == t).mean())})
    print(name, rows[-1]['coverage'], rows[-1]['accuracy_overall'], rows[-1]['macro_f1'], flush=True)

d = pd.DataFrame(rows); t = pd.DataFrame(types_rows)
d.to_csv(HERE / 'datasets.csv', index=False); t.to_csv(HERE / 'types.csv', index=False)
rng = np.random.default_rng(20261109)
summ = {}
for col in ['coverage', 'accuracy_assigned', 'accuracy_overall', 'macro_f1']:
    v = d[col].to_numpy()
    b = rng.choice(v, size=(2000, len(v))).mean(1)
    summ[col] = {'mean': float(v.mean()), 'ci': [float(x) for x in np.quantile(b, [.025, .975])],
                 'min': float(v.min()), 'median': float(np.median(v))}
r = t[t.rare]
per_ds_rare = r.groupby('dataset').recall.mean()
b = rng.choice(per_ds_rare.to_numpy(), size=(2000, len(per_ds_rare))).mean(1)
summ['rare_type_recall'] = {'n_rare_types': int(len(r)), 'n_datasets': int(per_ds_rare.size),
                            'mean_over_datasets': float(per_ds_rare.mean()), 'ci': [float(x) for x in np.quantile(b, [.025, .975])],
                            'pooled_type_mean': float(r.recall.mean()), 'share_types_recall_ge_0.8': float((r.recall >= 0.8).mean())}
summ['abundant_type_recall'] = {'pooled_type_mean': float(t[~t.rare].recall.mean())}
(HERE / 'summary.json').write_text(json.dumps(summ, indent=2) + '\n')
print(json.dumps(summ, indent=2))
