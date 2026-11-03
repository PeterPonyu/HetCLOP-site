"""E4-scVI: refit scVI (shared per-gene and per-cell dispersion) on each E4 corpus and score the
fixed-mean dispersion-versus-Poisson calibration contrast per (dataset, cell type) group.

Also records the instantiated scVI architecture and training settings, so the Methods can report
recovered values rather than "package defaults".
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scvi  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.architecture.stochastic_decoder import sample_nb  # noqa: E402
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

CORPORA = HERE.parent / 'e4-sampling-sensitivity' / 'corpora'
DESIGN = json.loads((HERE / 'design.json').read_text())


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def adata(counts, ds):
    A = ad.AnnData(sp.csr_matrix(counts), obs=pd.DataFrame({'dataset': ds.astype(str)}))
    A.layers['counts'] = A.X.copy()
    return A


def settings(model):
    mod = model.module
    rec = {'scvi_version': scvi.__version__, 'init_params': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                             for k, v in model.init_params_.get('non_kwargs', {}).items()}}
    for k in ['n_latent', 'n_hidden', 'n_layers', 'dropout_rate', 'dispersion', 'gene_likelihood', 'latent_distribution',
              'encode_covariates', 'use_batch_norm', 'use_layer_norm']:
        if hasattr(mod, k):
            rec[k] = getattr(mod, k) if isinstance(getattr(mod, k), (int, float, str, bool)) else str(getattr(mod, k))
    sig = inspect.signature(model.train)
    rec['train_defaults'] = {k: str(v.default) for k, v in sig.parameters.items() if v.default is not inspect._empty}
    tp = inspect.signature(scvi.train.TrainingPlan.__init__)
    rec['plan_defaults'] = {k: str(v.default) for k, v in tp.parameters.items() if v.default is not inspect._empty}
    rec['epochs_trained'] = int(len(model.history['elbo_train'])) if 'elbo_train' in model.history else -1
    return rec


def groups(real, draws, labels, datasets):
    rows = []
    for d in np.unique(datasets):
        m = datasets == d
        for t, c in zip(*np.unique(labels[m], return_counts=True)):
            if c < 20:
                continue
            inside = m & (labels == t)
            row = {'dataset': d, 'cell_type': t, 'n_test': int(c)}
            for k, x in draws.items():
                e = r_var_per_gene_expressed(real[inside], x[inside])['expressed']
                row[f'cal_{k}'] = e['median_abs_log2_ratio']
                row[f'signed_{k}'] = e['median_signed_ratio']
            rows.append(row)
    return rows


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    variants = sys.argv[1].split(',')
    modes = (sys.argv[2] if len(sys.argv) > 2 else 'gene,gene-cell').split(',')
    torch.set_float32_matmul_precision('high')
    for v in variants:
        meta = np.load(CORPORA / v / 'meta.npz', allow_pickle=True)
        counts = sp.load_npz(CORPORA / v / 'counts.npz').tocsr().astype(np.float32)
        ds, lab, split = meta['dataset_ids'].astype(str), meta['labels'].astype(str), meta['split']
        tr, te = split == 0, split == 2
        A_tr, A_te = adata(counts[tr], ds[tr]), adata(counts[te], ds[te])
        real = np.log1p(counts[te].toarray())
        for mode in modes:
            for seed in DESIGN['fit_seeds']:
                out = HERE / 'runs' / f'{v}_{mode}_{seed}'
                if (out / 'complete.json').exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                scvi.settings.seed = seed
                scvi.model.SCVI.setup_anndata(A_tr, layer='counts', batch_key='dataset')
                model = scvi.model.SCVI(A_tr, gene_likelihood='nb', dispersion=mode)
                model.train(accelerator='gpu', enable_progress_bar=False, early_stopping=True)
                params = model.get_likelihood_parameters(adata=A_te, n_samples=10, give_mean=True)
                mu = np.asarray(params['mean'], dtype=np.float64)
                theta = np.asarray(params['dispersions'], dtype=np.float64)
                if theta.ndim == 1:
                    theta = np.broadcast_to(theta, mu.shape).copy()
                draws = {'poisson': np.log1p(np.random.default_rng((seed, 2)).poisson(mu).astype(np.float32)),
                         'nb': np.log1p(sample_nb(mu, theta, seed=seed + 1))}
                rows = groups(real, draws, lab[te], ds[te])
                for r in rows:
                    r.update({'variant': v, 'mode': mode, 'seed': seed})
                pd.DataFrame(rows).to_csv(out / 'groups.csv', index=False)
                (out / 'fit.json').write_text(json.dumps({'variant': v, 'mode': mode, 'seed': seed,
                    'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'seconds': time.time() - t0,
                    'theta_median': float(np.median(theta)), 'settings': settings(model)}, indent=2, default=str))
                (out / 'complete.json').write_text(json.dumps({'groups': len(rows)}))
                log(f'{v} {mode} seed={seed} groups={len(rows)} epochs={settings(model)["epochs_trained"]} {time.time() - t0:.0f}s')
                del model, mu, theta, draws
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
