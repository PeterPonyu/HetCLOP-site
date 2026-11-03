"""E4-scVI analysis: scVI shared per-gene and per-cell dispersion against Poisson sampling at the model's own
fitted mean, on each E4 corpus. Follows design.json: seeds averaged within group, groups within dataset,
equal-weight mean over datasets, 2000-resample dataset-cluster bootstrap (seed 20261106), group-level
Poisson-win fraction, stability against the reference corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'design.json').read_text())
VARIANTS = ['reference', 'uniform_s1', 'uniform_s2', 'uniform_s3', 'type_floor', 'batch_balanced', 'cap10000', 'mad_qc']
MODES = {'gene': 'shared per-gene NB minus Poisson calibration error', 'gene-cell': 'per-cell NB minus Poisson calibration error'}


def classify(lo, hi):
    return 'below zero' if hi < 0 else ('above zero' if lo > 0 else 'spans zero')


def main():
    assert hashlib.sha256((HERE / 'design.json').read_bytes()).hexdigest() == (HERE / 'design.sha256').read_text().strip()
    done = sorted((HERE / 'runs').glob('*/complete.json'))
    frame = pd.concat([pd.read_csv(p.parent / 'groups.csv') for p in done], ignore_index=True)
    fits = [json.loads((p.parent / 'fit.json').read_text()) for p in done]
    rows = []
    for v in VARIANTS:
        for mode, label in MODES.items():
            sub = frame[(frame.variant == v) & (frame['mode'] == mode)]
            if sub.seed.nunique() < len(DESIGN['fit_seeds']):
                continue
            g = sub.assign(d=sub.cal_nb - sub.cal_poisson).groupby(['dataset', 'cell_type']).agg(
                d=('d', 'mean'), signed_nb=('signed_nb', 'mean'), signed_poisson=('signed_poisson', 'mean'))
            per_ds = g.d.groupby(level=0).mean().to_numpy()
            rng = np.random.default_rng(20261106)
            boot = rng.choice(per_ds, size=(2000, len(per_ds)), replace=True).mean(1)
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append({'variant': v, 'mode': mode, 'contrast': label, 'effect': float(per_ds.mean()), 'ci_low': float(lo),
                         'ci_high': float(hi), 'classification': classify(lo, hi),
                         'poisson_win_fraction': float((g.d > 0).mean()), 'n_groups': len(g), 'n_datasets': len(per_ds),
                         'median_signed_ratio_nb': float(g.signed_nb.median()),
                         'median_signed_ratio_poisson': float(g.signed_poisson.median()),
                         'n_seeds': int(sub.seed.nunique())})
    out = pd.DataFrame(rows)
    ref = out[out.variant == 'reference'].set_index('mode')
    out['stable'] = out.classification == out['mode'].map(ref.classification)
    out['difference_from_reference'] = out.effect - out['mode'].map(ref.effect)
    out.to_csv(HERE / 'summary.csv', index=False)
    epochs = sorted({f['settings']['epochs_trained'] for f in fits})
    s0 = fits[0]['settings']
    record = {'design_sha256': (HERE / 'design.sha256').read_text().strip(), 'fits_complete': len(done),
              'fits_expected': len(VARIANTS) * len(MODES) * len(DESIGN['fit_seeds']),
              'stable': f"{int(out[out.variant != 'reference'].stable.sum())}/{int((out.variant != 'reference').sum())}",
              'epochs_trained': epochs, 'seconds_per_fit_median': float(np.median([f['seconds'] for f in fits])),
              'scvi_settings': {k: s0[k] for k in ['scvi_version', 'n_latent', 'n_hidden', 'n_layers', 'gene_likelihood',
                                                    'latent_distribution', 'encode_covariates']},
              'scvi_dropout': s0['init_params'].get('dropout_rate'),
              'scvi_use_observed_lib_size': s0['init_params'].get('use_observed_lib_size'),
              'scvi_optimizer': {k: s0['plan_defaults'][k] for k in ['optimizer', 'lr', 'weight_decay', 'eps', 'n_epochs_kl_warmup']},
              'scvi_batch_size': s0['train_defaults']['batch_size']}
    (HERE / 'ledger.json').write_text(json.dumps(record, indent=2) + '\n')
    print(out[['variant', 'mode', 'effect', 'ci_low', 'ci_high', 'classification', 'poisson_win_fraction', 'stable']].round(4).to_string())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
