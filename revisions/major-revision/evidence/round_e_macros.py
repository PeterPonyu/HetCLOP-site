"""LaTeX macros for the follow-up analyses (selection, label transfer, gene dependence, accession clustering,
corpus composition, scVI sampling sensitivity, training size and ten-seed end-to-end generation).

Every value is read from the experiment output files; nothing is typed by hand. Sections whose outputs are not
yet present are skipped, so the manuscript cannot cite a value that was never computed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
EXP = REV / 'experiments'
OUT = REV / 'manuscript' / 'figs' / 'round_e_numbers.tex'
m = {}


def put(name, value, fmt='{:.3f}'):
    # Plain values; the text uses numeric macros inside math mode, so a hyphen typesets as a minus sign.
    m[name] = value if isinstance(value, str) else fmt.format(value)


def ci(prefix, effect, lo, hi, digits=3):
    # A nonzero value that would round to zero gets one more digit, so an interval never prints a false 0.000 limit.
    for name, x in [(prefix, effect), (prefix + 'Lo', lo), (prefix + 'Hi', hi)]:
        d = digits + 1 if x != 0 and abs(x) < 0.5 * 10 ** -digits else digits
        put(name, x, f'{{:.{d}f}}')


# E3: validation-only selection.
s = pd.read_csv(EXP / 'e3-prospective-selection' / 'summary.csv')
def e3(arm, scope, q):
    return s[(s.arm == arm) & (s.scope == scope) & (s.quantity == q)].iloc[0]
for arm in 'AB':
    A = 'A' if arm == 'A' else 'B'
    for q, key in [('regret', 'Regret'), ('vs_poisson', 'VsPois'), ('vs_per_cell', 'VsCell')]:
        r = e3(arm, 'all encoders', q)
        ci(f'Ethree{A}{key}', r.effect, r.ci_low, r.ci_high)
    for o, key in [('poisson', 'Pois'), ('shared', 'Shared'), ('per_type', 'Type'), ('per_cell', 'Cell')]:
        put(f'Ethree{A}Chose{key}', 100 * e3(arm, 'all encoders', f'chosen_{o}').effect, '{:.0f}')
    put(f'Ethree{A}Best', 100 * e3(arm, 'all encoders', 'share_best_option').effect, '{:.0f}')
    r = e3(arm, 'PCA-512', 'welch_vs_poisson')
    ci(f'Ethree{A}Welch', r.effect, r.ci_low, r.ci_high)
    # The encoder breakdown (Supplementary Table S4) is the reported source for PCA-512 calibration; the 'PCA-512'
    # scope repeats the same quantity with another bootstrap stream only to pair it with marker recall.
    r = e3(arm, 'pca_512', 'vs_poisson')
    ci(f'Ethree{A}PcaVsPois', r.effect, r.ci_low, r.ci_high)
    for enc, key in [('pca_512', 'Pca'), ('randproj_512', 'Rand'), ('scgpt_512', 'Scgpt'), ('geneformer', 'Gf')]:
        put(f'Ethree{A}{key}ChosePois', 100 * e3(arm, enc, 'chosen_poisson').effect, '{:.0f}')

# Matched comparison: masked rare-type F1 (sensitivity form), read from the E1 summary.
e1 = pd.read_csv(EXP / 'e1-armb-biology' / 'summary.csv')
r = e1[(e1.arm == 'B') & (e1.metric == 'rare_f1_masked') & (e1.a == 'per_cell') & (e1.b == 'poisson')].iloc[0]
ci('EoneBCellRareMasked', r.effect, r.ci_low, r.ci_high, 4)

# E4c: label-transfer validation.
t = json.loads((EXP / 'e4c-label-transfer' / 'summary.json').read_text())
for k, key in [('coverage', 'Coverage'), ('accuracy_assigned', 'AccAssigned'), ('accuracy_overall', 'AccOverall'), ('macro_f1', 'MacroF')]:
    ci(f'Efourc{key}', t[k]['mean'], t[k]['ci'][0], t[k]['ci'][1])
put('EfourcMacroFMin', t['macro_f1']['min'])
rr = t['rare_type_recall']
ci('EfourcRareRecall', rr['mean_over_datasets'], rr['ci'][0], rr['ci'][1])
put('EfourcRareTypes', rr['n_rare_types'], '{:d}')
put('EfourcRareShareHigh', 100 * rr['share_types_recall_ge_0.8'], '{:.0f}')
put('EfourcAbundantRecall', t['abundant_type_recall']['pooled_type_mean'])
put('EfourcRareRecallPooled', rr['pooled_type_mean'])
ds = pd.read_csv(EXP / 'e4c-label-transfer' / 'datasets.csv')
put('EfourcNdatasets', len(ds), '{:d}')
# Post hoc restriction of the saved per-type recalls to the primary (pooled test split) rare-type definition.
pr = json.loads((EXP / 'e4c-label-transfer' / 'pooled-rare-check.json').read_text())
put('EfourcPooledRareTypes', pr['pooled_rare_types'], '{:d}')
put('EfourcPooledRarePairs', pr['type_dataset_pairs'], '{:d}')
put('EfourcPooledRareRecall', pr['mean_recall'])
put('EfourcPooledRareMin', pr['min_recall'])
put('EfourcPooledRareHigh', pr['pairs_recall_ge_0.8'], '{:d}')

# E4 composition: transfer shares and per-dataset table summaries.
c = pd.read_csv(EXP / 'e4-sampling-sensitivity' / 'composition-summary.csv', index_col=0)
put('CompTransferUniform', 100 * c.loc[['uniform_s1', 'uniform_s2', 'uniform_s3'], 'transferred_share'].mean(), '{:.1f}')
put('CompTransferCap', 100 * c.loc['cap10000', 'transferred_share'], '{:.1f}')
put('CompTransferFloor', 100 * c.loc['type_floor', 'transferred_share'], '{:.1f}')
put('CompTransferBatch', 100 * c.loc['batch_balanced', 'transferred_share'], '{:.1f}')
cs = json.loads((EXP / 'e4-sampling-sensitivity' / 'composition-by-dataset-summary.json').read_text())
put('CompBatchDatasets', cs['datasets_with_batch_field'], '{:d}')
put('CompSampleDatasets', cs['datasets_with_sample_identity'], '{:d}')
put('CompRareRef', cs['rare_types_pooled']['reference'], '{:d}')
put('CompRareCap', cs['rare_types_pooled']['cap10000'], '{:d}')

# E6: gene-gene dependence.
e6 = pd.read_csv(EXP / 'e6-gene-dependence' / 'summary.csv')
def e6r(arm, metric, a, b):
    return e6[(e6.arm == arm) & (e6.metric == metric) & (e6.a == a) & (e6.b == b)].iloc[0]
for arm in 'AB':
    for metric, a, b, key in [('corr', 'cell', 'poisson', 'CorrCellPois'), ('corr', 'shared', 'poisson', 'CorrSharedPois'),
                              ('corr', 'cell', 'shared', 'CorrCellShared'), ('cal', 'cell', 'poisson', 'CalCellPois'),
                              ('cal', 'shared', 'poisson', 'CalSharedPois')]:
        r = e6r(arm, metric, a, b)
        ci(f'Esix{arm}{key}', r.effect, r.ci_low, r.ci_high, 4 if metric == 'corr' else 3)
    put(f'Esix{arm}Tradeoff', 100 * e6r(arm, 'tradeoff', 'cell', 'poisson').effect, '{:.1f}')
    lv = e6[(e6.arm == arm) & (e6.metric == 'levels')].iloc[0]
    for col, key in [('median_corr_reference', 'Ref'), ('median_corr_poisson', 'Pois'), ('median_corr_shared', 'Shared'), ('median_corr_cell', 'Cell')]:
        put(f'Esix{arm}Level{key}', lv[col])

# E7: accession clustering and the E4 replay.
e7 = json.loads((EXP / 'e7-accession-clustering' / 'summary.json').read_text())
put('EsevenComparisons', e7['comparisons'], '{:d}')
put('EsevenAgree', e7['agreeing'], '{:d}')
put('EsevenReplayRows', e7['replay_rows'], '{:d}')
put('EsevenReplayMatch', min(e7['replay_effects_match'], e7['replay_ci_within_tolerance'], e7['replay_classification_match']), '{:d}')
acc = pd.read_csv(EXP / 'e7-accession-clustering' / 'accession-summary.csv')
r = acc[(acc.analysis == 'e2_slope') & (acc.contrast == 'deficit-effect slope, pooled') & (acc.unit == 'accession')].iloc[0]
ci('EsevenSlope', r.effect, r.ci_low, r.ci_high, 4)
en = acc[(acc.analysis == 'armb_energy') & (acc.unit == 'accession')]
put('EsevenEnergyWorstP', en.wilcoxon_p_worst.max(), '{:.3f}')

# E4-scVI: sampling sensitivity of the deployed scVI contrast.
p = EXP / 'e4-scvi-sensitivity' / 'summary.csv'
if p.exists() and (EXP / 'e4-scvi-sensitivity' / 'ledger.json').exists():
    sv = pd.read_csv(p)
    led = json.loads((EXP / 'e4-scvi-sensitivity' / 'ledger.json').read_text())
    if led['fits_complete'] == led['fits_expected']:
        for mode, key in [('gene', 'Shared'), ('gene-cell', 'Cell')]:
            x = sv[sv['mode'] == mode]
            ref = x[x.variant == 'reference'].iloc[0]
            ci(f'Escvi{key}Ref', ref.effect, ref.ci_low, ref.ci_high)
            put(f'Escvi{key}Min', x.effect.min()); put(f'Escvi{key}Max', x.effect.max())
            put(f'Escvi{key}WinMin', 100 * x.poisson_win_fraction.min(), '{:.0f}')
            put(f'Escvi{key}WinMax', 100 * x.poisson_win_fraction.max(), '{:.0f}')
            put(f'Escvi{key}Stable', int(x[x.variant != 'reference'].stable.sum()), '{:d}')
            put(f'Escvi{key}Below', int((x.classification == 'below zero').sum()), '{:d}')
            put(f'Escvi{key}Above', int((x.classification == 'above zero').sum()), '{:d}')
            put(f'Escvi{key}Spans', int((x.classification == 'spans zero').sum()), '{:d}')
        # Guards for the wording of the Results paragraph; a failure means the text must be revisited.
        sh, ce = sv[sv['mode'] == 'gene'], sv[sv['mode'] == 'gene-cell']
        assert (sh.effect > 0).all() and (sh.classification == 'spans zero').all(), 'scVI shared wording no longer holds'
        assert (ce.classification == 'below zero').all(), 'scVI per-cell wording no longer holds'
        put('EscviStable', led['stable'])
        put('EscviEpochsMin', min(led['epochs_trained']), '{:d}'); put('EscviEpochsMax', max(led['epochs_trained']), '{:d}')
        put('EscviEpochsLow', sorted(led['epochs_trained'])[1], '{:d}')
        assert led['scvi_optimizer']['n_epochs_kl_warmup'] == '400'
        put('EscviKLCap', min(led['epochs_trained']) / 400, '{:.2f}')
        assert sorted(led['epochs_trained'])[0] < 50 <= sorted(led['epochs_trained'])[1], 'epoch wording no longer holds'

# E4b: training-set sizes (inputs fixed before fitting) and results.
ex = np.load(EXP / 'e4b-training-size' / 'extra_meta.npz', allow_pickle=True)
meta = np.load(REV.parents[1] / 'experiments/phase2a_decoder_2026-07-08/cache/meta.npz', allow_pickle=True)
n1 = int((meta['split'] == 0).sum())
put('EfourbTrainOne', n1, '{:,}'); put('EfourbTrainTwo', n1 + int(ex['half'].sum()), '{:,}')
put('EfourbTrainThree', n1 + len(ex['half']), '{:,}'); put('EfourbExtra', len(ex['half']), '{:,}')
# E4b: training size with fixed evaluation cells.
p = EXP / 'e4b-training-size' / 'summary.csv'
if p.exists():
    tb = pd.read_csv(p)
    if {'x1', 'x2', 'x3', 'x3 minus x1'} <= set(tb['size']):
        for label, key in [('Arm B per-cell minus Poisson calibration error', 'BCal'), ('Arm A per-cell minus Poisson calibration error', 'ACal'),
                           ('Arm B per-cell minus shared calibration error', 'BAlloc'), ('Arm B per-cell minus Poisson Welch recall', 'BWelch'),
                           ('Arm B per-cell minus shared Welch recall', 'BWelchAlloc')]:
            # x1 is the reference corpus, whose values are cited from the corpus analysis (one interval per estimate).
            for size, sk in [('x2', 'Two'), ('x3', 'Three'), ('x3 minus x1', 'Trend')]:
                r = tb[(tb['size'] == size) & (tb.contrast == label)].iloc[0]
                ci(f'Efourb{key}{sk}', r.effect, r.ci_low, r.ci_high)
        for size, n in [('x2', n1 + int(ex['half'].sum())), ('x3', n1 + len(ex['half']))]:
            assert int(tb[tb['size'] == size].n_train.iloc[0]) == n, 'fitted training size differs from the built set'
        # Guards for the Results wording on training size.
        g = lambda size, label: tb[(tb['size'] == size) & (tb.contrast == label)].iloc[0]
        assert g('x3 minus x1', 'Arm B per-cell minus Poisson calibration error').ci_high < 0, 'calibration trend wording'
        assert g('x3', 'Arm B per-cell minus Poisson Welch recall').ci_high < 0, 'persisting marker deficit wording'
        assert g('x3 minus x1', 'Arm B per-cell minus Poisson Welch recall').ci_low < 0 < g('x3 minus x1', 'Arm B per-cell minus Poisson Welch recall').ci_high
        assert g('x3', 'Arm A per-cell minus Poisson calibration error').ci_low > 0, 'Arm A penalty wording'
        for lab in ['Arm A per-cell minus Poisson calibration error', 'Arm B per-cell minus shared calibration error',
                    'Arm B per-cell minus shared Welch recall']:
            r = g('x3 minus x1', lab)
            assert r.ci_low < 0 < r.ci_high, f'unchanged-with-size wording: {lab}'
        cap = np.load(EXP / 'e4-sampling-sensitivity' / 'corpora' / 'cap10000' / 'meta.npz', allow_pickle=True)
        put('EfourCapTrain', int((cap['split'] == 0).sum()), '{:,}')
        assert int((cap['split'] == 0).sum()) < n1 + len(ex['half'])

# E5: ten-seed end-to-end generation.
p = EXP / 'e5-endtoend-seeds' / 'summary.csv'
if p.exists():
    e5 = pd.read_csv(p)
    n = int(e5[e5.quantity == 'poisson'].n_seeds.max())
    if n == 10:
        put('EfiveSeeds', n, '{:d}')
        for ep, ek in [('marker_recall_mean', 'Marker'), ('tstr_macro_f1', 'Macro'), ('tstr_rare_f1', 'Rare'), ('tstr_rare_f1_unmasked', 'RareU')]:
            x = e5[e5.endpoint == ep].set_index('quantity')
            for arm, ak in [('poisson', 'Pois'), ('shared', 'Shared'), ('per_cell', 'Cell')]:
                put(f'Efive{ek}{ak}', x.loc[arm, 'mean']); put(f'Efive{ek}{ak}Sd', x.loc[arm, 'sd'])
            for q, qk in [('per_cell minus poisson', 'CellPois'), ('shared minus poisson', 'SharedPois'), ('per_cell minus shared', 'CellShared')]:
                ci(f'Efive{ek}{qk}', x.loc[q, 'mean'], x.loc[q, 'ci_low'], x.loc[q, 'ci_high'])
                put(f'Efive{ek}{qk}Below', int(x.loc[q, 'seeds_below_zero']), '{:d}')
        # Guards for the Results, Discussion, abstract and Conclusion wording on ten seeds.
        q = lambda ep, k: e5[(e5.endpoint == ep) & (e5.quantity == k)].iloc[0]
        for k in ['per_cell minus poisson', 'shared minus poisson']:
            assert q('marker_recall_mean', k).seeds_below_zero == n and q('marker_recall_mean', k).ci_high < 0
            assert q('tstr_macro_f1', k).seeds_below_zero == 0 and q('tstr_macro_f1', k).ci_low > 0
        assert q('tstr_rare_f1', 'per_cell minus poisson').ci_high < 0
        assert q('tstr_rare_f1', 'shared minus poisson').ci_low < 0 < q('tstr_rare_f1', 'shared minus poisson').ci_high
        m_ = e5[e5.endpoint == 'marker_recall_mean'].set_index('quantity')['mean']
        assert m_['poisson'] > max(m_['shared'], m_['per_cell'])
        per5 = pd.read_csv(EXP / 'e5-endtoend-seeds' / 'per-seed.csv')
        put('EfiveScoreMin', int(per5.n_scoreable_types.min()), '{:d}'); put('EfiveScoreMax', int(per5.n_scoreable_types.max()), '{:d}')
        w5 = per5.pivot(index='seed', columns='arm', values='tstr_rare_f1')
        d5 = (w5['per_cell'] - w5['poisson']).to_numpy()
        put('EfiveRareCellPoisNear', int(((d5 < 0) & (d5 > -1e-4)).sum()), '{:d}')
        put('EfiveRareOrderings', int(e5[(e5.endpoint == 'tstr_rare_f1') & e5.quantity.str.startswith('ordering')].shape[0]), '{:d}')

lines = ['% Follow-up analyses; generated by evidence/round_e_macros.py from the experiment outputs']
lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.items()]
OUT.write_text('\n'.join(lines) + '\n')
print(len(m), 'macros written to', OUT)
