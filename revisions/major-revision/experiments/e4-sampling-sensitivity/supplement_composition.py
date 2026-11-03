"""Per-dataset composition of the E4 corpora for the supplementary table (descriptive; no inference).

Reads the saved corpora, composition.csv and sample-semantics.csv and writes composition-by-dataset.csv
plus the LaTeX table manuscript/figs/tab_composition.tex. Rare types follow the Methods definition used for
classification: cell-type labels with fewer than 50 cells in a corpus's pooled test split.
"""
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MS = HERE.parents[1] / 'manuscript'
sys.path.insert(0, str(ROOT))
from src.data.labels import build_cell_type_labels  # noqa: E402
from src.utils.paths import CORPUS_SOURCE_ROOT  # noqa: E402

VARIANTS = ['reference', 'uniform_s1', 'uniform_s2', 'uniform_s3', 'type_floor', 'batch_balanced', 'cap10000', 'mad_qc']
# Sample identity recorded outside the harmonised batch field, found by inspecting every source file's
# cell metadata (sample-semantics.csv lists all columns). No source file records donor identity.
SAMPLE_FIELDS = {
    'GSE130148_LungHmDev': ('GEO_Sample', 'GEO sample'),
    'bm_GSE120446': ('gsm_id', 'GEO sample'),
    'dentate': ('SampleID', '10x channel'),
    'lung': ('identifier', 'sample'),
    'hemato': ('orig.ident', 'sample = batch'),
}


def labels_for_rows():
    """Map dataset id -> accession label used in Table 1 (accession plus a/b superscript)."""
    dt = pd.read_csv(MS / 'figs' / 'dataset_table.csv')
    tex = (MS / 'figs' / 'tab_datasets.tex').read_text()
    out = {}
    for acc, sup, sp, n, t in re.findall(r'(GSE\d+)(?:\$\^\{\\mathrm\{([ab])\}\}\$)? & (Mm|Hs) & (\d+) & (\d+)', tex):
        hit = dt[(dt.accession == acc) & (dt.n_cells == int(n)) & (dt.n_cell_types == int(t))]
        assert len(hit) == 1, (acc, n, t)
        out[hit.dataset_id.iloc[0]] = (acc + (f'$^{{\\mathrm{{{sup}}}}}$' if sup else ''), sp)
    assert len(out) == 50
    # Table 1 reading order: accession number, then the a/b superscript.
    return dict(sorted(out.items(), key=lambda kv: (int(re.match(r'GSE(\d+)', kv[1][0]).group(1)), kv[1][0])))


def main():
    comp = pd.read_csv(HERE / 'composition.csv').set_index(['variant', 'dataset'])
    sem = pd.read_csv(HERE / 'sample-semantics.csv').set_index('dataset')
    info = json.loads((CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000/metadata_structured.json').read_text())
    acc = labels_for_rows()
    rare_sets, per = {}, {}
    for v in VARIANTS:
        z = np.load(HERE / 'corpora' / v / 'meta.npz', allow_pickle=True)
        ds, lab, split = z['dataset_ids'].astype(str), z['labels'].astype(str), z['split']
        u, c = np.unique(lab[split == 2], return_counts=True)
        rare = set(u[c < 50])
        rare_sets[v] = rare
        for d in np.unique(ds):
            t = lab[(ds == d) & (split == 2)]
            types, counts = np.unique(t, return_counts=True)
            per[(v, d)] = {'test_cells': int(len(t)), 'test_types': int(len(types)),
                           'scored_groups': int((counts >= 20).sum()),
                           'rare_test_types': int(sum(x in rare for x in types)),
                           'rare_test_cells': int(sum(n for x, n in zip(types, counts) if x in rare))}
    e1_rare = set(json.loads((HERE.parent / 'e1-armb-biology/runs/B_42/fit-log.json').read_text())['rare_types'])
    assert rare_sets['reference'] == e1_rare, 'reference rare set differs from the classification analyses'

    rows = []
    for d in acc:
        r = {'dataset': d, 'accession': acc[d][0], 'species': acc[d][1], 'source_cells': int(comp.loc[('reference', d), 'source_cells'])}
        r['batch_levels_source'] = int(comp.loc[('reference', d), 'batch_levels_source']) if sem.loc[d, 'has_batch'] else 0
        for v in ['reference', 'batch_balanced', 'cap10000']:
            r[f'batch_levels_{v}'] = int(comp.loc[(v, d), 'batch_levels_retained']) if sem.loc[d, 'has_batch'] else 0
        r['batch_fallback'] = bool(comp.loc[('batch_balanced', d), 'batch_fallback'])
        if d in SAMPLE_FIELDS:
            col, meaning = SAMPLE_FIELDS[d]
            src = ad.read_h5ad(info[d]['source_file'], backed='r')
            proc = ad.read_h5ad(CORPUS_SOURCE_ROOT / 'data/processed_h5ad_hvg8000' / f'{d}_processed.h5ad', backed='r')
            plab = build_cell_type_labels(d, proc, CORPUS_SOURCE_ROOT)
            kept = np.array([x is not None for x in plab])
            r['sample_identity'] = meaning
            r['sample_levels_source'] = int(src.obs[col].astype(str).nunique())
            r['sample_levels_reference'] = int(proc.obs[col].astype(str)[kept].nunique())
            assert kept.sum() == int(comp.loc[('reference', d), 'analysed_cells']), f'{d}: labelled processed cells differ from the reference corpus'
        else:
            r['sample_identity'] = 'not recorded'
            r['sample_levels_source'] = r['sample_levels_reference'] = 0
        r['donor_identity'] = 'not recorded'
        for v in VARIANTS:
            for k, val in per[(v, d)].items():
                r[f'{k}_{v}'] = val
            r[f'transferred_{v}'] = float(comp.loc[(v, d), 'transferred_share'])
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / 'composition-by-dataset.csv', index=False)

    def lv(x):
        return '--' if x == 0 else str(x)
    # Every corpus retained every batch level (checked here), so one batch column suffices.
    for v in VARIANTS:
        has = t.batch_levels_source > 0
        assert (comp.loc[v].loc[t.dataset[has], 'batch_levels_retained'].to_numpy() == t.batch_levels_source[has].to_numpy()).all()
    head = [' & & & & & & \\multicolumn{6}{c}{Rare test types} & \\multicolumn{2}{c}{Transferred (\\%)}\\\\',
            '\\cmidrule(lr){7-12}\\cmidrule(l){13-14}',
            'Accession & Sp. & Cells & Batch & Sample identity & Types & Ref & Uni & Floor & Bal & 10k & MAD & Uni & 10k\\\\',
            '\\midrule']
    # A longtable so the table can start on the first supplementary page; the caption text is supplied by the
    # including document as \CompositionCaption.
    lines = ['% AUTO-GENERATED by experiments/e4-sampling-sensitivity/supplement_composition.py; do not edit.',
             '\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}',
             '\\begin{longtable}{@{}lcrrlr rrrrrr rr@{}}',
             '\\caption{\\CompositionCaption}\\label{tab:s-composition}\\\\', '\\toprule'] + head + ['\\endfirsthead',
             '\\multicolumn{14}{@{}l}{\\emph{Table S1, continued}}\\\\', '\\toprule'] + head + ['\\endhead',
             '\\bottomrule', '\\endlastfoot']
    for _, r in t.iterrows():
        def span(vals):
            vals = sorted(vals)
            return str(vals[0]) if vals[0] == vals[-1] else f'{vals[0]}--{vals[-1]}'
        sid = '--' if r.sample_identity == 'not recorded' else f'{r.sample_identity}, {r.sample_levels_reference}/{r.sample_levels_source}'
        tu = 100 * np.mean([r[f'transferred_uniform_s{i}'] for i in (1, 2, 3)])
        lines.append(' & '.join([r.accession, r.species, f'{r.source_cells:,}', lv(r.batch_levels_source), sid,
                                 span([r[f'test_types_{v}'] for v in VARIANTS]),
                                 str(r.rare_test_types_reference), span([r[f'rare_test_types_uniform_s{i}'] for i in (1, 2, 3)]),
                                 str(r.rare_test_types_type_floor), str(r.rare_test_types_batch_balanced),
                                 str(r.rare_test_types_cap10000), str(r.rare_test_types_mad_qc),
                                 f'{tu:.0f}', f'{100 * r.transferred_cap10000:.0f}']) + '\\\\')
    tot = {v: len(rare_sets[v]) for v in VARIANTS}
    lines += ['\\midrule',
              f'\\multicolumn{{6}}{{@{{}}l}}{{Rare types in the pooled test split}} & {tot["reference"]} & '
              f'{min(tot[f"uniform_s{i}"] for i in (1, 2, 3))}--{max(tot[f"uniform_s{i}"] for i in (1, 2, 3))} & '
              f'{tot["type_floor"]} & {tot["batch_balanced"]} & {tot["cap10000"]} & {tot["mad_qc"]} & & \\\\',
              '\\end{longtable}', '\\endgroup']
    (MS / 'figs' / 'tab_composition.tex').write_text('\n'.join(lines) + '\n')
    summary = {'rare_types_pooled': tot,
               'datasets_with_batch_field': int((t.batch_levels_source > 0).sum()),
               'datasets_with_sample_identity': int((t.sample_identity != 'not recorded').sum()),
               'datasets_with_donor_identity': 0,
               'batch_levels_fully_retained_reference': int(((t.batch_levels_source > 0) & (t.batch_levels_reference == t.batch_levels_source)).sum()),
               'sample_levels_fully_retained_reference': int(((t.sample_levels_source > 0) & (t.sample_levels_reference == t.sample_levels_source)).sum())}
    (HERE / 'composition-by-dataset-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
