"""Write pgfplots sources for the panels added in the consolidation round.

Every plotted value of a result panel is read from a saved experiment summary or group table; nothing is typed by
hand. The one exception is the worked example of Figure 1b, computed here from its stated means and dispersions. Each source follows the existing figure style:
Arial via the figure preamble, regular-weight titles, bold panel labels at
(-19pt, +12pt) from the plot area's top-left corner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / 'experiments'
FIGS = HERE.parent / 'manuscript' / 'figs'
AXIS = ('scale only axis=true, tick align=outside, tick pos=left, '
        'tick label style={font=\\footnotesize}, label style={font=\\footnotesize}, '
        'title style={font=\\footnotesize}')
LABEL = '\\panellabel{%s}'


def wrap(body):
    return ('\\begingroup\\sffamily\\mdseries\\sansmath\n'
            '\\begin{tikzpicture}[font=\\sffamily\\mdseries]\n' + body + '\\end{tikzpicture}\n\\endgroup\n')


def fmt(x):
    return f'{x:.4f}'.replace('-', '-')


def e2_panels():
    """Deficit-effect panels. Points are decile means of the raw group effects with dataset-cluster bootstrap
    intervals (descriptive). The line and band are the random-intercept model's prediction at the observed
    encoder and seed composition, using the same bootstrap resamples as the reported slope interval."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('e2analyze', EXP / 'e2-dose-response' / 'analyze.py')
    e2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2)
    design = json.loads((EXP / 'e2-dose-response' / 'design.json').read_text())
    g = pd.read_csv(EXP / 'e2-dose-response' / 'groups.csv')
    s = pd.read_csv(EXP / 'e2-dose-response' / 'summary.csv').set_index('analysis')
    ledger, panels = {}, {}
    brng = np.random.default_rng(20261113)
    for k, arm in enumerate(['A', 'B'], start=1):
        sub = g[g.arm == arm].reset_index(drop=True)
        sub['bin'] = pd.qcut(sub.deficit, 10, labels=False, duplicates='drop')
        nb = int(sub['bin'].max()) + 1
        b = sub.groupby('bin').agg(x=('deficit', 'mean'), y=('effect', 'mean'))
        where = {d: np.flatnonzero(sub.dataset.to_numpy() == d) for d in sub.dataset.unique()}
        names = list(where)
        boots = np.empty((2000, nb))
        for i in range(2000):
            idx = np.concatenate([where[d] for d in brng.choice(names, size=len(names))])
            part = sub.iloc[idx]
            boots[i] = part.groupby('bin').effect.mean().reindex(range(nb)).to_numpy()
        b['lo'], b['hi'] = np.nanquantile(boots, .025, axis=0), np.nanquantile(boots, .975, axis=0)
        stats, X, y = e2.sufficient_statistics(sub)
        beta, _, _ = e2.fit_random_intercept(stats)
        row = s.loc[f'Arm_{arm}']
        assert abs(beta[1] - row.slope) < 1e-9, 'model slope differs from the reported estimate'
        xbar = X.mean(0)
        x0, x1 = float(sub.deficit.quantile(.02)), float(sub.deficit.quantile(.98))
        grid = np.linspace(x0, x1, 41)
        pred = float(xbar @ beta) + beta[1] * (grid - xbar[1])
        rng = np.random.default_rng(design['inference']['seed'] + k)
        n_ds, curves, slopes = len(stats[0]), [], []
        for _ in range(design['inference']['resamples']):
            idx = rng.integers(n_ds, size=n_ds)
            bb, _, _ = e2.fit_random_intercept(tuple(a[idx] for a in stats))
            curves.append(float(xbar @ bb) + bb[1] * (grid - xbar[1]))
            slopes.append(bb[1])
        slo, shi = np.quantile(slopes, [.025, .975])
        assert abs(slo - row.ci_low) < 1e-9 and abs(shi - row.ci_high) < 1e-9, 'bootstrap does not reproduce the reported interval'
        band = np.quantile(np.array(curves), [.025, .975], axis=0)
        panels[arm] = (b, grid, pred, band, row)
        ledger[f'E2 Arm {arm}'] = {'bins': b.reset_index().round(6).to_dict('records'), 'slope': row.slope,
                                   'ci': [row.ci_low, row.ci_high], 'prediction_at_composition': True,
                                   'band_resamples': design['inference']['resamples']}
    ylo = min(min(p[0].lo.min(), p[3][0].min()) for p in panels.values()) - 0.015
    yhi = max(max(p[0].hi.max(), p[3][1].max()) for p in panels.values()) + 0.015
    xlo = min(p[0].x.min() for p in panels.values()) - 0.12
    xhi = max(p[0].x.max() for p in panels.values()) + 0.12
    body = ('\\begin{groupplot}[' + AXIS + ', group style={group size=2 by 1, horizontal sep=1.7cm}, '
            'width=6.05cm, height=4.6cm, xlabel={validation variance deficit (log$_2$ units)}, '
            f'ylabel={{NB minus Poisson calibration error}}, ymin={ylo:.3f}, ymax={yhi:.3f}, xmin={xlo:.3f}, xmax={xhi:.3f}]\n')
    for letter, arm, colour in [('(a)', 'A', 'black!70'), ('(b)', 'B', 'hviolet')]:
        b, grid, pred, band, row = panels[arm]
        title = 'Arm A: encoder sees target genes' if arm == 'A' else 'Arm B: target genes held out'
        body += f'\\nextgroupplot[title={{{title}}}]\n'
        body += f'\\addplot[hgray, dashed, domain={xlo:.3f}:{xhi:.3f}, forget plot] {{0}};\n'
        poly = [f'({fmt(x)},{fmt(v)})' for x, v in zip(grid, band[1])] + [f'({fmt(x)},{fmt(v)})' for x, v in zip(grid[::-1], band[0][::-1])]
        body += f'\\addplot[draw=none, fill={colour}, fill opacity=0.18, forget plot] coordinates {{' + ' '.join(poly) + '} -- cycle;\n'
        body += (f'\\addplot[{colour}, thick] coordinates {{' + ' '.join(f'({fmt(x)},{fmt(v)})' for x, v in zip(grid, pred)) + '};\n')
        body += (f'\\addplot[{colour}, only marks, mark=*, mark size=1.7pt, error bars/.cd, y dir=both, y explicit] '
                 'coordinates {\n' + '\n'.join(f'  ({fmt(r.x)},{fmt(r.y)}) -= (0,{fmt(r.y - r.lo)}) += (0,{fmt(r.hi - r.y)})'
                                               for r in b.itertuples()) + '\n};\n')
        body += (f'\\node[font=\\footnotesize, anchor=south west, align=left] at (rel axis cs:0.03,0.03) '
                 f'{{slope ${row.slope:.3f}$\\\\95\\% interval $[{row.ci_low:.3f}, {row.ci_high:.3f}]$}};\n')
        body += LABEL % letter + '\n'
    body += '\\end{groupplot}\n'
    (FIGS / 'fig_e2dose.tex').write_text(wrap(body))
    return ledger


def e1_panel():
    s = pd.read_csv(EXP / 'e1-armb-biology' / 'summary.csv')
    order = [('welch', 'Welch recall'), ('auroc', 'AUROC recall'),
             ('calibration', 'calibration'), ('macro_f1', 'macro-F1'), ('rare_f1_full', 'rare-type F1')]
    s = s[(s.a == 'per_cell') & (s.b == 'poisson')]
    sel = s[(s.a == 'per_cell') & (s.b == 'poisson') & s.metric.isin([m for m, _ in order])].copy()
    flip = sel.metric == 'calibration'
    lows = (-sel.ci_high).where(flip, sel.ci_low); highs = (-sel.ci_low).where(flip, sel.ci_high)
    xlo, xhi = float(lows.min()) - 0.01, float(highs.max()) + 0.01
    body = ('\\begin{axis}[' + AXIS + f', width=4.0cm, height=5.0cm, xmin={xlo:.3f}, xmax={xhi:.3f}, '
            'ymin=0.4, ymax=5.6, ytick={1,2,3,4,5}, yticklabels={' + ','.join('{' + t + '}' for _, t in reversed(order)) + '}, '
            'y dir=normal, xlabel={per-cell advantage over Poisson}, title={Matched design, both arms}, '
            'xtick={-0.12,-0.06,0,0.06}, xticklabels={$-0.12$,$-0.06$,0,0.06}, scaled x ticks=false, '
            'legend style={font=\\footnotesize, draw=none, at={(0.03,0.97)}, anchor=north west, legend columns=1, fill=white, fill opacity=0.85, text opacity=1, /tikz/every even column/.append style={column sep=6pt}}]\n'
            '\\addplot[hgray, dashed, forget plot] coordinates {(0,0.4) (0,5.6)};\n')
    ledger = {}
    for arm, colour, mark, off in [('A', 'black!70', '*', 0.15), ('B', 'hviolet', 'square*', -0.15)]:
        pts = []
        for k, (metric, _) in enumerate(reversed(order), 1):
            r = s[(s.arm == arm) & (s.metric == metric)].iloc[0]
            e, lo, hi = r.effect, r.ci_low, r.ci_high
            if metric == 'calibration':
                e, lo, hi = -e, -hi, -lo
            pts.append(f'  ({fmt(e)},{k + off:.2f}) -= ({fmt(e - lo)},0) += ({fmt(hi - e)},0)')
            ledger[f'E1 Arm {arm} {metric}'] = [e, lo, hi]
        body += (f'\\addplot[{colour}, only marks, mark={mark}, mark size=1.8pt, error bars/.cd, x dir=both, x explicit] '
                 'coordinates {\n' + '\n'.join(pts) + '\n};\n')
        body += f'\\addlegendentry{{Arm {arm}}}\n'
    body += LABEL % '(b)' + '\n\\end{axis}\n'
    (FIGS / 'fig_e1matched.tex').write_text(wrap(body))
    return ledger


def e4_panels():
    """Corpus sensitivity as small multiples with direct row labels: decoder contrasts (rows 1-2), scVI
    contrasts on the same corpora and the fixed-evaluation training-size experiment (row 3)."""
    path = EXP / 'e4-sampling-sensitivity' / 'summary.csv'
    if not path.exists():
        return {}
    s = pd.read_csv(path)
    sv_path = EXP / 'e4-scvi-sensitivity' / 'summary.csv'
    sv = pd.read_csv(sv_path) if sv_path.exists() else None
    tb_path = EXP / 'e4b-training-size' / 'summary.csv'
    tb = pd.read_csv(tb_path) if tb_path.exists() else None
    names = {'reference': 'reference', 'uniform_s1': 'uniform, seed 1', 'uniform_s2': 'uniform, seed 2',
             'uniform_s3': 'uniform, seed 3', 'type_floor': 'cell-type stratified', 'batch_balanced': 'batch balanced',
             'mad_qc': 'outlier QC', 'cap10000': '10,000-cell cap'}
    variants = list(names)
    ypos = {v: len(variants) - i for i, v in enumerate(variants)}
    panels = [
        ('(a)', 'Arm A: per-cell $-$ Poisson', 'calibration error', lambda v: s[(s.variant == v) & (s.contrast == 'Arm A per-cell minus Poisson calibration error')]),
        ('(b)', 'Arm B: per-cell $-$ Poisson', 'calibration error', lambda v: s[(s.variant == v) & (s.contrast == 'Arm B per-cell minus Poisson calibration error')]),
        ('(c)', 'Arm B: per-cell $-$ shared', 'calibration error', lambda v: s[(s.variant == v) & (s.contrast == 'Arm B per-cell minus shared calibration error')]),
        ('(d)', 'Arm B: per-cell $-$ Poisson', 'Welch recall', lambda v: s[(s.variant == v) & (s.contrast == 'Arm B per-cell minus Poisson Welch recall')]),
        ('(e)', 'Arm B: per-cell $-$ shared', 'Welch recall', lambda v: s[(s.variant == v) & (s.contrast == 'Arm B per-cell minus shared Welch recall')]),
        ('(f)', 'Deficit--effect slope', 'slope (both arms)', lambda v: s[(s.variant == v) & (s.contrast == 'Deficit-effect slope (pooled arms)')]),
    ]
    if sv is not None and len(sv) == 2 * len(variants):
        panels += [
            ('(g)', 'scVI: shared $-$ Poisson', 'calibration error', lambda v: sv[(sv.variant == v) & (sv['mode'] == 'gene')]),
            ('(h)', 'scVI: per-cell $-$ Poisson', 'calibration error', lambda v: sv[(sv.variant == v) & (sv['mode'] == 'gene-cell')]),
        ]
    ledger = {}
    body = ('\\begin{groupplot}[' + AXIS + ', group style={group size=3 by 3, horizontal sep=0.55cm, vertical sep=1.8cm}, '
            'width=3.55cm, height=2.75cm, ymin=0.4, ymax=8.6, ytick={8,7,6,5,4,3,2,1}, yticklabels={}, '
            'scaled x ticks=false, x tick label style={/pgf/number format/fixed, /pgf/number format/precision=3}, '
            'xticklabel style={font=\\scriptsize}]\n')
    for i, (letter, title, xlabel, get) in enumerate(panels):
        rows = [(v, get(v).iloc[0]) for v in variants]
        lo = min(r.ci_low for _, r in rows); hi = max(r.ci_high for _, r in rows)
        pad = 0.1 * (hi - lo)
        lo, hi = min(lo - pad, -pad), max(hi + pad, pad)
        ref_class = rows[0][1].classification
        opts = f'title={{{title}}}, xlabel={{{xlabel}}}, xmin={lo:.4f}, xmax={hi:.4f}'
        if i % 3 == 0:
            opts += ', yticklabels={' + ','.join('{' + names[v] + '}' for v in variants) + '}, y tick label style={font=\\scriptsize}'
        body += f'\\nextgroupplot[{opts}]\n'
        body += '\\addplot[hgray, dashed, forget plot] coordinates {(0,0.4) (0,8.6)};\n'
        for v, r in rows:
            stable = r.classification == ref_class
            colour = 'black'  # a changed classification is shown by an open marker
            mark = 'diamond*' if v == 'reference' else ('*' if stable else 'o')
            size = 2.3 if v == 'reference' else 1.5
            body += (f'\\addplot[{colour}, only marks, mark={mark}, mark size={size}pt, error bars/.cd, x dir=both, x explicit] '
                     f'coordinates {{({fmt(r.effect)},{ypos[v]}) -= ({fmt(r.effect - r.ci_low)},0) += ({fmt(r.ci_high - r.effect)},0)}};\n')
            ledger[f'E4 {letter} {v}'] = [r.effect, r.ci_low, r.ci_high, r.classification]
        body += LABEL % letter + '\n'
    if tb is not None and {'x1', 'x2', 'x3'} <= set(tb['size']):
        sizes = [('x1', 3), ('x2', 2), ('x3', 1)]
        series = [('Arm B per-cell minus Poisson calibration error', 'black', '*', 0.14, 'calibration'),
                  ('Arm B per-cell minus Poisson Welch recall', 'hviolet', 'square*', -0.14, 'Welch recall')]
        sub = tb[tb['size'].isin(['x1', 'x2', 'x3']) & tb.contrast.isin([c for c, *_ in series])]
        refs = s[(s.variant == 'reference') & s.contrast.isin([c for c, *_ in series])]
        lo, hi = min(sub.ci_low.min(), refs.ci_low.min()), max(sub.ci_high.max(), refs.ci_high.max())
        pad = 0.1 * (hi - lo)
        lo, hi = min(lo - pad, -pad), max(hi + pad, pad)
        labels = {k: f'{int(tb[tb["size"] == k].n_train.iloc[0]):,}' for k, _ in sizes}
        body += (f'\\nextgroupplot[title={{Arm B: training size}}, xlabel={{per-cell $-$ Poisson}}, xmin={lo:.4f}, xmax={hi:.4f}, '
                 'ymin=0.4, ymax=3.6, ytick={3,2,1}, yticklabels={' + ','.join('{' + labels[k] + '}' for k, _ in sizes) + '}, '
                 'yticklabel pos=right, ylabel={training cells}, ylabel near ticks, ylabel style={font=\\scriptsize}, '
                 'y tick label style={font=\\scriptsize}]\n')  # marker key in the caption: no legend over the data
        body += '\\addplot[hgray, dashed, forget plot] coordinates {(0,0.4) (0,3.6)};\n'
        for contrast, colour, mark, off, lab in series:
            pts = []
            for k, y in sizes:
                r = tb[(tb['size'] == k) & (tb.contrast == contrast)].iloc[0]
                if k == 'x1':
                    # The original training set is the reference corpus: plot its interval from panels (b) and (d).
                    ref = s[(s.variant == 'reference') & (s.contrast == contrast)].iloc[0]
                    assert abs(ref.effect - r.effect) < 1e-9
                    r = ref
                pts.append(f'({fmt(r.effect)},{y + off:.2f}) -= ({fmt(r.effect - r.ci_low)},0) += ({fmt(r.ci_high - r.effect)},0)')
                ledger[f'E4b {k} | {contrast}'] = [r.effect, r.ci_low, r.ci_high]
            body += (f'\\addplot[{colour}, only marks, mark={mark}, mark size=1.6pt, error bars/.cd, x dir=both, x explicit] '
                     'coordinates {' + ' '.join(pts) + '};\n')
        body += LABEL % '(i)' + '\n'
    body += '\\end{groupplot}\n'
    (FIGS / 'fig_e4sampling.tex').write_text(wrap(body))
    return ledger


def e5_panel():
    """Ten-seed end-to-end differences from Poisson: seed outcomes, their mean and a seed-bootstrap 95% interval."""
    per = EXP / 'e5-endtoend-seeds' / 'per-seed.csv'
    summ = EXP / 'e5-endtoend-seeds' / 'summary.csv'
    if not per.exists():
        return {}
    d = pd.read_csv(per)
    s = pd.read_csv(summ).set_index(['endpoint', 'quantity'])
    endpoints = [('marker_recall_mean', 'marker\\\\recall'), ('tstr_macro_f1', 'TSTR\\\\macro-F1'), ('tstr_rare_f1', 'rare-type\\\\F1')]
    contrasts = [('shared', 'horange', -0.17), ('per_cell', 'hgreen', 0.17)]
    vals = []
    for ep, _ in endpoints:
        w = d.pivot(index='seed', columns='arm', values=ep)
        for arm, _, _ in contrasts:
            vals += list(w[arm] - w['poisson'])
            r = s.loc[(ep, f'{arm} minus poisson')]
            vals += [r.ci_low, r.ci_high]
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo)
    n = d.seed.nunique()
    body = ('\\begin{axis}[' + AXIS + f', width=4.0cm, height=4.6cm, xmin=0.5, xmax=3.5, ymin={lo - pad:.3f}, ymax={hi + pad:.3f}, '
            'xtick={1,2,3}, xticklabels={' + ','.join('{' + t + '}' for _, t in endpoints) + '}, '
            'x tick label style={align=center}, ylabel={difference from Poisson}, '
            'scaled y ticks=false, y tick label style={/pgf/number format/fixed, /pgf/number format/precision=2}, '
            f'title={{End to end, {n} seeds}}]\n'
            '\\addplot[hgray, dashed, forget plot] coordinates {(0.5,0) (3.5,0)};\n')
    ledger = {}
    for k, (ep, _) in enumerate(endpoints, start=1):
        w = d.pivot(index='seed', columns='arm', values=ep)
        for arm, colour, off in contrasts:
            diff = (w[arm] - w['poisson']).to_numpy()
            jit = np.linspace(-0.07, 0.07, len(diff))
            body += (f'\\addplot[only marks, mark=o, mark size=1.0pt, draw={colour}!80!black, forget plot] coordinates {{'
                     + ' '.join(f'({k + off + j:.3f},{fmt(v)})' for j, v in zip(jit, diff)) + '};\n')
            r = s.loc[(ep, f'{arm} minus poisson')]
            legend = ', forget plot'  # colours follow the figure's shared key
            body += (f'\\addplot[only marks, mark=diamond*, mark size=2.2pt, color={colour}!85!black{legend}, error bars/.cd, '
                     f'y dir=both, y explicit] coordinates {{({k + off + 0.13:.3f},{fmt(r["mean"])}) -= (0,{fmt(r["mean"] - r.ci_low)}) '
                     f'+= (0,{fmt(r.ci_high - r["mean"])})}};\n')
            ledger[f'E5 {ep} {arm} minus poisson'] = {'seeds': diff.round(6).tolist(), 'mean': r['mean'], 'ci': [r.ci_low, r.ci_high]}
    body += LABEL % '(a)' + '\n\\end{axis}\n'
    (FIGS / 'fig_e5endtoend.tex').write_text(wrap(body))
    return ledger


def overview_panel():
    """Figure 1: study overview (a) and a worked example with known means (b). The example values follow from the
    negative-binomial variance mu + mu^2/theta; they illustrate misallocation and are not data."""
    mu, th = 4.0, {'type 1': 0.5, 'type 2': 8.0}
    real = {t: mu + mu ** 2 / v for t, v in th.items()}
    shared_var = float(np.mean(list(real.values())))
    ratios = {'Poisson': {t: mu / real[t] for t in th},
              'shared per-gene': {t: shared_var / real[t] for t in th},
              'per-cell-type': {t: 1.0 for t in th}}
    # (a) Roadmap: the question, what each preliminary analysis found, the main analysis and the checks it implies.
    grey = lambda t: '\\textcolor{hgray!60!black}{' + t + '}'
    box = lambda at, anchor, name, width, head, body: (
        f'\\node[road, anchor={anchor}, text width={width}] ({name}) at ({at}) {{{head}\\\\[1pt]{grey(body)}}};\n')
    body = ('\\tikzset{road/.style={rectangle, draw=hgray!70!black, line width=0.6pt, fill=white, align=left, '
            'inner sep=3.5pt, font=\\footnotesize, execute at begin node={\\hyphenpenalty=10000\\exhyphenpenalty=10000}}, '
            'arr/.style={-{Stealth[length=1.8mm]}, line width=0.6pt, draw=hgray!50!black}}\n')
    body += box('0,0', 'north west', 'f1', '4.55cm', 'F1\\ \\ Tightness ratio',
                'rewards random projections and added noise, so it cannot select an encoder')
    body += box('[yshift=-0.16cm]f1.south west', 'north west', 'f2', '4.55cm', 'F2\\ \\ Held-out-gene reconstruction',
                'encoders differ little and their ranking is unresolved')
    body += box('[yshift=-0.16cm]f2.south west', 'north west', 'f3', '4.55cm', 'F3\\ \\ Expressed-gene variance',
                'deterministic decoders lose variance, which all-gene ratios hide')
    body += box('[xshift=-0.5cm]f2.west', 'east', 'q', '2.75cm', 'Question',
                'when does adding dispersion help a count decoder, and on which readouts?')
    body += box('[xshift=0.55cm]f2.east', 'west', 'f4', '3.3cm', 'F4\\ \\ Dispersion at a fixed mean and end to end',
                'a measured deficit predicts the calibration effect; allocation matters; marker recovery does not '
                'follow calibration')
    body += box('[xshift=0.5cm]f4.east', 'west', 'app', '2.9cm', 'Checks for analysts',
                '1.\\ measure the deficit on held-out cells\\\\2.\\ compare allocations on validation cells\\\\'
                '3.\\ score each readout on test cells')
    for f in ['f1', 'f2', 'f3']:
        body += f'\\draw[arr] (q.east) -- ({f}.west);\n\\draw[arr] ({f}.east) -- (f4.west);\n'
    body += '\\draw[arr] (f4.east) -- (app.west);\n'
    body += '\\node[inner sep=0pt, minimum size=0pt] (ptl) at ([xshift=0.95cm]q.west |- f1.north) {};\n\\panellabelat{ptl}{(a)}\n'
    body += '\\node[inner sep=0pt, minimum size=0pt] (pbl) at (q.west |- f3.south) {};\n'
    colours = {'Poisson': ('hblue', 'square*', -0.2), 'shared per-gene': ('horange', 'diamond*', 0.0),
               'per-cell-type': ('hgreen', '*', 0.2)}
    # (b) Worked example below the roadmap; the legend sits to the right of the plot area.
    body += ('\\begin{axis}[' + AXIS + ', at={(pbl)}, anchor=north west, xshift=0.95cm, yshift=-1.35cm, '
             'width=7.6cm, height=3.0cm, '
             'xmin=0.5, xmax=2.5, xtick={1,2}, '
             'xticklabels={{type 1 (true variance %.0f)},{type 2 (true variance %.0f)}}, '
             'ymin=-3.6, ymax=2.3, ylabel={$\\log_2$ decoded / real variance}, title={Worked example, known means}, '
             'legend style={font=\\footnotesize, draw=none, fill=none, at={(1.04,0.5)}, anchor=west}, '
             'legend cell align=left]\n' % (real['type 1'], real['type 2']))
    body += '\\draw[hgray, dashed] (axis cs:0.5,0) -- (axis cs:2.5,0);\n'
    ledger = {}
    for name, vals in ratios.items():
        colour, mark, off = colours[name]
        pts = ' '.join(f'({k + 1 + off:.2f},{fmt(float(np.log2(v)))})' for k, v in enumerate(vals.values()))
        body += (f'\\addplot[ycomb, color={colour}, line width=1.2pt, mark={mark}, mark size=2.4pt] coordinates {{{pts}}};\n'
                 f'\\addlegendentry{{{name}}}\n')
        ledger[f'overview {name}'] = {t: float(np.log2(v)) for t, v in vals.items()}
    body += LABEL % '(b)' + '\n\\end{axis}\n'
    (FIGS / 'fig_overview.tex').write_text(wrap(body))
    return ledger


def main():
    ledger = {}
    ledger.update(e2_panels())
    ledger.update(e1_panel())
    ledger.update(e4_panels())
    ledger.update(e5_panel())
    ledger.update(overview_panel())
    (HERE / 'new-panel-ledger.json').write_text(json.dumps(ledger, indent=1, default=float) + '\n')
    print('panels written:', sorted({k.split()[0] for k in ledger}))


if __name__ == '__main__':
    main()
