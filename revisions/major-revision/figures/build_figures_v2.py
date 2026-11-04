"""Compose the revision's 11 main and 2 supplementary figures from their plot sources.

Every block is an existing figure source copied with layout-only edits: panel re-lettering, plot
widths and heights, label wording and legend placement. Plotted coordinates are never edited; the
coordinate audit checks that every numeric coordinate block of every source survives verbatim in the
composed sources.

Alignment model. Each plot block is trimmed horizontally to its plot area (pgfplots "trim axis"),
so a block placed at x has its first plot area starting exactly at x. Panel labels come from one
macro (\\panellabel in manuscript/figure-style.tex), so their offset from the plot corner is
identical everywhere. After composition, the alignment audit reads the axis frames from the output
PDF and fails the build when rows that should share column edges do not.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import fitz

R = Path(__file__).resolve().parent
WORK = R.parent / 'manuscript'
FIGS = WORK / 'figs'
BUILD = R / 'build-v2'
BLOCKS = BUILD / 'blocks'

# Colours and number macros shared with the article (figure-colours.tex is extracted from its preamble).
COLORS = (WORK / 'figure-colours.tex').read_text().strip()
NUMBERS = '\n'.join(f'\\input{{figs/{p.name}}}' for p in sorted(FIGS.glob('*numbers.tex')))

# One vocabulary for dispersion choices in every figure (most specific strings first).
NAMES = [
    ('addlegendentry{NB global $\\theta_g$} \\addplot[draw=black!30, fill=hgreen', 'addlegendentry{Shared per-gene} \\addplot[draw=black!30, fill=hgreen'),
    ('NB global $\\theta_g$', 'Shared per-gene'),
    ('NB conditional $\\theta_{ng}$', 'Per-cell'),
    ('amortised $\\theta_{ng}$', 'Per-cell'),
    ('{global $\\theta_g$}', '{shared $\\theta_g$}'),
    ('{conditional $\\theta_{ng}$}', '{per-cell $\\theta_{ng}$}'),
    ('~global\\quad', '~shared per-gene\\quad'),
    ('~conditional}', '~per-cell}'),
    ('Oracle conditional', 'True per-group'),
    ('oracle conditional\\\\lowest error in deficit regimes', 'true per-group value\\\\lowest error with a deficit'),
    ('Estimated pooled', 'Estimated shared'),
    ('Estimated conditional', 'Estimated per-group'),
    ('Global (true median)', 'Shared (true median)'),
    ('addlegendentry{Global}', 'addlegendentry{Shared per-gene}'),
    ('addlegendentry{Conditional}', 'addlegendentry{Per-cell}'),
    ('Fixed: global', 'Fixed: shared'),
    ('End-to-end: global', 'End-to-end: shared'),
    ('addlegendentry{Shared}', 'addlegendentry{Shared per-gene}'),
    ('addlegendentry{Per type}', 'addlegendentry{Per-cell-type}'),
    ('addlegendentry{Per cell}', 'addlegendentry{Per-cell}'),
]

LABEL_RE = re.compile(r'\\node\[anchor=base west,font=\\(?:footnotesize|small)\\bfseries,xshift=-19pt,yshift=12pt\] '
                      r'at \(rel axis cs:0,1\) \{(\([a-z]\))\};')


def preamble(border):
    return (r'''\documentclass[border={%s}]{standalone}
\usepackage{amsmath,amssymb,graphicx,xcolor,tikz,pgfplots}
\input{figure-fonts.tex}
\input{figure-style.tex}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{positioning,arrows.meta,calc,fit}
''' % border) + COLORS + '\n' + NUMBERS + '\n'


def B(name, src, letters=None, subs=(), trim='group'):
    return ('spec', name, src, letters, tuple(subs), trim)


def block(name, src, letters=None, subs=(), trim='group'):
    """Copy a figure source with layout-only edits and return its block path."""
    text = (FIGS / f'{src}.tex').read_text()
    if letters:
        for old, new in letters.items():
            text = text.replace('{(%s)}' % old, '{(@%s@)}' % new)
        text = text.replace('(@', '(').replace('@)', ')')
    for old, new in subs:
        assert old in text, f'{src}: substitution target missing: {old!r}'
        text = text.replace(old, new, 1)
    for old, new in NAMES:
        text = text.replace(old, new)
    text = LABEL_RE.sub(lambda m: '\\panellabel{%s}' % m.group(1), text)
    opt = {'group': 'trim axis group left, trim axis group right',
           'axis': 'trim axis left, trim axis right', None: None}[trim]
    if opt:
        # Works whether or not the source's picture already has options.
        text, n = re.subn(r'\\begin\{tikzpicture\}(\[)?', lambda m: '\\begin{tikzpicture}[' + opt + (', ' if m.group(1) else ']'), text, count=1)
        assert n == 1, f'{src}: no tikzpicture to trim'
    BLOCKS.mkdir(parents=True, exist_ok=True)
    (BLOCKS / f'{name}.tex').write_text(text)
    return name


def LABEL(letter):
    """Substitution adding a panel label to a single-axis source that had none."""
    return ('\\end{axis}\\end{tikzpicture}', '\\panellabel{(%s)}\n\\end{axis}\\end{tikzpicture}' % letter)


def two(width_old, sep, span):
    """Substitution giving a two-panel row the plot span `span` (cm)."""
    return (f'width={width_old}cm', f'width={(span - sep) / 2:.3f}cm')


def H(old, new):
    return (f'height={old}cm', f'height={new}cm')


def SEP(old, new):
    return (f'horizontal sep={old}cm', f'horizontal sep={new}cm')


def broken_forest():
    """Figure 5c as a broken axis: the full first-dataset interval is shown in a short upper axis.

    The upper and lower axes draw the same source body verbatim; only ranges and frames differ.
    """
    s = (FIGS / 'fig_forest.tex').read_text()
    a = s.index('\\begin{axis}[') + len('\\begin{axis}[')
    depth, i = 1, a
    while depth:
        depth += {'[': 1, ']': -1}.get(s[i], 0)
        i += 1
    opts, body_end = s[a:i - 1], s.index('\\end{axis}')
    body = s[i:body_end]
    note = re.search(r'\\node\[font=\\footnotesize, anchor=north west, align=left\] at \(rel axis cs:0\.01,0\.99\)\{[^}]*\};', body)
    assert note, 'forest annotation not found'
    body_hi = body.replace(note.group(0), '')
    body_lo = body.replace(note.group(0), '')  # the caption states what the band and points are
    lo = (opts.replace('width=12.6cm', 'width=12.8cm').replace('height=6.0cm', 'height=2.9cm')
          .replace('ymin=-0.0578, ymax=0.2070', 'ymin=-0.0578, ymax=0.112, clip mode=global')
          .replace('ytick={-0.05,0,0.05,0.10,0.15,0.20}', 'ytick={-0.05,0,0.05,0.10}'))
    lo = re.sub(r'ylabel=\{[^}]*\}', lambda m: 'ylabel={Median difference\\\\(NB $-$ Poisson)}, ylabel style={align=center}', lo)
    hi = (opts.replace('width=12.6cm', 'width=12.8cm').replace('height=6.0cm', 'height=0.85cm')
          .replace('ymin=-0.0578, ymax=0.2070', 'ymin=0.125, ymax=0.21, clip mode=global')
          .replace('ytick={-0.05,0,0.05,0.10,0.15,0.20}', 'ytick={0.15,0.20}'))
    hi = re.sub(r'xlabel=\{[^}]*\}', lambda m: 'xlabel={}', hi)
    hi = re.sub(r'ylabel=\{[^}]*\}', lambda m: 'ylabel={}', hi)
    hi += ', xticklabels={}, name=forhi, at={(forlo.north west)}, anchor=south west, yshift=4pt'
    brk = ('\\foreach \\c in {forlo.north west, forlo.north east}{\\draw[black, line width=.5pt] '
           '($(\\c)+(-2.2pt,-1.6pt)$)--($(\\c)+(2.2pt,1.6pt)$);}\n'
           '\\foreach \\c in {forhi.south west, forhi.south east}{\\draw[black, line width=.5pt] '
           '($(\\c)+(-2.2pt,-1.6pt)$)--($(\\c)+(2.2pt,1.6pt)$);}\n')
    out = (s[:s.index('\\begin{axis}[')] + '\\begin{axis}[name=forlo, ' + lo + ']' + body_lo + '\\end{axis}\n'
           + '\\begin{axis}[' + hi + ']' + body_hi + '\\end{axis}\n' + brk
           + '\\panellabelat{forhi}{(c)}\n' + s[body_end + len('\\end{axis}'):])
    (FIGS / 'fig_forest_broken.tex').write_text(out)


# Final-width design: the text block is 14.6 cm; spans plus label overhang fill about 14.4 cm and
# every composed PDF is cropped to its ink.
FIGURES = {
    'Figure1': ('4pt', [[(B('f1', 'fig_overview', trim=None), 0, None)]]),
    'Figure2': ('4pt', [[(B('f2', 'fig_design', trim=None, subs=[
        ('{Encoder--decoder geometry}', '{Experimental arms}'),
        ('{Mean-fixed ablation}', '{Fixed-mean comparison}'),
        ('\\node[note, font=\\footnotesize] at (3.85,-1.05) {near-invertible: input $=$ target};\n', ''),
        ('(mu) at (10.25,-1.15)', '(mu) at (10.25,-1.25)'),
        ('(pois) at (13.15,-0.25)', '(pois) at (13.15,0.0)'),
        ('(glob) at (13.15,-1.15)', '(glob) at (13.15,-0.83)'),
        ('(cond) at (13.15,-2.05) {conditional $\\theta_{ng}$};',
         '(cond) at (13.15,-2.49) {conditional $\\theta_{ng}$};\n'
         '\\node[rectangle, rounded corners=2pt, align=center, draw=holive, line width=0.65pt, fill=holive!12, '
         'text width=2.20cm, minimum height=0.78cm, inner sep=2.5pt, font=\\footnotesize, outer sep=1.2pt] '
         '(ptype) at (13.15,-1.66) {per-cell-type $\\theta_{tg}$};'),
        ('\\coordinate (bus) at ($(mu.east)!0.42!(glob.west)$);', '\\coordinate (bus) at ($(mu.east)!0.42!(mu.east -| glob.west)$);'),
        ('\\draw[arr] (bus) -- (glob.west);', '\\draw[arr] (bus) |- (glob.west);\n\\draw[arr] (bus) |- (ptype.west);'),
        ('\\node[note] at ($(bus)!0.5!(glob.west)+(0,-1.45)$) {only dispersion varies};', '\\node[note] at (12.05,-3.25) {only dispersion varies};'),
        ('{\\color{hgreen}\\rule{1.6ex}{1.6ex}}~conditional}', '{\\color{holive}\\rule{1.6ex}{1.6ex}}~per-cell-type\\quad\n  {\\color{hgreen}\\rule{1.6ex}{1.6ex}}~conditional}'),
    ]), 0, None)]]),
    'Figure3': ('60pt', [
        [(B('f3ab', 'fig_tightness', subs=[two(6.1, 1.6, 12.8), SEP(1.5, 1.6), H(5.4, 4.0),
            ]), 0, None)],
        [(B('f3cd', 'fig_trap', {'a': 'c', 'b': 'd'}, subs=[two(6.2, 1.6, 12.8), SEP(1.7, 1.6), H(5.4, 4.0),
            ('\\node[font=\\footnotesize, anchor=north] at (rel axis cs:0.5,0.99){higher scores with added noise};\n', ''),
            ('\\node[font=\\footnotesize, anchor=north] at (rel axis cs:0.5,0.99){matched noise does not minimize distance};\n', ''),
            ('decoded tightness ratio (higher looks better)', 'decoded tightness ratio'),
            ('\\draw[hgray, dashed, thin] (axis cs:0.887,0.5) -- (axis cs:3.812,0.5);\n', ''),  # unexplained guide line
            ('energy distance (lower is better)', 'energy distance')]), 0, None)]]),
    'Figure4': ('80pt', [
        [(B('f4ab', 'fig_encoder', subs=[two(5.7, 2.5, 12.3), H(5.5, 4.0),
            ('ytick={randproj-50,scGPT-512,randproj-512,PCA-512,oracle 1250-d,PCA-50,Geneformer},',
             'ytick={randproj-50,scGPT-512,randproj-512,PCA-512,oracle 1250-d,PCA-50,Geneformer}, '
             'yticklabels={randproj-50,scGPT-512,randproj-512,PCA-512,identity 1250-d,PCA-50,Geneformer},'),
            ('yticklabels={{ridge\\\\{closed form}},{oracle MLP\\\\{5.9M}},{big MLP\\\\{21.9M}}}',
             'yticklabels={{ridge\\\\{closed form}},{standard MLP\\\\{5.9M}},{large MLP\\\\{21.9M}}}'),
            ('\\node[font=\\footnotesize, black, anchor=west] at (axis cs:0.649,randproj-50){falls back};', '')]), 0, None)],
        [(B('f4cde', 'fig_variance', {'a': 'c', 'b': 'd', 'c': 'e'}, subs=[H(5.3, 4.0),
            ('\\nextgroupplot[title={expressed-gene signed', '\\nextgroupplot[width=4.4cm, title={expressed-gene signed'),
            ('\\nextgroupplot[title={All genes versus expressed genes}', '\\nextgroupplot[width=2.6cm, title={All vs expressed}'),
            ('xticklabels={PCA-512 matched, scGPT matched}', 'xticklabels={PCA-512, scGPT}'),
            ('enlarge x limits=0.52', 'enlarge x limits=0.34'),
            ('\\node[font=\\footnotesize, anchor=north west] at (rel axis cs:0.01,0.99){Gaussian: underdispersed}; \\node[font=\\footnotesize, anchor=north east] at (rel axis cs:0.99,0.99){};', ''),
            ('ylabel={var(gen)/var(real)}', 'ylabel={signed $r_{\\mathrm{var}}$}'),
            ('(axis cs:2,50)--(axis cs:2,90)', '(axis cs:2,65)--(axis cs:2,82)'),
            ('/pgf/bar width=20pt', '/pgf/bar width=12pt'),
            ('at={(3.6cm,-2.5cm)},anchor=north west,width=6.1cm,height=3.3cm', 'at={(9.95cm,0cm)},anchor=south west,width=2.35cm,height=4.0cm'),
            ('ymin=50,ymax=90,ytick={50,60,70,80,90}', 'ymin=65,ymax=82,ytick={65,70,75,80}'),
            ('title={Expression-threshold sensitivity}', 'title={Threshold}'),
            ('\\node[font=\\footnotesize,anchor=north west] at (axis cs:2.05,89) {primary cut};',
             '\\node[font=\\footnotesize,anchor=north west] at (axis cs:2.05,81.5) {primary};')]), 0, None)]]),
    'Figure5': ('60pt', [
        [(B('f5ab', 'fig_theta', subs=[two(6.1, 1.5, 12.8), SEP(1.7, 1.5), H(5.3, 3.9),
            ('\\node[font=\\footnotesize, black, anchor=north] at (axis cs:35,2.12){Poisson-like plateau};', '')]), 0, None)],
        [(B('f5c', 'fig_forest_broken', trim='axis'), 0, None)]]),
    'Figure6': ('60pt', [
        [(B('f6ab', 'fig_e2dose', subs=[two(6.05, 1.55, 12.8), SEP(1.7, 1.55), H(4.6, 3.4)]), 0, None)],
        [(B('f6cd', 'fig_scvimodes', {'a': 'c', 'b': 'd'}, subs=[two(5.6, 1.55, 12.8), SEP(1.7, 1.55), H(5.4, 3.6),
            ('at={(0.5,1.19)}', 'at={(1.138,1.19)}'),
            (', legend style={font=\\footnotesize, at={(0.5,1.19)}, anchor=south, legend columns=2, draw=none}', ''),
            ('\\addlegendentry{global $\\theta_g$}', ''), ('\\addlegendentry{amortised $\\theta_{ng}$}', ''),
            ('title={Dispersion at a fixed mean}', 'title={Frozen scVI mean}'),
            ('xticklabels={{$|\\log_2 r_{\\mathrm{var}}|$\\\\(expr)},{energy\\\\distance}}', 'xticklabels={{calibration\\\\error},{energy\\\\distance}}')]), 0, None)]]),
    'Figure7': ('60pt', [
        [(B('f7ab', 'fig_lawgrid', subs=[two(6.1, 1.7, 12.8), SEP(1.6, 1.7), H(5.4, 3.6),
            ('addlegendentry{NB global $\\theta_g$}', 'addlegendentry{Shared (true median)}'),
            ('\\node[font=\\footnotesize, black, anchor=north, align=center] at (rel axis cs:0.66,0.99){oracle conditional\\\\lowest error in deficit regimes};', ''),
            ('\\node[font=\\footnotesize, black, anchor=north east] at (rel axis cs:0.99,0.995){$=1$ calibrated};', ''),
            ('title={calibration error (lower is better)}', 'title={calibration error}')]), 0, None)],
        [(B('f7cf', 'fig_estimateddispersion', {'a': 'c', 'b': 'd', 'c': 'e', 'd': 'f'},
                subs=[two(6, 1.7, 12.8), SEP(1.8, 1.7), H(4.3, 2.7), ('vertical sep=1.9cm', 'vertical sep=1.75cm'),
                      ('title={Near-Poisson},ymax=0.16', 'title={No deficit},ymin=0.095,ymax=0.135')]), 0, None)]]),
    'Figure8': ('90pt', [[(B('f8', 'fig_scientificboundaries', subs=[two(5.7, 1.5, 11.6), SEP(1.7, 1.5), H(5.1, 3.4),
            ('vertical sep=2.7cm', 'vertical sep=2.15cm'), ('addlegendentry{Independent real draw}', 'addlegendentry{Independent true sample}')]), 0, None)]]),
    'Figure9': ('80pt', [
        [(B('f9a', 'fig_bio', subs=[('width=10.2cm', 'width=6.0cm'), H(5.0, 3.6), ('legend columns=4', 'legend columns=2'),
                ('xticklabels={{marker recall\\\\(Welch)},{marker recall\\\\(AUROC)},{TSTR\\\\macro-F1},{rare-type\\\\F1}}',
                 'xticklabels={{Welch\\\\recall},{AUROC\\\\recall},{TSTR\\\\macro-F1},{rare-type\\\\F1}}'), LABEL('a')], trim='axis'), 0, None),
         (B('f9b', 'fig_e1matched', subs=[('width=4.0cm', 'width=3.5cm'), H(5.0, 3.6)], trim='axis'), (12.6 - 3.5, -0.32), None)],
        [(B('f9ce', 'fig_markerqual', {'a': 'c', 'b': 'd', 'c': 'e'},
                subs=[('width=4.3cm', f'width={(12.6 - 2 * 1.2) / 3:.3f}cm'), H(4.6, 3.2)]), 0, None)]]),
    'Figure10': ('60pt', [
        [(B('f10a', 'fig_e5endtoend', trim='axis'), (0, -0.52), None),
         (B('f10bc', 'fig_biologicalsupport', {'a': 'b', 'b': 'c'}, subs=[('width=5.4cm', 'width=2.7cm'), H(5.2, 3.8),
                SEP(3.0, 2.55), ('at={(1.22,1.20)}', 'at={(0.4,1.22)}'),
                ('xticklabels={Fixed mean,End-to-end}', 'xticklabels={{Fixed\\\\mean},{End to\\\\end}}, x tick label style={align=center}'),
                ('xlabel={Conditional minus comparator}', 'xlabel={Per-cell $-$ comparator}')]), 6.1, None)]]),
    'Figure11': ('4pt', [[(B('f11', 'fig_e4sampling', trim=None), 0, None)]]),
    'FigureS2': ('80pt', [[(B('s2', 'fig_seedreplication', subs=[SEP(1.8, 0.5), ('width=6cm', 'width=5.9cm'), H(6.2, 4.6),
        ('\\node[font=\\scriptsize,anchor=south west] at (rel axis cs:0.02,0.01) {30 seeds; median and central 95\\%};', ''),
        ('\\node[font=\\scriptsize,anchor=south west] at (rel axis cs:0.02,0.01) {30 seeds; median and central 95\\%};', ''),
        ('Near-Poisson', 'No deficit'), ('title={Variance calibration}', 'title={Calibration error}'),
        ('Difference from Poisson (lower is better)', 'Difference from Poisson'),
        ('Difference from Poisson (lower is better)', 'Difference from Poisson')]), 0, None)]]),
}

# Alignment expectations: rows whose first plot areas share a left edge and whose last share a right edge.
ALIGN = {'Figure3': 'columns', 'Figure4': 'edges', 'Figure5': 'edges', 'Figure6': 'columns', 'Figure7': 'columns',
         'Figure8': 'columns', 'Figure9': 'edges', 'FigureS2': 'columns'}


def compose(name, border, rows, gap=0.28):
    rows = [[(block(*spec[1:]), x, label) for spec, x, label in row] for row in rows]
    lines = [preamble(border)]
    for i, row in enumerate(rows):
        for j, _ in enumerate(row):
            lines.append(f'\\newsavebox{{\\blk{chr(65 + i)}{chr(65 + j)}}}')
    lines.append('\\begin{document}')
    for i, row in enumerate(rows):
        for j, (blk, _, _) in enumerate(row):
            lines.append(f'\\sbox{{\\blk{chr(65 + i)}{chr(65 + j)}}}{{\\input{{{(BLOCKS / blk).as_posix()}.tex}}}}')
    lines.append('\\begin{tikzpicture}')
    for i, row in enumerate(rows):
        for j, (blk, x, label) in enumerate(row):
            x, dy = x if isinstance(x, tuple) else (x, 0.0)
            where = f'({x}cm,{dy}cm)' if i == 0 else f'($(row{i - 1})+({x}cm,{dy - gap}cm)$)'
            lines.append(f'\\node[anchor=north west, inner sep=0pt, outer sep=0pt] (r{i}c{j}) at {where} '
                         f'{{\\usebox{{\\blk{chr(65 + i)}{chr(65 + j)}}}}};')
        lowest = ' '.join(f'(r{i}c{j})' for j in range(len(row)))
        lines.append(f'\\node[fit={lowest}, inner sep=0pt] (rowfit{i}) {{}};')
        lines.append(f'\\coordinate (row{i}) at (0,0 |- rowfit{i}.south);')
    lines.append('\\end{tikzpicture}\n\\end{document}')
    p = BUILD / f'{name}.tex'
    p.write_text('\n'.join(lines))
    out = subprocess.run(['xelatex', '-halt-on-error', '-interaction=nonstopmode', f'-output-directory={BUILD}', str(p)],
                         cwd=WORK, capture_output=True, text=True)
    (BUILD / f'{name}.console.log').write_text(out.stdout + out.stderr)
    if out.returncode:
        raise RuntimeError(name + '\n' + out.stdout[-3000:])
    target = R / 'v2'
    target.mkdir(exist_ok=True)
    subprocess.run(['pdfcrop', '--margins', '2', str(BUILD / f'{name}.pdf'), str(target / f'{name}.pdf')],
                   check=True, capture_output=True)
    strip_ptex(target / f'{name}.pdf')
    subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '150', str(target / f'{name}.pdf'), str(target / name)], check=True)
    print('built', name, flush=True)


def strip_ptex(pdf):
    """Remove the pdfTeX inclusion records (PTEX.FileName and friends) that pdfcrop writes with absolute local
    paths; the page content is untouched."""
    doc = fitz.open(pdf)
    changed = False
    for xref in range(1, doc.xref_length()):
        for key in ('PTEX.FileName', 'PTEX.PageNumber', 'PTEX.InfoDict'):
            if doc.xref_get_key(xref, key)[0] != 'null':
                doc.xref_set_key(xref, key, 'null')
                changed = True
    if changed:
        tmp = pdf.with_suffix('.tmp.pdf')
        doc.save(tmp, garbage=3, deflate=True)
        doc.close()
        tmp.replace(pdf)
    else:
        doc.close()


def frames(pdf):
    """Axis frames: black stroked quadrilaterals larger than 40 x 20 pt."""
    rects = []
    for d in fitz.open(pdf)[0].get_drawings():
        r = d['rect']
        if d.get('fill') is None and d.get('color') == (0.0, 0.0, 0.0) and r.width > 40 and r.height > 20 \
                and all(it[0] == 'qu' for it in d['items']):
            rects.append((round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)))
    rows = []
    for rc in sorted(set(rects), key=lambda t: (t[1], t[0])):
        for row in rows:
            if abs(row[0][1] - rc[1]) < 3:
                row.append(rc)
                break
        else:
            rows.append([rc])
    return [sorted(r) for r in rows]


def alignment_audit(names, tol=0.75):
    report, bad = {}, []
    for name in names:
        rows = frames(R / 'v2' / f'{name}.pdf')
        rec = {'rows': [[(a, c) for a, _, c, _ in row] for row in rows]}
        mode = ALIGN.get(name)
        if mode and len(rows) > 1:
            lefts = [row[0][0] for row in rows]
            rights = [row[-1][2] for row in rows]
            rec['left_spread'] = round(max(lefts) - min(lefts), 2)
            rec['right_spread'] = round(max(rights) - min(rights), 2)
            ok = rec['left_spread'] <= tol and rec['right_spread'] <= tol
            if mode == 'columns':
                widths = {len(r) for r in rows}
                if len(widths) == 1:
                    for k in range(len(rows[0])):
                        xs = [r[k][0] for r in rows]
                        ok &= max(xs) - min(xs) <= tol
                else:
                    ok = False
            rec['aligned'] = bool(ok)
            if not ok:
                bad.append(name)
        report[name] = rec
    (R / 'v2' / 'alignment-audit.json').write_text(json.dumps(report, indent=1))
    print('alignment audit:', 'PASS' if not bad else f'FAIL {bad}')
    return bad


def coordinate_audit():
    """Every numeric coordinate block of each plot source must survive verbatim in its block copy."""
    pat = re.compile(r'coordinates\s*\{[^}]*\}', re.S)
    origin = {'fig_forest_broken': 'fig_forest'}
    report = {}
    for name, (_, rows) in FIGURES.items():
        for row in rows:
            for spec, _, _ in row:
                blk, src = spec[1], spec[2]
                orig = pat.findall((FIGS / f'{origin.get(src, src)}.tex').read_text())
                copy = pat.findall((BLOCKS / f'{blk}.tex').read_text())
                missing = [c for c in orig if c not in copy]
                report[blk] = {'source': origin.get(src, src), 'blocks': len(orig), 'all_present': not missing}
    (R / 'v2' / 'coordinate-audit.json').write_text(json.dumps(report, indent=1))
    bad = [k for k, v in report.items() if not v['all_present']]
    assert not bad, f'coordinate blocks changed: {bad}'
    print('coordinate audit PASS:', sum(v['blocks'] for v in report.values()), 'source blocks present in', len(report), 'figure blocks')


def sync(names):
    """Copy built figures into the manuscript folder that the LaTeX sources include. Figure S1 is the
    R-rendered UMAP (figures/refine_umap.R), copied unchanged."""
    import shutil
    shutil.copy2(FIGS / 'fig_umap.pdf', R / 'v2' / 'FigureS1.pdf')
    subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '150', str(R / 'v2' / 'FigureS1.pdf'), str(R / 'v2' / 'FigureS1')], check=True)
    for name in list(names) + ['FigureS1']:
        shutil.copy2(R / 'v2' / f'{name}.pdf', FIGS / f'{name}.pdf')


if __name__ == '__main__':
    import sys
    broken_forest()
    wanted = sys.argv[1:] or list(FIGURES)
    for name in wanted:
        border, rows = FIGURES[name]
        compose(name, border, rows)
    sync(wanted)
    bad = alignment_audit(wanted)
    if not sys.argv[1:]:
        coordinate_audit()
    raise SystemExit(1 if bad else 0)
