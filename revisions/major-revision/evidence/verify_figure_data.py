"""Strict check that figure composition did not add, drop or alter plotted data.

The builder's own coordinate audit (figures/build_figures_v2.py) is a subset test on `coordinates {...}` blocks:
every source block must appear in the composed copy. That test would pass if a block were added, and it does
not look at data drawn with `\\draw`/`\\fill` in axis coordinates (error bars, intervals, bands). This verifier
compares, for every composed block against its plot source:

  1. the multiset of `coordinates {...}` blocks, which must be identical up to a declared multiplicity
     (the broken-axis Figure 5c draws the same body in two axes, so its multiplicity is 2);
  2. the multiset of `(axis cs:x,y)` tuples inside `\\draw`, `\\fill` and `\\filldraw` commands, with the same
     multiplicity rule; `\\node` placements are listed separately because layout edits may move annotations;
  3. for the broken axis only, that no plotted marker value falls in the hidden gap between the two y ranges
     and that every marker lies inside the union of the two ranges.

Output: evidence/figure-data-verification.json. Exit status 1 on any data difference.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
FIGS = REV / 'manuscript' / 'figs'
BLOCKS = REV / 'figures' / 'build-v2' / 'blocks'
sys.path.insert(0, str(REV / 'figures'))

COORD = re.compile(r'coordinates\s*\{[^}]*\}', re.S)
CMD = re.compile(r'\\(draw|fill|filldraw|node)\b[^;]*;', re.S)
AXIS = re.compile(r'\(\s*axis cs:\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)')
MULTIPLICITY = {'fig_forest_broken': 2}
# Declared layout-only changes to axis-coordinate drawing commands. Each entry maps a source tuple to its composed
# replacement, with the reason. Anything not declared here must be identical.
ALLOWED = {
    'f4cde': {(('2', '50'), ('2', '65')): 'vertical reference line at the primary threshold (x = 2) shortened to the '
                                          'panel\'s visible y range of 65-82; its x position is unchanged',
              (('2', '90'), ('2', '82')): 'same reference line, upper end'},
    'f3cd': {(('0.887', '0.5'), None): 'unlabelled horizontal guide line at 0.5 removed (no data)',
             (('3.812', '0.5'), None): 'same guide line, other end'},
}
ORIGIN = {'fig_forest_broken': 'fig_forest'}


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def data(text):
    coords = Counter(norm(c) for c in COORD.findall(text))
    marks, nodes = Counter(), Counter()
    for m in CMD.finditer(text):
        tuples = [(a.strip(), b.strip()) for a, b in AXIS.findall(m.group(0))]
        (nodes if m.group(1) == 'node' else marks).update(tuples)
    return coords, marks, nodes


NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
AXIS_START = re.compile(r'\\(?:begin\{axis\}|nextgroupplot)\s*\[')
PT = re.compile(r'\(\s*(' + NUM + r'|[A-Za-z][\w-]*)\s*,\s*(' + NUM + r')\s*\)'
                r'(?:\s*\+-\s*\(\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\))?'
                r'(?:\s*-=\s*\(\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\)\s*\+=\s*\(\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\))?')


def bracket(text, i):
    """Return the option string of an axis that starts with '[' at text[i-1]."""
    depth, j = 1, i
    while depth and j < len(text):
        depth += {'[': 1, ']': -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1], j


def range_check(text):
    """Plotted points and error-bar ends that fall outside an axis's explicit numeric limits (would be clipped)."""
    out = []
    starts = [m.end() for m in AXIS_START.finditer(text)]
    group_opts = ''
    gm = re.search(r'\\begin\{groupplot\}\s*\[', text)
    if gm:
        group_opts, _ = bracket(text, gm.end())
    for k, st in enumerate(starts):
        opts, body_start = bracket(text, st)
        end = starts[k + 1] if k + 1 < len(starts) else len(text)
        body = text[body_start:end]
        merged = group_opts + ',' + opts
        lim = {}
        for key in ['xmin', 'xmax', 'ymin', 'ymax']:
            found = re.findall(r'(?<![\w])' + key + r'\s*=\s*(' + NUM + r')', merged)
            if found:
                lim[key] = float(found[-1])
        if not lim:
            continue
        # Only marks, error bars and bars carry evidence that clipping would hide; lines and bands that run past
        # the frame are cut at the frame without losing a plotted observation.
        for am in re.finditer(r'\\addplot\+?\s*\[(.*?)\]\s*(coordinates\s*\{[^}]*\})', body, re.S):
            aopts, block = am.group(1), am.group(2)
            if not re.search(r'only marks|error bars|ybar|xbar|mark=', aopts):
                continue
            for m in PT.finditer(block):
                pre = block[:m.start()].rstrip()
                if pre.endswith('+-') or pre.endswith('-=') or pre.endswith('+='):
                    continue
                x, y = m.group(1), float(m.group(2))
                ys = [y]; xs = []
                try:
                    xs = [float(x)]
                except ValueError:
                    pass
                if m.group(3):
                    ex, ey = float(m.group(3)), float(m.group(4)); ys += [y - ey, y + ey]
                    if xs: xs += [xs[0] - ex, xs[0] + ex]
                if m.group(5):
                    ys += [y - float(m.group(6)), y + float(m.group(8))]
                    if xs: xs += [xs[0] - float(m.group(5)), xs[0] + float(m.group(7))]
                tol = 1e-9
                bad_y = [v for v in ys if ('ymin' in lim and v < lim['ymin'] - tol) or ('ymax' in lim and v > lim['ymax'] + tol)]
                bad_x = [v for v in xs if ('xmin' in lim and v < lim['xmin'] - tol) or ('xmax' in lim and v > lim['xmax'] + tol)]
                if bad_x or bad_y:
                    out.append({'axis': k, 'limits': lim, 'point': m.group(0)[:80]})
    return out


def figure_specs():
    src = (REV / 'figures' / 'build_figures_v2.py').read_text()
    # Parse B('block', 'source', ...) calls without executing the builder (it needs PyMuPDF and a LaTeX run).
    return re.findall(r"B\('(\w+)',\s*'(\w+)'", src)


def main():
    report, bad = {}, []
    for blk, src in figure_specs():
        origin = ORIGIN.get(src, src)
        k = MULTIPLICITY.get(src, 1)
        s_coords, s_marks, s_nodes = data((FIGS / f'{origin}.tex').read_text())
        c_coords, c_marks, c_nodes = data((BLOCKS / f'{blk}.tex').read_text())
        exp_coords = Counter({key: v * k for key, v in s_coords.items()})
        exp_marks = Counter({key: v * k for key, v in s_marks.items()})
        coord_diff = {'missing': sum((exp_coords - c_coords).values()), 'extra': sum((c_coords - exp_coords).values())}
        allowed = ALLOWED.get(blk, {})
        for (old, new) in allowed:
            if new is None:
                if exp_marks[old] > c_marks[old]:
                    exp_marks[old] -= 1
            elif exp_marks[old] and c_marks[new] > exp_marks[new]:
                exp_marks[old] -= 1; exp_marks[new] += 1
        mark_diff = {'missing': sum((exp_marks - c_marks).values()), 'extra': sum((c_marks - exp_marks).values()),
                     'declared_layout_changes': [f'{o} -> {n}: {why}' for (o, n), why in allowed.items()]}
        node_moved = sum((Counter({key: v * k for key, v in s_nodes.items()}) - c_nodes).values())
        ok = not any(coord_diff.values()) and not mark_diff['missing'] and not mark_diff['extra']
        report[blk] = {'source': origin, 'multiplicity': k, 'coordinate_blocks': sum(s_coords.values()),
                       'axis_marks': sum(s_marks.values()), 'coordinate_diff': coord_diff, 'axis_mark_diff': mark_diff,
                       'node_positions_changed': node_moved, 'data_identical': ok}
        if not ok:
            bad.append(blk)
    # Points or error-bar ends outside explicit axis limits in the composed blocks (reported for inspection).
    clipped = {}
    for blk, src in figure_specs():
        if src == 'fig_forest_broken':
            continue  # each half shows its own range; the union is checked below
        hits = range_check((BLOCKS / f'{blk}.tex').read_text())
        if hits:
            clipped[blk] = hits
    report['range_check'] = clipped
    if clipped:
        bad.append('range_check')
    # Broken axis: markers must not sit in the hidden gap.
    broken = (FIGS / 'fig_forest_broken.tex').read_text()
    lo_max = float(re.search(r'name=forlo,.*?ymax=([-\d.]+)', broken, re.S).group(1))
    lo_min = float(re.search(r'name=forlo,.*?ymin=([-\d.]+)', broken, re.S).group(1))
    axes = re.findall(r'\\begin\{axis\}\[(.*?)\](?=\s*\n)', broken, re.S)
    ranges = []
    for opts in axes:
        ymin = float(re.search(r'ymin=([-\d.]+)', opts).group(1)); ymax = float(re.search(r'ymax=([-\d.]+)', opts).group(1))
        ranges.append((ymin, ymax))
    ranges.sort()
    markers = [float(y) for block in COORD.findall((FIGS / 'fig_forest.tex').read_text())
               for _, y in re.findall(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', block)]
    in_gap = [y for y in markers if ranges[0][1] < y < ranges[1][0]]
    outside = [y for y in markers if not any(a <= y <= b for a, b in ranges)]
    report['fig_forest_broken:gap'] = {'ranges': ranges, 'markers': len(markers), 'in_gap': in_gap, 'outside_ranges': outside,
                                       'lower_axis': [lo_min, lo_max]}
    if in_gap or outside:
        bad.append('fig_forest_broken:gap')
    (REV / 'evidence' / 'figure-data-verification.json').write_text(json.dumps({'pass': not bad, 'failures': bad, 'blocks': report}, indent=1) + '\n')
    n_coords = sum(v['coordinate_blocks'] * v['multiplicity'] for k, v in report.items() if 'multiplicity' in v)
    n_marks = sum(v['axis_marks'] * v['multiplicity'] for k, v in report.items() if 'multiplicity' in v)
    print(f'{len(report) - 1} composed blocks; {n_coords} coordinate blocks and {n_marks} axis-coordinate marks compared exactly')
    print('node placements changed by layout edits:', {k: v['node_positions_changed'] for k, v in report.items() if v.get('node_positions_changed')})
    print('broken axis:', report['fig_forest_broken:gap'])
    print('outside explicit limits:', {k: len(v) for k, v in clipped.items()})
    print('PASS' if not bad else f'FAIL: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
