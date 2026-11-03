"""List macros with a negative value that are used outside math mode (they would print a hyphen, not a minus).

Scans manuscript/parts_v2/*.tex and manuscript/supplementary.tex against every macro file the manuscript inputs.
Exit status 1 when any such use exists. Output: evidence/minus-sign-check.json.
"""
import json
import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
MS = REV / 'manuscript'
macros = {}
for f in sorted((MS / 'figs').glob('*numbers*.tex')):
    for k, v in re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', f.read_text()):
        macros[k] = v
negative = {k for k, v in macros.items() if re.match(r'\s*-\s*\d', v)}


def strip_comments(text):
    return re.sub(r'(?<!\\)%.*', '', text)


def outside_math_uses(text):
    hits = []
    math = False
    i = 0
    env_math = re.compile(r'\\begin\{(equation|align|gather|multline)\*?\}')
    env_end = re.compile(r'\\end\{(equation|align|gather|multline)\*?\}')
    while i < len(text):
        if text.startswith('\\$', i):
            i += 2
            continue
        m = env_math.match(text, i)
        if m:
            math = True; i = m.end(); continue
        m = env_end.match(text, i)
        if m:
            math = False; i = m.end(); continue
        if text.startswith('\\(', i) or text.startswith('\\[', i):
            math = True; i += 2; continue
        if text.startswith('\\)', i) or text.startswith('\\]', i):
            math = False; i += 2; continue
        if text[i] == '$':
            math = not math; i += 1; continue
        if text[i] == '\\':
            m = re.match(r'\\([A-Za-z]+)', text[i:])
            if m:
                name = m.group(1)
                if name in negative and not math:
                    line = text.count('\n', 0, i) + 1
                    hits.append({'macro': name, 'value': macros[name], 'line': line})
                i += len(m.group(0)); continue
        i += 1
    return hits


report = {}
for f in sorted((MS / 'parts_v2').glob('*.tex')) + [MS / 'supplementary.tex']:
    hits = outside_math_uses(strip_comments(f.read_text()))
    if hits:
        report[str(f.relative_to(REV))] = hits
(REV / 'evidence' / 'minus-sign-check.json').write_text(json.dumps(report, indent=1) + '\n')
n = sum(len(v) for v in report.values())
for f, hits in report.items():
    for h in hits:
        print(f"{f}:{h['line']}: \\{h['macro']} = {h['value']}")
print(f'{n} negative macro uses outside math mode')
sys.exit(1 if n else 0)
