from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
E2 = BASE / 'experiments/e2-dose-response'
E1 = BASE / 'experiments/e1-armb-biology'
FIGS = BASE / 'manuscript/figs'


def fmt(value):
    return f'{float(value):.4f}' if abs(float(value)) >= .00005 else (r'\num{' + f'{float(value):.3g}' + '}' if float(value) else '0.0000')


def main():
    for folder in [E2, E1]:
        assert json.loads((folder / 'verification.json').read_text())['status'] == 'PASS'
    e2 = pd.read_csv(E2 / 'summary.csv')
    e1 = pd.read_csv(E1 / 'summary.csv')
    interaction = pd.read_csv(E1 / 'arm-interaction.csv')
    pooled = e2[e2.analysis == 'pooled'].iloc[0]
    numbers = {'EtwoFits': '24', 'EtwoRows': str(int(pooled.n_rows)), 'EtwoGroups': str(int(pooled.n_groups)), 'EtwoDatasets': str(int(pooled.n_datasets))}
    tab2 = [r'\begin{tabular}{lrrrr}', r'\toprule',
            r'Comparison & Groups & Datasets & Slope & 95\% interval \\', r'\midrule']
    for _, row in e2.iterrows():
        code = {'pooled': 'Pooled', 'Arm_A': 'ArmA', 'Arm_B': 'ArmB'}[row.analysis]
        name = {'pooled': 'Both arms', 'Arm_A': 'Arm A', 'Arm_B': 'Arm B'}[row.analysis]
        for suffix, field in [('Slope', 'slope'), ('Lo', 'ci_low'), ('Hi', 'ci_high')]:
            numbers[f'Etwo{code}{suffix}'] = fmt(row[field])
        tab2.append(f'{name} & {row.n_groups} & {row.n_datasets} & $\\Etwo{code}Slope$ & $[\\Etwo{code}Lo,\\Etwo{code}Hi]$ ' + r'\\')
    tab2 += [r'\bottomrule', r'\end{tabular}']
    (FIGS / 'table_e2.tex').write_text('\n'.join(tab2) + '\n')
    metric_names = {'calibration': 'Cal', 'welch': 'Welch', 'auroc': 'Auroc', 'macro_f1': 'Macro', 'rare_f1_full': 'Rare'}
    metric_headers = ['Calibration', 'Welch recall', 'AUROC recall', 'Macro-F1', 'Rare-type F1']
    variant_names = {'shared_per_gene': ('Shared', 'Shared per-gene'), 'per_cell_type': ('Type', 'Per-cell-type'), 'per_cell': ('Cell', 'Per-cell')}
    tab1 = [r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llrrrrr}', r'\toprule',
            'Arm & Dispersion & ' + ' & '.join(metric_headers) + r' \\', r'\midrule']
    for arm in ['A', 'B']:
        for variant, (vcode, vname) in variant_names.items():
            cells = []
            for metric, mcode in metric_names.items():
                row = e1[(e1.arm == arm) & (e1.a == variant) & (e1.b == 'poisson') & (e1.metric == metric)].iloc[0]
                prefix = f'Eone{arm}{vcode}{mcode}'
                for suffix, field in [('Effect', 'effect'), ('Lo', 'ci_low'), ('Hi', 'ci_high')]:
                    numbers[prefix + suffix] = fmt(row[field])
                cells.append(rf'\shortstack{{${chr(92)}{prefix}Effect$\\$[{chr(92)}{prefix}Lo,{chr(92)}{prefix}Hi]$}}')
            tab1.append(f'{arm} & {vname} & ' + ' & '.join(cells) + r' \\[4pt]')
        if arm == 'A':
            tab1.append(r'\midrule')
    tab1 += [r'\bottomrule', r'\end{tabular}']
    (FIGS / 'table_e1.tex').write_text('\n'.join(tab1) + '\n')
    for arm in ['A', 'B']:
        for metric, mcode in metric_names.items():
            row = e1[(e1.arm == arm) & (e1.a == 'per_cell') & (e1.b == 'shared_per_gene') & (e1.metric == metric)].iloc[0]
            for suffix, field in [('Effect', 'effect'), ('Lo', 'ci_low'), ('Hi', 'ci_high')]:
                numbers[f'Eone{arm}Allocation{mcode}{suffix}'] = fmt(row[field])
    for _, row in interaction.iterrows():
        if row.metric not in metric_names:
            continue
        prefix = 'EoneInteraction' + metric_names[row.metric]
        for suffix, field in [('Effect', 'effect'), ('Lo', 'ci_low'), ('Hi', 'ci_high')]:
            numbers[prefix + suffix] = fmt(row[field])
    marker = pd.read_csv(E1 / 'dose-response-marker-summary.csv')
    for _, row in marker.iterrows():
        prefix = 'EoneDose' + metric_names[row.analysis]
        for suffix, field in [('Slope','slope'), ('Lo','ci_low'), ('Hi','ci_high')]:
            numbers[prefix + suffix] = fmt(row[field])
    (FIGS / 'stage_a_numbers.tex').write_text('\n'.join('\\newcommand{\\' + key + '}{' + val + '}' for key, val in numbers.items()) + '\n')
    (BASE / 'evidence/stages/manuscript-number-ledger.json').write_text(json.dumps({'sources': ['experiments/e2-dose-response/summary.csv',
        'experiments/e1-armb-biology/summary.csv', 'experiments/e1-armb-biology/arm-interaction.csv',
        'experiments/e1-armb-biology/dose-response-marker-summary.csv'], 'macros': numbers}, indent=2) + '\n')
    print(f'Generated {len(numbers)} macros and two tables from verified new-run summaries.')


if __name__ == '__main__':
    main()
