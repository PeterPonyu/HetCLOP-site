"""Post hoc check added after review (not part of design.json): the deficit-effect slope refitted on the two encoders
whose means pass the acceptance gate (PCA-512 and random projection-512), because the measured deficit of scGPT and
Geneformer can include mean error. Same model, bootstrap and resample count as analyze.py; new bootstrap seeds."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze

HERE = Path(__file__).resolve().parent


def main():
    frame = pd.read_csv(HERE / 'groups.csv')
    lin = frame[frame.encoder.isin(['pca_512', 'randproj_512'])]
    rows = []
    for i, (label, subset) in enumerate([('linear pooled', lin), ('linear Arm_A', lin[lin.arm == 'A']),
                                         ('linear Arm_B', lin[lin.arm == 'B'])]):
        row, _ = analyze.analyze_frame(subset, label, analyze.DESIGN['inference']['resamples'],
                                       analyze.DESIGN['inference']['seed'] + 100 + i)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / 'linear-encoders.csv', index=False)
    print(out[['analysis', 'n_rows', 'n_groups', 'slope', 'ci_low', 'ci_high', 'direction']].to_string())


if __name__ == '__main__':
    main()
