"""Phase 2x helper -- refit scVI gene-cell on CPU for seeds where no mu/theta
cache exists, using the IDENTICAL call phase2q/phase2t used:
fit_and_extract(data, "gene-cell", seed, n_param_samples=10, None, device).
Writes cache_mu_theta_seed{S}.npz for run_rarity.py.

Imports run_scvi.fit_and_extract unmodified.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHASE2J = REPO / "experiments" / "phase2j_external_2026-07-09"
for p in (str(REPO), str(PHASE2J)):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase2j_lib as L  # noqa: E402
from run_scvi import fit_and_extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    outp = HERE / f"cache_mu_theta_seed{args.seed}.npz"
    if outp.exists():
        L.log(f"cache exists: {outp}")
        return 0
    t0 = time.time()
    L.log(f"fitting scVI gene-cell on CPU, seed {args.seed} ...")
    data = L.load_corpus()
    mu, theta_ic, meta = fit_and_extract(
        data, "gene-cell", args.seed, 10, None, "cpu")
    np.savez_compressed(outp, mu=mu, theta=theta_ic)
    L.log(f"done ({time.time()-t0:.0f}s, epochs={meta['n_epochs_trained']}) -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
