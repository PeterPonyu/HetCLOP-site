"""Phase 2u: qualitative marker-gene panel (FIGURE ONLY -- anchors no headline number).

Design (mean-fixed, identical to the phase2q biological readout and the phase2t UMAP
phase): the cached scVI ``gene-cell`` mu_ic / theta_ic (phase2t cache_mu_theta.npz,
the 20,113 held-out test cells) supplies ONE fitted mean; three draws share it:
    (i)   poisson        ~ Poisson(mu)
    (ii)  nb_global      ~ NB(mu, theta_g), theta_g = per-gene median of theta_ic
    (iii) nb_conditional ~ NB(mu, theta_ic)
The cell type shown is selected by the paper's own statistic: the scoreable type
(>= MIN_CELLS test cells) with the largest marker-recall gap
(recall@50[nb_conditional] - recall@50[nb_global]), computed with phase2q's exact
code path (Welch-standardised one-vs-rest mean difference, recall@50 against the
real top-50). For that type we export, for its top-3 real marker genes, shared-bin
log1p-count histograms per variant (real + the three draws) for the figure
pipeline (make_figs.R -> fig_markerqual.tex). No numbers.tex macros are minted;
nothing here feeds a prose claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2j_external_2026-07-09"))
sys.path.insert(0, str(ROOT / "experiments" / "phase2q_biological_2026-07-10"))
sys.path.insert(0, str(ROOT))

import phase2j_lib as L  # noqa: E402
from run_biological import MIN_CELLS, TOPK, marker_recall, marker_scores  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402

CACHE_MU_THETA = ROOT / "experiments" / "phase2t_umap_2026-07-10" / "cache_mu_theta.npz"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 1          # same drawing seed as the phase2q seed-0 run (args.seed + 1)
N_BINS = 25
N_GENES = 3


def main() -> int:
    L.log("=== Phase 2u: qualitative marker panel (mean-fixed draws from cached mu/theta) ===")
    data = L.load_corpus()
    te = data["te"]
    real = data["logX"][te]
    lab = data["labels"][te]

    z = np.load(CACHE_MU_THETA)
    mu = np.clip(np.asarray(z["mu"], np.float64), 0.0, None)
    theta_ic = np.clip(np.asarray(z["theta"], np.float64), 1e-8, None)
    assert mu.shape[0] == int(te.sum()), (
        f"cached mu rows ({mu.shape[0]}) != number of test cells ({int(te.sum())})"
    )
    theta_g = np.median(theta_ic, axis=0)
    L.log(f"  mu/theta_ic {mu.shape}; theta_g per-gene median in "
          f"[{theta_g.min():.3g}, {np.median(theta_g):.3g}, {theta_g.max():.3g}]")

    draws = {
        "poisson": sample_nb_fixed_theta(mu, float("inf"), seed=SEED),
        "nb_global": sample_nb(mu, np.broadcast_to(theta_g, mu.shape).copy(), seed=SEED),
        "nb_conditional": sample_nb(mu, theta_ic, seed=SEED),
    }
    syn = {k: L._log1p(v) for k, v in draws.items()}

    u, c = np.unique(lab, return_counts=True)
    types = [t for t, n in zip(u, c) if n >= MIN_CELLS]
    real_ms = marker_scores(real, lab, types)
    rec = {"real": None}
    for name, X in syn.items():
        rec[name] = marker_recall(real_ms, marker_scores(X, lab, types), types, TOPK)

    # select the type: largest conditional-over-global recall gap, tie-broken by
    # the conditional recall itself (we want a type where the good draw is good).
    gaps = {t: rec["nb_conditional"][t] - rec["nb_global"][t] for t in types}
    best = max(types, key=lambda t: (gaps[t], rec["nb_conditional"][t]))
    n_cells = int((lab == best).sum())
    L.log(f"  selected type '{best}' ({n_cells} test cells): recall "
          f"global={rec['nb_global'][best]:.3f} poisson={rec['poisson'][best]:.3f} "
          f"cond={rec['nb_conditional'][best]:.3f}")

    # top marker genes of the selected type, by the real-data Welch statistic
    order = np.argsort(-real_ms[best])
    panel = json.loads((ROOT / "experiments" / "phase2a_decoder_2026-07-08" / "cache" / "panel.json").read_text())
    genes = panel["panel"]
    sel = order[:N_GENES]

    m = lab == best
    out_genes = []
    for g in sel:
        symbol = genes[int(g)]
        series = {"real": real[m, g]}
        for name, X in syn.items():
            series[name] = X[m, g]
        hi = max(float(np.max(v)) for v in series.values())
        edges = np.linspace(0.0, hi * 1.02 + 1e-9, N_BINS + 1)
        # shared x grid; ECDF per variant (robust to the zero-inflation spike that
        # makes density histograms unreadable for overdispersed draws)
        xs = np.linspace(0.0, hi * 1.02 + 1e-9, 201)
        entry = {
            "symbol": symbol,
            "welch_real": float(real_ms[best][g]),
            "bin_edges": [float(e) for e in edges],
            "hist": {name: [float(x) for x in np.histogram(v, bins=edges, density=True)[0]]
                     for name, v in series.items()},
            "ecdf_x": [float(x) for x in xs],
            "ecdf": {name: [float((v <= x).mean()) for x in xs] for name, v in series.items()},
            "frac_zero": {name: float((v == 0).mean()) for name, v in series.items()},
            "mean_count_real": float(np.expm1(real[m, g]).mean()),
        }
        out_genes.append(entry)
        L.log(f"  gene {symbol}: Welch={entry['welch_real']:.2f} "
              f"mean count={entry['mean_count_real']:.2f}")

    out = {
        "purpose": "qualitative marker-gene panel (figure only; no headline number)",
        "design": "mean-fixed draws from the phase2t cached scVI gene-cell mu/theta_ic "
                  "(20113 held-out test cells); theta_g = per-gene median of theta_ic; "
                  "identical drawing code (sample_nb) as phase2q; draw seed 1, which is exactly the " 
                  "phase2q seed-0 run's draw seed (that run draws with args.seed + 1 = 1)",
        "seed": SEED,
        "cell_type": best,
        "n_cells": n_cells,
        "recall_at_50": {name: float(rec[name][best]) for name in syn},
        "recall_gap_cond_minus_global": float(gaps[best]),
        "genes": out_genes,
    }
    (RESULTS / "markerqual.json").write_text(json.dumps(out, indent=2))
    L.log(f"=== done -> {RESULTS / 'markerqual.json'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
