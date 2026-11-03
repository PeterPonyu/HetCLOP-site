"""Phase 2r (Tier 3) -- is F4's headline robust to the expression cut?

Limitation stated in the manuscript: "The expression cut (mean count >= 0.5) is arbitrary and is
corroborated, not proven, on the top-200-by-variance genes." The expression restriction exists
because the UNRESTRICTED per-gene variance ratio is directionally meaningless on a sparse count
panel (most genes have within-type mean << 1, and discrete low counts pin the median at exactly
1.0 for any theta). But if F4's conclusion depended on WHERE we put the cut, the whole primary
claim would be an artefact of one threshold.

This sweeps the cut and re-runs F4's primary contrast at each value, in the paper's own primary
setting: Arm A, PCA-512, the NB decoder's OWN fitted mean held fixed, dispersion-free Poisson draw
versus the learned-theta negative binomial, on the expression-restricted |log2 r_var|.

  diff = NB - Poisson  (both lower-is-better)  =>  diff > 0 means POISSON wins that group.

For every cut we report the Poisson win-fraction over the (dataset x cell type) groups, the
dataset-clustered Wilcoxon p, and how many genes survive the cut. The claim gate is that the
DIRECTION (Poisson wins, clustered-significant) is invariant across the whole sweep, including the
degenerate cut 0.0 -- where the statistic is known to be meaningless and is reported for contrast,
not as support.

"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2f_f4_resolution_2026-07-09"))

import phase2j_lib as L  # noqa: E402
from clustered_inference import clustered_dataset_medians, naive_paired  # noqa: E402
from src.architecture.stochastic_decoder import (  # noqa: E402
    sample_nb, sample_nb_fixed_theta, train_nb_decoder,
)
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

CUTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]   # 0.0 = unrestricted (known-degenerate; contrast only)
PAPER_CUT = 0.5
MIN_CELLS = 20


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def group_masks(labels, dataset_ids, min_cells=MIN_CELLS):
    groups = []
    for ds in np.unique(dataset_ids):
        m_ds = dataset_ids == ds
        for t in np.unique(labels[m_ds]):
            m = m_ds & (labels == t)
            if m.sum() >= min_cells:
                groups.append((str(ds), str(t), m))
    return groups


def sweep_one_cut(real_log, syn_nb_log, syn_pois_log, groups, cut):
    """Return per-group diffs (NB - Poisson) on |log2 r_var| at this expression cut."""
    recs, n_genes = [], []
    for ds, t, m in groups:
        R = real_log[m]
        a = r_var_per_gene_expressed(R, syn_nb_log[m], min_mean_count=cut)["expressed"]
        b = r_var_per_gene_expressed(R, syn_pois_log[m], min_mean_count=cut)["expressed"]
        if not (np.isfinite(a["median_abs_log2_ratio"]) and np.isfinite(b["median_abs_log2_ratio"])):
            continue
        n_genes.append(a.get("n_genes_scored", np.nan))
        av, bv = float(a["median_abs_log2_ratio"]), float(b["median_abs_log2_ratio"])
        # naive_paired() requires the two paired values, not just their difference.
        recs.append({"dataset": ds, "cell_type": t, "a": av, "b": bv, "diff": av - bv})
    return recs, n_genes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    L.log("=== Tier 3: expression-cut sensitivity of F4's primary claim (Arm A, PCA-512) ===")
    data = L.load_corpus()
    logX, raw = data["logX"], data["raw"]
    tr, va, te = data["tr"], data["va"], data["te"]
    lab, ds = data["labels"], data["dataset_ids"]

    lib = raw.sum(axis=1).astype(np.float32)
    mean_lib = float(lib[tr].mean())

    cache = RESULTS.parent / "cache_mu_theta.npz"
    if cache.exists():
        L.log(f"reusing cached mu/theta from {cache.name} (skips PCA + decoder training)")
        z = np.load(cache)
        mu, theta = z["mu"], z["theta"]
        return finish(data, mu, theta, args)

    L.log("fitting PCA-512 on the train split (Arm A: encoder sees the panel it decodes) ...")
    Z = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)

    L.log("training the NB decoder (library-size offset) ...")
    nva = int(va.sum()) if va.dtype == bool else len(va)
    Z_arg = np.vstack([Z[va], Z[te]])
    lib_arg = np.concatenate([lib[va], lib[te]]).astype(np.float32)
    mu_all, th_all, _, _, _ = train_nb_decoder(
        Z[tr], raw[tr], lib[tr], Z[va], raw[va], lib[va], Z_arg, lib_arg,
        mean_libsize=mean_lib, seed=args.seed, device=args.device,
        logtheta_min=-4.0, logtheta_max=8.0, warmup_epochs=15, logtheta_lr_scale=0.1)
    mu, theta = mu_all[nva:], th_all[nva:]
    L.log(f"  theta median {np.median(theta):.3f}")
    np.savez_compressed(cache, mu=mu, theta=theta)
    L.log(f"  cached mu/theta -> {cache.name}")
    return finish(data, mu, theta, args)


def finish(data, mu, theta, args):
    logX, te = data["logX"], data["te"]
    lab, ds = data["labels"], data["dataset_ids"]
    real_log = logX[te]
    nb = _log1p(sample_nb(mu, theta, seed=args.seed + 1))
    pois = _log1p(sample_nb_fixed_theta(mu, float("inf"), seed=args.seed + 1))

    groups = group_masks(lab[te], ds[te])
    L.log(f"  {len(groups)} (dataset x cell type) groups with >= {MIN_CELLS} test cells")

    out_cuts = {}
    for cut in CUTS:
        recs, n_genes = sweep_one_cut(real_log, nb, pois, groups, cut)
        diffs = np.array([r["diff"] for r in recs])
        cl = clustered_dataset_medians(recs)
        out_cuts[str(cut)] = {
            "n_groups": int(diffs.size),
            "median_genes_surviving": (float(np.median(n_genes)) if n_genes and np.isfinite(n_genes[0]) else None),
            "poisson_win_fraction_groups": float(np.mean(diffs > 0)),
            "median_paired_diff_nb_minus_pois": float(np.median(diffs)),
            "naive_p": naive_paired(recs)["p_value"],
            "clustered_wilcoxon_p": cl["wilcoxon_p"],
            "clustered_n_datasets": cl["n_datasets"],
        }
        c = out_cuts[str(cut)]
        L.log(f"  [cut {cut:>4}] poisson_win={c['poisson_win_fraction_groups']:.3f} "
              f"medDiff={c['median_paired_diff_nb_minus_pois']:+.4f} "
              f"clustWil_p={c['clustered_wilcoxon_p']:.2e} "
              f"{'(unrestricted: known-degenerate, contrast only)' if cut == 0.0 else ''}")

    informative = [c for c in CUTS if c > 0.0]
    direction_stable = all(out_cuts[str(c)]["poisson_win_fraction_groups"] > 0.5 for c in informative)
    sig_stable = all(out_cuts[str(c)]["clustered_wilcoxon_p"] < 0.05 for c in informative)

    out = {
        "purpose": "Sensitivity of F4's primary claim to the arbitrary expression cut "
                   "(mean count >= 0.5). Arm A, PCA-512, NB's own fitted mean held fixed.",
        "convention": "diff = NB - Poisson on |log2 r_var| (lower better); diff>0 => Poisson wins",
        "paper_cut": PAPER_CUT, "cuts": CUTS, "seed": args.seed,
        "per_cut": out_cuts,
        "direction_stable_across_informative_cuts": bool(direction_stable),
        "significance_stable_across_informative_cuts": bool(sig_stable),
        "note": "cut 0.0 is the unrestricted statistic F3 discredits (sparsity pins the median at 1.0); "
                "it is reported for contrast and is NOT evidence for or against the claim.",
    }
    (RESULTS / "exprcut.json").write_text(json.dumps(out, indent=2))
    L.log(f"direction stable across informative cuts: {direction_stable}; significance stable: {sig_stable}")
    L.log(f"=== done -> {RESULTS / 'exprcut.json'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
