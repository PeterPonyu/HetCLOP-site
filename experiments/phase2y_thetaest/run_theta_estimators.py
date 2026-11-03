"""Phase 2y -- pin the theta-hat estimator (review panel 2026-07-19, SYNTHESIS #7).

The paper reports theta-hat ~= 1.31 (\\FfourFittedTheta) as "the median of the
decoder's per-cell, per-gene fitted values on held-out cells" -- an amortised
NB-NLL maximum-likelihood fit taken over ALL 2,500 panel genes, with no
expression restriction. With mu << 1 the NB likelihood is flat in theta for most
genes, so MLE and moment-matching are different estimands and the headline
number is unpinned. This script quantifies:

  * MOMENT-MATCHING (MoM): per-gene theta_g = m_g^2 / (v_g - m_g) on train
    counts (genes with v_g > m_g), median over three gene sets:
    all 2,500 / expressed (train mean count >= 0.5, the paper's restriction) /
    top-200 by train variance.
  * MAXIMUM-LIKELIHOOD (MLE): per-gene NB MLE on the SAME train counts
    (score equation sum psi(y+theta) - n psi(theta) + n log(theta/(theta+m)) = 0
    with m = sample mean, solved by brentq; theta -> inf recorded at the upper
    bound when the score is positive throughout = Poisson-indistinguishable).
    Same three gene sets. This isolates estimator choice from model choice: same
    data, no network.
  * WITHIN-GROUP MoM: per (dataset x cell type) group (>= 20 cells, the paper's
    convention), theta = m^2/(v-m) per gene; median over group-gene pairs with
    group mean >= 0.5 -- the conditional estimand the paper's claims concern.
  * DEPLOYED MODEL VALUES (from existing result JSONs, gene set = all 2,500):
    HetCLOP amortised NB decoder 1.308-1.314 (e1_theta_sweep.json);
    scVI dispersion="gene" (per-gene MLE) 0.600-0.603 (scvi.json);
    scVI dispersion="gene-cell" (amortised) 0.272-0.282 (scvi.json), plus the
    expressed-restricted median of the amortised scVI theta from the phase2t
    cache (only model whose full theta matrix is persisted).

Boundary rule: all code under HetCLOP/; corpus loaded via phase2j_lib.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import digamma  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHASE2J = REPO / "experiments" / "phase2j_external_2026-07-09"
PHASE2D = REPO / "experiments" / "phase2d_completion_2026-07-09"
PHASE2T = REPO / "experiments" / "phase2t_umap_2026-07-10"
for p in (str(REPO), str(PHASE2J)):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase2j_lib as L  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EXPR_MIN_MEAN = 0.5     # the paper's expressed-gene restriction (mean count)
TOPK_VAR = 200          # the paper's top-variance corroboration set
MIN_GROUP_CELLS = 20    # the paper's scoreable-group floor


def mom_theta(mean, var):
    """Method-of-moments NB inverse-dispersion; NaN where var <= mean (Poisson-
    or-underdispersed sample: moment estimator undefined, MLE runs to inf)."""
    out = np.full_like(mean, np.nan, dtype=np.float64)
    ok = var > mean
    out[ok] = mean[ok] ** 2 / (var[ok] - mean[ok])
    return out


def mle_theta_gene(y, lo=1e-3, hi=np.exp(8.0)):
    """Per-gene NB MLE for theta given mu_hat = sample mean (the conditional
    MLE; the joint MLE of mu is the sample mean). Returns inf when the score is
    positive at the upper bound (likelihood increasing in theta = Poisson
    indistinguishable)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    m = y.mean()
    if m <= 0:
        return np.nan

    def score(log_th):
        th = np.exp(log_th)
        return float(digamma(y + th).sum() - n * digamma(th) + n * np.log(th / (th + m)))

    slo, shi = score(np.log(lo)), score(np.log(hi))
    if slo <= 0:
        return lo          # maximal dispersion at the bound
    if shi >= 0:
        return float("inf")  # Poisson-indistinguishable at e^8
    return float(np.exp(brentq(score, np.log(lo), np.log(hi), xtol=1e-6)))


def gene_sets(mean_tr, var_tr):
    expressed = mean_tr >= EXPR_MIN_MEAN
    topk = np.zeros_like(expressed)
    topk[np.argsort(-var_tr)[:TOPK_VAR]] = True
    return {"all": np.ones_like(expressed), "expressed": expressed, "top200var": topk}


def summarise(vals, mask, tag):
    v = vals[mask]
    v = v[np.isfinite(v) | np.isinf(v)]
    finite = v[np.isfinite(v)]
    return {
        "gene_set": tag, "n_genes": int(mask.sum()),
        "n_finite": int(finite.size),
        "frac_poisson_indistinguishable": float(np.mean(np.isinf(v))) if v.size else None,
        "median_theta": float(np.median(finite)) if finite.size else None,
        "q25_theta": float(np.percentile(finite, 25)) if finite.size else None,
        "q75_theta": float(np.percentile(finite, 75)) if finite.size else None,
    }


def main() -> int:
    t0 = time.time()
    L.log("=== phase2y: pin the theta-hat estimator ===")
    data = L.load_corpus()
    raw_tr = np.asarray(data["raw"][data["tr"]], dtype=np.float64)
    mean_tr, var_tr = raw_tr.mean(0), raw_tr.var(0)
    sets = gene_sets(mean_tr, var_tr)
    L.log(f"train counts {raw_tr.shape}; expressed genes: {int(sets['expressed'].sum())}")

    # ---- marginal MoM + MLE per gene on train counts --------------------------
    mom = mom_theta(mean_tr, var_tr)
    L.log("solving per-gene NB MLE on train counts (2,500 brentq solves) ...")
    mle = np.array([mle_theta_gene(raw_tr[:, g]) for g in range(raw_tr.shape[1])])

    marginal = {}
    for tag, mask in sets.items():
        marginal[f"mom_{tag}"] = summarise(mom, mask & np.isfinite(mom), tag)
        marginal[f"mle_{tag}"] = summarise(mle, mask, tag)
        L.log(f"  [{tag:9s}] MoM median={marginal[f'mom_{tag}']['median_theta']}  "
              f"MLE median={marginal[f'mle_{tag}']['median_theta']} "
              f"(poisson-indist frac={marginal[f'mle_{tag}']['frac_poisson_indistinguishable']:.3f})")

    # ---- within-group (conditional) MoM ---------------------------------------
    labels, ds = data["labels"], data["dataset_ids"]
    tr = data["tr"]
    vals = []
    for d in np.unique(ds[tr]):
        dm = tr & (ds == d)
        for t in np.unique(labels[dm]):
            mask = dm & (labels == t)
            if int(mask.sum()) < MIN_GROUP_CELLS:
                continue
            X = np.asarray(data["raw"][mask], dtype=np.float64)
            m, v = X.mean(0), X.var(0)
            keep = (m >= EXPR_MIN_MEAN) & (v > m)
            if keep.any():
                vals.append((m[keep] ** 2 / (v[keep] - m[keep])))
    wg = np.concatenate(vals)
    within_group_mom = {
        "group_definition": "dataset x cell type, >= 20 train cells; genes with group mean >= 0.5 and var > mean",
        "n_group_gene_pairs": int(wg.size),
        "median_theta": float(np.median(wg)),
        "q25_theta": float(np.percentile(wg, 25)),
        "q75_theta": float(np.percentile(wg, 75)),
    }
    L.log(f"  [within-group MoM] median={within_group_mom['median_theta']:.3f} over {wg.size} pairs")

    # ---- deployed-model values already on disk ---------------------------------
    e1 = json.loads((PHASE2D / "results" / "e1_theta_sweep.json").read_text())
    hetclop = {s: r["gate"]["fitted_median_theta"] for s, r in e1["per_seed"].items()}
    scvi_json = json.loads((PHASE2J / "results" / "scvi.json").read_text())
    scvi_gene = {s: r["theta_summary"]["median"] for s, r in scvi_json["results"]["gene"].items()}
    scvi_gc = {s: r["theta_summary"]["median"] for s, r in scvi_json["results"]["gene-cell"].items()}

    # expressed-restricted median of the amortised scVI gene-cell theta (phase2t cache)
    z = np.load(PHASE2T / "cache_mu_theta.npz")
    theta_ic = np.asarray(z["theta"], dtype=np.float64)   # test cells x genes
    scvi_gc_expr = float(np.median(theta_ic[:, sets["expressed"]]))
    scvi_gc_all = float(np.median(theta_ic))

    deployed = {
        "hetclop_nb_decoder_amortised_mle_all_genes": {
            "values_per_seed": hetclop,
            "gene_set": "all 2,500 panel genes; median over held-out cells x genes",
            "source": "phase2d e1_theta_sweep.json (\\FfourFittedTheta)"},
        "scvi_gene_mode_per_gene_mle_all_genes": {
            "values_per_seed": scvi_gene,
            "gene_set": "all 2,500 panel genes; median over genes",
            "source": "phase2j scvi.json theta_summary"},
        "scvi_gene_cell_amortised_all_genes": {
            "values_per_seed": scvi_gc,
            "gene_set": "all 2,500 panel genes; median over held-out cells x genes",
            "source": "phase2j scvi.json theta_summary"},
        "scvi_gene_cell_amortised_expressed_only": {
            "value": scvi_gc_expr,
            "gene_set": f"expressed (train mean >= {EXPR_MIN_MEAN}); median over held-out cells x genes",
            "source": "phase2t cache_mu_theta.npz"},
        "scvi_gene_cell_amortised_all_genes_phase2t_cache": scvi_gc_all,
    }

    out = {
        "purpose": "pin the theta-hat estimator: MLE vs moment-matching, over stated gene sets",
        "data": "train split of the shared corpus (raw counts, 2,500-gene panel)",
        "expressed_min_mean": EXPR_MIN_MEAN, "topk_var": TOPK_VAR,
        "marginal_train": marginal,
        "within_group_mom": within_group_mom,
        "deployed_models": deployed,
        "elapsed_s": time.time() - t0,
    }
    outp = RESULTS / "theta_estimators.json"
    outp.write_text(json.dumps(out, indent=2))
    L.log(f"=== done -> {outp} ({time.time()-t0:.0f}s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
