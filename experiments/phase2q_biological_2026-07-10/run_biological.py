"""Phase 2q (Tier 1) -- does the dispersion law matter BIOLOGICALLY?

Everything in this paper so far is measured in abstract statistical terms (expression-restricted
|log2 r_var|, energy distance). For a computational-biology venue the load-bearing question is
whether getting the dispersion right changes anything a biologist would recognise. This
experiment answers that with two standard biological readouts, and it does so under a design that
makes the answer attributable to DISPERSION ALONE.

DESIGN (mean-fixed; identical logic to the theta probe).
scVI's `gene-cell` mode also fits a worse conditional mean than `gene`, so comparing the two
as deployed would confound the dispersion parameterisation with mean quality. Instead we take ONE
trained model's own fitted mean mu_ic and its amortised theta_ic, and draw three ways:

    (i)   poisson        ~ Poisson(mu)                      [no dispersion]
    (ii)  nb_global      ~ NB(mu, theta_g)                  [theta_g = per-gene median of theta_ic]
    (iii) nb_conditional ~ NB(mu, theta_ic)                 [the law's prescription]

All three share one mean, so any biological difference is caused by the dispersion parameterisation.

READOUTS (both on log1p counts, evaluated on held-out test cells):
  1. MARKER-GENE RECOVERY. For each cell type with >= MIN_CELLS test cells, rank genes by a
     one-vs-rest standardised mean difference computed on REAL cells (the ground-truth markers),
     and the same on each synthetic draw. Report recall@k: the fraction of the real top-k marker
     genes that the synthetic data also places in its own top-k. A decoder that mis-calibrates
     within-type variance inflates or deflates the marker score's denominator and mis-ranks genes.
  2. TSTR (train on synthetic, test on real). Split the test cells into halves A/B by cell.
     Train a multinomial logistic-regression cell-type classifier on the SYNTHETIC version of half
     A (each synthetic cell inherits its source cell's label), and evaluate on the REAL half B.
     Report macro-F1 over all types and, separately, over the RARE types (< RARE_MAX real cells),
     which is where destroyed within-type heterogeneity should hurt most. A classifier trained on
     REAL half A gives the ceiling.

Both readouts are label-aware only through the cell-type annotation already used throughout the
paper; no new supervision is introduced, and the generative draws never see the labels.

"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import phase2j_lib as L  # noqa: E402
from run_scvi import fit_and_extract  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

MIN_CELLS = 20      # a cell type needs this many test cells to be scored
RARE_MAX = 50       # "rare" cell types have fewer than this many real test cells
TOPK = 50           # marker recall@k


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def marker_scores(X, labels, types):
    """One-vs-rest standardised mean difference per (type, gene). Higher = more marker-like.

    score = (mean_in - mean_out) / sqrt(var_in/n_in + var_out/n_out)   (Welch-style)
    This depends on the WITHIN-TYPE VARIANCE, which is exactly what the dispersion sets.
    """
    out = {}
    for t in types:
        m = labels == t
        Xi, Xo = X[m], X[~m]
        ni, no = Xi.shape[0], Xo.shape[0]
        mi, mo = Xi.mean(0), Xo.mean(0)
        vi, vo = Xi.var(0) + 1e-8, Xo.var(0) + 1e-8
        out[t] = (mi - mo) / np.sqrt(vi / ni + vo / no)
    return out


def marker_scores_auroc(X, labels, types):
    """One-vs-rest AUROC per (type, gene): the standard rank-based DE ranking (the direction a
    Wilcoxon rank-sum test / scanpy rank_genes_groups orders genes by). Higher = more marker-like.
    A rank-based score is scale- and monotone-transform-invariant, so it is a robustness check on
    the Welch statistic of marker_scores(); a finding that survives both is not a metric artifact.
    """
    from scipy.stats import rankdata
    out = {}
    for t in types:
        m = labels == t
        ni, no = int(m.sum()), int((~m).sum())
        if ni == 0 or no == 0:
            continue
        ranks = rankdata(X, axis=0)            # rank each gene over all cells
        u = ranks[m].sum(0) - ni * (ni + 1) / 2.0   # Mann-Whitney U for the in-group
        out[t] = (u / (ni * no)).astype(np.float64)  # AUROC per gene
    return out


def marker_recall(real_scores, syn_scores, types, k=TOPK):
    rec = {}
    for t in types:
        if t not in real_scores or t not in syn_scores:
            continue
        r = set(np.argsort(-real_scores[t])[:k].tolist())
        s = set(np.argsort(-syn_scores[t])[:k].tolist())
        rec[t] = len(r & s) / k
    return rec


def _ovr_score(X, in_mask, out_mask):
    """One-vs-rest Welch-standardised mean difference for an explicit in/out split."""
    Xi, Xo = X[in_mask], X[out_mask]
    ni, no = Xi.shape[0], Xo.shape[0]
    mi, mo = Xi.mean(0), Xo.mean(0)
    vi, vo = Xi.var(0) + 1e-8, Xo.var(0) + 1e-8
    return (mi - mo) / np.sqrt(vi / ni + vo / no)


def marker_ceiling(real, labels, types, seed, k=TOPK):
    """Real-vs-real reliability ceiling for marker recall. For each type, split its real cells
    in half, rank markers (one-vs-rest against all non-type cells) on each half independently,
    and measure top-k overlap. This is the maximum recall attainable given finite-sample noise
    in the ranking itself; a synthetic draw cannot be expected to exceed it. Marker recall now
    has the reliability ceiling that TSTR already reports."""
    rng = np.random.default_rng(seed + 7)
    recs = {}
    for t in types:
        idx = np.where(labels == t)[0]
        if len(idx) < 4:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        h = len(idx) // 2
        A, B = idx[:h], idx[h:]
        out = labels != t
        inA = np.zeros(len(labels), bool); inA[A] = True
        inB = np.zeros(len(labels), bool); inB[B] = True
        sA = _ovr_score(real, inA, out)
        sB = _ovr_score(real, inB, out)
        ra = set(np.argsort(-sA)[:k].tolist())
        rb = set(np.argsort(-sB)[:k].tolist())
        recs[t] = len(ra & rb) / k
    return recs


def marker_recall_by_group(real, syn, labels, dataset_ids, min_cells=MIN_CELLS, k=TOPK):
    """Marker recall per (dataset, cell type) group, with one-vs-rest computed WITHIN each
    dataset. This is the dataset-clustered analogue of the pooled per-type readout: markers are
    defined against the same-dataset background, so the score is not confounded by cross-dataset
    pooling, and the resulting groups can be clustered by their source dataset."""
    out = []
    for d in np.unique(dataset_ids):
        dm = dataset_ids == d
        labs_d = labels[dm]
        Xr_d, Xs_d = real[dm], syn[dm]
        u, c = np.unique(labs_d, return_counts=True)
        types_d = [t for t, n in zip(u, c) if n >= min_cells]
        if len(types_d) < 2:  # one-vs-rest within a dataset needs >= 2 types
            continue
        for t in types_d:
            inm = labs_d == t
            outm = ~inm
            sr = _ovr_score(Xr_d, inm, outm)
            ss = _ovr_score(Xs_d, inm, outm)
            r = set(np.argsort(-sr)[:k].tolist())
            s = set(np.argsort(-ss)[:k].tolist())
            out.append({"dataset": str(d), "type": str(t), "recall": len(r & s) / k})
    return out


def dataset_clustered_paired(groups_a, groups_b, n_boot=10000, seed=0):
    """Dataset-clustered paired comparison of two per-(dataset,type) recall lists.
    Aligns on (dataset,type); reduces to one median difference per dataset; runs a sign test
    across datasets and a cluster (over datasets) bootstrap 95% CI on the group-level win
    fraction (a>b). diff = a - b."""
    from scipy.stats import binomtest
    ka = {(g["dataset"], g["type"]): g["recall"] for g in groups_a}
    kb = {(g["dataset"], g["type"]): g["recall"] for g in groups_b}
    keys = sorted(set(ka) & set(kb))
    diffs = np.array([ka[k] - kb[k] for k in keys], dtype=np.float64)
    datasets = [k[0] for k in keys]
    uds = sorted(set(datasets))
    # per-dataset median difference
    ds_arr = np.array(datasets)
    med_by_ds = np.array([np.median(diffs[ds_arr == d]) for d in uds])
    npos = int(np.sum(med_by_ds > 0)); nneg = int(np.sum(med_by_ds < 0))
    n_eff = npos + nneg
    sign_p = binomtest(npos, n_eff, 0.5).pvalue if n_eff > 0 else float("nan")
    # cluster bootstrap over datasets for the group-level win fraction
    rng = np.random.default_rng(seed + 11)
    by_ds = {d: diffs[ds_arr == d] for d in uds}
    wins = []
    for _ in range(n_boot):
        samp = rng.choice(uds, size=len(uds), replace=True)
        pooled = np.concatenate([by_ds[d] for d in samp])
        wins.append(float(np.mean(pooled > 0)))
    lo, hi = np.percentile(wins, [2.5, 97.5])
    return {
        "n_groups": int(diffs.size), "n_datasets": len(uds),
        "win_fraction_groups": float(np.mean(diffs > 0)),
        "median_diff": float(np.median(diffs)),
        "datasets_favouring_a": npos, "datasets_favouring_b": nneg,
        "clustered_sign_p": float(sign_p),
        "win_fraction_ci_lo": float(lo), "win_fraction_ci_hi": float(hi),
    }


def tstr(syn_A, lab_A, real_B, lab_B, rare_types, seed=0):
    """Train a cell-type classifier on SYNTHETIC half A, evaluate on REAL half B."""
    # sklearn >= 1.7 removed `multi_class`; multinomial is the default for multi-class lbfgs.
    clf = LogisticRegression(max_iter=300, n_jobs=-1, C=1.0, random_state=seed)
    clf.fit(syn_A, lab_A)
    pred = clf.predict(real_B)
    macro = f1_score(lab_B, pred, average="macro", zero_division=0)
    # Restrict the rare-type macro to the rare classes themselves. Without an explicit
    # `labels=`, sklearn averages over the union of true and predicted labels within the
    # mask, so an abundant type appearing only as a false positive on a rare cell would
    # enter the average with F1=0 and deflate the score. `labels=rare` scores the 19 rare
    # identities we actually claim recovery for (recall on rare-true cells, precision on
    # predictions of those classes among them).
    rare = sorted(rare_types)
    rare_mask = np.isin(lab_B, rare)
    rare_macro = (f1_score(lab_B[rare_mask], pred[rare_mask], labels=rare,
                           average="macro", zero_division=0)
                  if rare_mask.sum() > 0 else float("nan"))
    # Standard (unmasked) rare-type macro-F1 over the FULL held-out set: precision now counts
    # abundant-class cells misassigned to a rare type as false positives, the stricter reading.
    rare_macro_full = (f1_score(lab_B, pred, labels=rare, average="macro", zero_division=0)
                       if len(rare) > 0 else float("nan"))
    return float(macro), float(rare_macro), float(rare_macro_full)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-param-samples", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    L.log("=== Tier 1: biological readout of the dispersion law (mean-fixed) ===")
    data = L.load_corpus()

    L.log("fitting scVI gene-cell to obtain its own mu and amortised theta ...")
    mu, theta_ic, meta = fit_and_extract(data, "gene-cell", args.seed, args.n_param_samples, None, args.device)
    mu = np.clip(np.asarray(mu, np.float64), 0.0, None)
    theta_ic = np.clip(np.asarray(theta_ic, np.float64), 1e-8, None)
    theta_g = np.median(theta_ic, axis=0)
    L.log(f"  theta_g per-gene median: min={theta_g.min():.3g} med={np.median(theta_g):.3g} max={theta_g.max():.3g}")

    te = data["te"]
    real_counts = data["raw"][te]
    lab = data["labels"][te]
    ds_te = data["dataset_ids"][te]
    real = _log1p(real_counts)

    draws = {
        "poisson": sample_nb_fixed_theta(mu, float("inf"), seed=args.seed + 1),
        "nb_global": sample_nb(mu, np.broadcast_to(theta_g, mu.shape).copy(), seed=args.seed + 1),
        "nb_conditional": sample_nb(mu, theta_ic, seed=args.seed + 1),
    }
    syn = {k: _log1p(v) for k, v in draws.items()}

    # scoreable + rare cell types
    u, c = np.unique(lab, return_counts=True)
    types = [t for t, n in zip(u, c) if n >= MIN_CELLS]
    rare_types = {t for t, n in zip(u, c) if n < RARE_MAX}
    L.log(f"  {len(types)} scoreable types (>= {MIN_CELLS} cells); {len(rare_types)} rare (< {RARE_MAX})")

    # ---- readout 1: marker-gene recovery -----------------------------------------------------
    L.log("computing real marker scores ...")
    real_ms = marker_scores(real, lab, types)
    marker = {}
    for name, X in syn.items():
        ms = marker_scores(X, lab, types)
        rec = marker_recall(real_ms, ms, types, TOPK)
        vals = np.array(list(rec.values()))
        marker[name] = {"mean_recall": float(vals.mean()), "median_recall": float(np.median(vals)),
                        "per_type": {t: float(rec[t]) for t in types}}
        L.log(f"  [marker recall@{TOPK}] {name:15s} mean={vals.mean():.4f} median={np.median(vals):.4f}")

    # paired per-type comparison: conditional vs each other draw
    marker_paired = {}
    for other in ("poisson", "nb_global"):
        d = np.array([marker["nb_conditional"]["per_type"][t] - marker[other]["per_type"][t] for t in types])
        from scipy.stats import wilcoxon
        try:
            p = float(wilcoxon(d).pvalue)
        except Exception:
            p = float("nan")
        marker_paired[f"nb_conditional_vs_{other}"] = {
            "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "win_fraction_types": float(np.mean(d > 0)), "wilcoxon_p": p, "n_types": int(d.size)}
        L.log(f"  [marker paired] conditional vs {other:10s} mean_diff={d.mean():+.4f} "
              f"win={np.mean(d > 0):.3f} p={p:.2e}")

    # ---- readout 1b: rank-based (AUROC / Wilcoxon) marker recall -----------------------------
    # A standard rank-based DE ranking, as a robustness check that the Welch statistic is not
    # driving the result. Reported as mean recall@k per draw; the ordering should match readout 1.
    L.log("computing rank-based (AUROC) marker scores ...")
    real_auroc = marker_scores_auroc(real, lab, types)
    marker_rank = {}
    for name, X in syn.items():
        rec = marker_recall(real_auroc, marker_scores_auroc(X, lab, types), types, TOPK)
        vals = np.array(list(rec.values()))
        marker_rank[name] = {"mean_recall": float(vals.mean()), "median_recall": float(np.median(vals))}
        L.log(f"  [marker AUROC recall@{TOPK}] {name:15s} mean={vals.mean():.4f}")

    # ---- readout 2: TSTR ---------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(real.shape[0])
    half = idx.size // 2
    A, B = idx[:half], idx[half:]
    L.log(f"TSTR: train on synthetic half A ({A.size} cells), test on real half B ({B.size} cells)")
    tstr_res = {}
    for name, X in syn.items():
        macro, rare, rare_full = tstr(X[A], lab[A], real[B], lab[B], rare_types, seed=args.seed)
        tstr_res[name] = {"macro_f1": macro, "rare_macro_f1": rare, "rare_macro_f1_unmasked": rare_full}
        L.log(f"  [TSTR] {name:15s} macro_F1={macro:.4f}  rare_masked={rare:.4f}  rare_unmasked={rare_full:.4f}")
    macro, rare, rare_full = tstr(real[A], lab[A], real[B], lab[B], rare_types, seed=args.seed)
    tstr_res["real_ceiling"] = {"macro_f1": macro, "rare_macro_f1": rare, "rare_macro_f1_unmasked": rare_full}
    L.log(f"  [TSTR] {'real (ceiling)':15s} macro_F1={macro:.4f}  rare_masked={rare:.4f}  rare_unmasked={rare_full:.4f}")

    # ---- readout 3: marker reliability ceiling (real-vs-real split) ---------------------------
    ceil_rec = marker_ceiling(real, lab, types, seed=args.seed)
    marker_ceil = float(np.mean(list(ceil_rec.values()))) if ceil_rec else float("nan")
    L.log(f"  [marker ceiling] real-vs-real split recall@{TOPK} = {marker_ceil:.4f}")

    # ---- readout 4: dataset-clustered marker inference ----------------------------------------
    # Recompute marker recall per (dataset, type) with one-vs-rest WITHIN each dataset, so the
    # comparison respects the paper's dataset-clustered design instead of pooling types globally.
    grp = {name: marker_recall_by_group(real, _log1p(draws[name]), lab, ds_te)
           for name in ("poisson", "nb_global", "nb_conditional")}
    marker_clustered = {}
    for other in ("poisson", "nb_global"):
        r = dataset_clustered_paired(grp["nb_conditional"], grp[other], seed=args.seed)
        marker_clustered[f"nb_conditional_vs_{other}"] = r
        L.log(f"  [marker clustered] cond vs {other:10s} win={r['win_fraction_groups']:.3f} "
              f"(95% CI {r['win_fraction_ci_lo']:.3f}-{r['win_fraction_ci_hi']:.3f}) "
              f"clustered sign p={r['clustered_sign_p']:.2e} over {r['n_datasets']} datasets")

    out = {
        "purpose": "Biological readout of the dispersion law under a MEAN-FIXED design: one model's "
                   "own mu, three dispersion parameterisations. Marker-gene recall@k and "
                   "train-on-synthetic/test-on-real cell-type F1.",
        "design": "mu identical across poisson / nb_global / nb_conditional; only dispersion varies",
        "seed": args.seed, "min_cells": MIN_CELLS, "rare_max": RARE_MAX, "topk": TOPK,
        "n_types_scored": len(types), "n_rare_types": len(rare_types),
        "theta_summary": {"g_median": float(np.median(theta_g)), "ic_median": float(np.median(theta_ic))},
        "marker_recovery": marker, "marker_paired": marker_paired,
        "marker_recovery_auroc": marker_rank,
        "marker_ceiling": marker_ceil,
        "marker_clustered": marker_clustered,
        "tstr": tstr_res, "train_meta": meta,
    }
    (RESULTS / "biological.json").write_text(json.dumps(out, indent=2))
    L.log(f"=== done -> {RESULTS / 'biological.json'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
