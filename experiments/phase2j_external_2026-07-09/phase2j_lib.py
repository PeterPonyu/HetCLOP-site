"""Phase 2j -- test F4 against REAL, published single-cell generative decoders.

F4 (per the spec, ss20): holding a negative-binomial decoder's own fitted mean
fixed, a dispersion-free Poisson draw (theta -> inf) beats the learned-theta NB on
expression-restricted ``|log2 r_var|`` (the per-gene within-type variance
calibration error). The referee objection F4 must survive is that it is a control
on a home-built decoder. This module runs the SAME within-decoder control on two
DEPLOYED arrangements:

  * scVI (Lopez et al. 2018, ``scvi-tools``) -- real library, its own VAE
    encoder/decoder; fit twice, ``dispersion="gene"`` (global, CFGen-like) and
    ``dispersion="gene-cell"`` (per-cell amortized). F4 predicts the effect should
    be WEAKER under amortized dispersion.
  * CFGen's own ``EncoderModel`` (Palma et al. ICLR 2025; the exact autoencoder /
    decoder code from the cloned ``theislab/CFGen`` repo) -- ``mu = l*softmax(
    h_psi(z))`` with a GLOBAL gene-wise ``theta_g``. This is CFGen's real decoder
    code, trained with a matched custom loop on our corpus (the flow-matching
    stage is irrelevant to F4, which touches only the decoder).

THE CONTROL (identical across models, matches Phase 2c/2d/2f exactly): given per
test cell/gene the model's fitted count mean ``mu_ng`` and dispersion ``theta``
(``theta_g`` global or ``theta_ng`` amortized), draw counts two ways holding
``mu_ng`` FIXED:
  * ``nb_stochastic``   -- sample from NB(mu_ng, theta), as the model does.
  * ``poisson_on_mean`` -- sample from Poisson(mu_ng): same mean, theta -> inf.
Evaluate both on log1p with our metrics and compare paired. Because both arms use
the model's OWN mu, the acceptance gate (which governs mean/procedure conclusions)
CANNOT license excluding a model from this dispersion contrast (spec ss20/ss22).

Discriminating metrics (only these two, per the standing rules): expression-
restricted ``|log2 r_var|`` (``r_var_abs_log2_expr``, mean count >= 0.5,
corroborated on top-200-by-variance) and ``energy_distance``. Signed direction
(``r_var_signed_expr``, ``r_var_frac_gt1_expr``) is measured, not inferred.
Inference: naive paired Wilcoxon over 314 groups AND dataset-clustered (50
datasets). Energy claims report a >=10 count-redraw p-range.

"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHASE2A = REPO / "experiments" / "phase2a_decoder_2026-07-08"
PHASE2F = REPO / "experiments" / "phase2f_f4_resolution_2026-07-09"
for p in (str(REPO), str(PHASE2A), str(PHASE2F)):
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE = PHASE2A / "cache"

from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.evaluation.distributional import (  # noqa: E402
    aggregate_over_groups, energy_distance,
)
from clustered_inference import (  # noqa: E402
    clustered_dataset_medians, dataset_signflip_permutation, naive_paired,
    paired_group_diffs, within_dataset_signflip_permutation,
)

GATE_TOL = 0.02
# Only these two are valid discriminators (standing rule). r_var is the
# EXPRESSION-RESTRICTED calibration error; that is what carries F4.
RVAR_METRIC = "r_var_abs_log2_expr"       # primary (expressed genes, |log2 ratio|)
RVAR_METRIC_TOPK = "r_var_abs_log2_topk"  # corroboration (top-200 by variance)
ENERGY_METRIC = "energy_distance"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpus():
    """Load the shared corpus and the SAME train/val/test split used everywhere.

    raw_full.npy (counts), logX_full.npy (= log1p(raw)), meta.npz (labels,
    dataset_ids, split). Returns a dict of arrays and boolean split masks."""
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    raw = np.load(CACHE / "raw_full.npy").astype(np.float32)
    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    # Sanity-assert the reused inputs are internally consistent (standing rule).
    assert np.allclose(np.log1p(raw[:2000]), logX[:2000], atol=1e-5), \
        "logX must equal log1p(raw) for metric comparability"
    tr, va, te = split == 0, split == 1, split == 2
    return {
        "raw": raw, "logX": logX, "labels": labels, "dataset_ids": dataset_ids,
        "split": split, "tr": tr, "va": va, "te": te,
        "lib": raw.sum(axis=1).astype(np.float32),
    }


def deterministic_baseline_log_pearson(data, seed=42, device="cuda"):
    """Deterministic MSE-on-log1p decoder on a PCA-512 latent -- the same
    reference the Phase 2c/2d/2f acceptance gate uses (det_log_pearson ~ 0.742 on
    pca_512). Returns its test-set mean-gene log1p Pearson. The gate is a coarse
    one-sided underfit guard; it governs mean/procedure conclusions ONLY and does
    NOT gate the within-decoder dispersion contrast."""
    from sklearn.decomposition import PCA
    from decoder_eval import reconstruction_metrics, train_decoder
    logX, tr, va, te = data["logX"], data["tr"], data["va"], data["te"]
    Z = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)
    det_log_te, _, _, _ = train_decoder(
        Z[tr], logX[tr], Z[va], logX[va], Z[te], seed=seed, device=device)
    return float(reconstruction_metrics(det_log_te, logX[te])["mean_gene_pearson"])


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def per_group_map(agg):
    return {(g["dataset"], g["cell_type"]): g for g in agg["per_group"]}


def _clustered(records):
    """All three clustered views + naive, for one paired contrast (a=nb, b=pois).
    Returns p-values and win fractions. diff = nb - poisson; for lower-is-better
    metrics diff>0 => POISSON better. poisson_win_fraction reported explicitly."""
    nv = naive_paired(records)
    cl = clustered_dataset_medians(records)
    perm = dataset_signflip_permutation(records, n_perm=10000, seed=0)
    wperm = within_dataset_signflip_permutation(records, n_perm=10000, seed=0)
    diffs = np.array([r["diff"] for r in records], dtype=np.float64)
    return {
        "n_groups": int(diffs.size),
        "poisson_win_fraction_groups": float(np.mean(diffs > 0)),
        "median_paired_diff_nb_minus_pois": float(np.median(diffs)),
        "naive_p": nv["p_value"],
        "clustered_wilcoxon_p": cl["wilcoxon_p"],
        "clustered_sign_p": cl["sign_test_p"],
        "clustered_n_datasets": cl["n_datasets"],
        "poisson_win_fraction_datasets": float(np.mean(np.array(cl["dataset_medians"]) > 0)),
        "dataset_signflip_perm_p": perm["p_value"],
        "within_dataset_signflip_perm_p": wperm["p_value"],
    }


def evaluate_dispersion_control(
    model_name, mu_te, theta_te, data, det_log_pearson,
    seed=0, n_energy_redraws=10, min_cells=20, top_k=200,
):
    """Run the within-decoder Poisson-vs-NB control for one fitted model.

    mu_te : (n_te, G) the model's fitted COUNT-space mean per test cell/gene.
    theta_te : (n_te, G) inverse-dispersion (scVI convention: var = mu + mu^2/theta);
        broadcast a global theta_g to this shape for the gene-wise case.
    Draws nb_stochastic ~ NB(mu, theta) and poisson_on_mean ~ Poisson(mu) (theta ->
    inf), HOLDING mu FIXED, evaluates both on log1p, and runs the paired + clustered
    tests on the two valid discriminators. Also does an energy count-redraw sweep."""
    logX, te, labels, dataset_ids = data["logX"], data["te"], data["labels"], data["dataset_ids"]
    real_te = logX[te]
    lab_te, ds_te = labels[te], dataset_ids[te]
    mu_te = np.clip(np.asarray(mu_te, dtype=np.float64), 0.0, None)
    theta_te = np.asarray(theta_te, dtype=np.float64)

    # Acceptance gate (informational; does NOT gate the dispersion contrast).
    nb_mean_log = _log1p(mu_te)
    from decoder_eval import reconstruction_metrics
    model_log_pearson = float(reconstruction_metrics(nb_mean_log, real_te)["mean_gene_pearson"])
    gap = det_log_pearson - model_log_pearson
    gate_passed = mean_acceptance_gate(det_log_pearson, model_log_pearson, GATE_TOL)
    log(f"  [{model_name}] mean_log1p_r={model_log_pearson:.4f} "
        f"(det={det_log_pearson:.4f} gap={gap:+.4f}) "
        f"gate={'PASS' if gate_passed else 'FAIL'} "
        f"theta[med={np.median(theta_te):.3f} min={theta_te.min():.3g} max={theta_te.max():.3g}]")

    # Draw the two variants, holding mu fixed.
    nb_counts = sample_nb(mu_te, theta_te, seed=seed)
    pois_counts = sample_nb_fixed_theta(mu_te, float("inf"), seed=seed)
    decoded = {"nb_stochastic": _log1p(nb_counts), "poisson_on_mean": _log1p(pois_counts)}

    aggs, agg_full = {}, {}
    for vname, dec in decoded.items():
        agg = aggregate_over_groups(real_te, dec, lab_te, ds_te,
                                    min_cells=min_cells, energy_subsample=300,
                                    top_k=top_k, seed=0)
        aggs[vname] = per_group_map(agg)
        agg_full[vname] = agg
    n_groups = agg_full["nb_stochastic"]["n_groups"]

    # Paired + clustered on the two valid discriminators. a=nb, b=poisson.
    contrasts = {}
    for label, metric in [("rvar_expressed", RVAR_METRIC),
                          ("rvar_topk", RVAR_METRIC_TOPK),
                          ("energy", ENERGY_METRIC)]:
        recs = paired_group_diffs(aggs["nb_stochastic"], aggs["poisson_on_mean"], metric)
        contrasts[label] = {"metric": metric, **_clustered(recs)}
        c = contrasts[label]
        log(f"    [{label:16s}] pois_win_grp={c['poisson_win_fraction_groups']:.3f} "
            f"medDiff(nb-pois)={c['median_paired_diff_nb_minus_pois']:+.4f} "
            f"naive_p={c['naive_p']:.2e} clustWil_p={c['clustered_wilcoxon_p']:.2e} "
            f"clustSign_p={c['clustered_sign_p']:.2e}")

    # Signed direction of the miscalibration (over- vs under-inflation), on the
    # EXPRESSED support, for each variant (medians across groups).
    direction = {}
    for vname in decoded:
        a = agg_full[vname]
        direction[vname] = {
            "r_var_signed_expr_median": a["r_var_signed_expressed"]["median"],
            "r_var_frac_gt1_expr_median": a["r_var_frac_gt1_expressed"]["median"],
            "r_var_abs_log2_expr_median": a["r_var_abs_log2_expressed"]["median"],
            "energy_median": a["energy_distance"]["median"],
            "r_var_signed_topk_median": a["r_var_signed_topk"]["median"],
        }
    log(f"    [direction] nb signed_expr="
        f"{direction['nb_stochastic']['r_var_signed_expr_median']:.3f} "
        f"frac>1={direction['nb_stochastic']['r_var_frac_gt1_expr_median']:.3f} | "
        f"pois signed_expr={direction['poisson_on_mean']['r_var_signed_expr_median']:.3f} "
        f"frac>1={direction['poisson_on_mean']['r_var_frac_gt1_expr_median']:.3f}")

    # Energy count-redraw sweep: the energy estimator sits near its noise floor and
    # the single stochastic count draw is the real source of variability (spec E8).
    # Re-sample BOTH variants at >=10 fresh seeds, recompute per-group energy, and
    # report the clustered-Wilcoxon and naive p ranges. (<=300 subsample is inert:
    # max test group <= 286 cells.)
    energy_redraw = _energy_redraw_sweep(
        mu_te, theta_te, real_te, lab_te, ds_te, n_energy_redraws, min_cells)
    log(f"    [energy redraw x{n_energy_redraws}] clustWil_p in "
        f"[{energy_redraw['clustered_wilcoxon_p_min']:.2e}, "
        f"{energy_redraw['clustered_wilcoxon_p_max']:.2e}] "
        f"naive_p in [{energy_redraw['naive_p_min']:.2e}, {energy_redraw['naive_p_max']:.2e}] "
        f"any>0.05={energy_redraw['any_clustered_gt_0.05']}")

    return {
        "model": model_name, "seed": seed, "n_groups": int(n_groups),
        "gate": {
            "det_log_pearson": det_log_pearson,
            "model_mean_log_pearson": model_log_pearson,
            "gap_vs_det": float(gap), "gate_passed": bool(gate_passed),
            "note": "one-sided underfit guard; governs mean/procedure conclusions "
                    "only; does NOT gate the dispersion contrast (mu held fixed in "
                    "both arms).",
        },
        "theta_summary": {
            "median": float(np.median(theta_te)), "min": float(theta_te.min()),
            "max": float(theta_te.max()),
            "is_amortized": bool(theta_te.std(axis=0).max() > 1e-6),
        },
        "contrasts": contrasts,
        "direction": direction,
        "energy_redraw": energy_redraw,
    }


def _energy_redraw_sweep(mu_te, theta_te, real_te, lab_te, ds_te, n_redraws, min_cells):
    """For n_redraws fresh count-sample seeds, recompute per-group energy for
    nb_stochastic and poisson_on_mean and the clustered/naive p of nb-vs-poisson.
    Groups mirror aggregate_over_groups (>= min_cells; no subsample needed since
    max group <= 286 < 300). Returns per-redraw p arrays and their ranges."""
    real_te = np.asarray(real_te, dtype=np.float64)
    groups = []
    for d in np.unique(ds_te):
        dm = ds_te == d
        for t in np.unique(lab_te[dm]):
            mask = dm & (lab_te == t)
            if int(mask.sum()) >= min_cells:
                groups.append((str(d), str(t), np.where(mask)[0]))
    naive_ps, cl_ps, sign_ps = [], [], []
    for k in range(n_redraws):
        nb = _log1p(sample_nb(mu_te, theta_te, seed=1000 + k))
        po = _log1p(sample_nb_fixed_theta(mu_te, float("inf"), seed=2000 + k))
        recs = []
        for d, t, idx in groups:
            R = real_te[idx]
            e_nb = energy_distance(R, nb[idx])
            e_po = energy_distance(R, po[idx])
            if np.isfinite(e_nb) and np.isfinite(e_po):
                recs.append({"dataset": d, "cell_type": t,
                             "a": e_nb, "b": e_po, "diff": e_nb - e_po})
        naive_ps.append(naive_paired(recs)["p_value"])
        cl = clustered_dataset_medians(recs)
        cl_ps.append(cl["wilcoxon_p"])
        sign_ps.append(cl["sign_test_p"])
    naive_ps = np.array(naive_ps); cl_ps = np.array(cl_ps); sign_ps = np.array(sign_ps)
    return {
        "n_redraws": int(n_redraws),
        "naive_p": naive_ps.tolist(),
        "clustered_wilcoxon_p": cl_ps.tolist(),
        "clustered_sign_p": sign_ps.tolist(),
        "naive_p_min": float(naive_ps.min()), "naive_p_max": float(naive_ps.max()),
        "naive_p_median": float(np.median(naive_ps)),
        "clustered_wilcoxon_p_min": float(cl_ps.min()),
        "clustered_wilcoxon_p_max": float(cl_ps.max()),
        "clustered_wilcoxon_p_median": float(np.median(cl_ps)),
        "any_clustered_gt_0.05": bool((cl_ps > 0.05).any()),
    }
