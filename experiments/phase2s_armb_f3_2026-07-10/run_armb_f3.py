"""Phase 2s (Tier 3) -- Arm-B replication of F3.

Manuscript limitation being closed: "F3 runs in Arm A only, which rewards linear encoders through
near-invertibility; its Arm-B replication is left to future work."

F3 makes two claims:
  (i)  deterministic decoding destroys within-type variance (it emits E[Y|Z], one output per
       latent, so the decoded within-type variance collapses);
  (ii) neither tested noise family is calibrated -- Gaussian-on-log1p noise UNDER-inflates.

Claim (i) should be geometry-independent: it is a property of deterministic decoding, not of the
arm. Claim (ii)'s DIRECTION is arm-dependent and we already know it flips for the count model
(Arm A: Poisson over-inflates; Arm B: Poisson under-inflates, F4's factor (a)). What has never
been measured is the Gaussian arm in Arm B. This run measures both.

Setting: Arm B. PCA-512 is fitted on the G_in genes only; the decoder predicts the disjoint G_out
genes that no encoder ever saw. Variants, all on log1p(G_out):

  deterministic   -- MSE decoder output E[Y|Z]                       (no noise at all)
  matched_noise   -- deterministic mean + per-gene residual Gaussian noise, LABEL-FREE
                     (the trivial control: it can restore variance without learning anything)

Reported per (dataset x cell type) group: the EXPRESSION-RESTRICTED signed variance ratio
var(decoded)/var(real), whose direction is meaningful (unlike the unrestricted statistic, which
sparsity pins at 1.0). Claim gates:

  det_destroys_variance   : deterministic signed r_var is far below 1 in every seed.
  gaussian_under_inflates : matched_noise signed r_var is below 1 in every seed.

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
sys.path.insert(0, str(ROOT / "experiments" / "phase2g_armb_f4_2026-07-09"))

import phase2j_lib as L  # noqa: E402
from decoder_eval import train_decoder  # noqa: E402
from phase2g_lib import assert_log1p_scale  # noqa: E402
from src.evaluation.distributional import r_var_per_gene_expressed  # noqa: E402

CACHE = ROOT / "experiments" / "phase2a_decoder_2026-07-08" / "cache"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
MIN_CELLS = 20


def group_masks(labels, dataset_ids, min_cells=MIN_CELLS):
    out = []
    for ds in np.unique(dataset_ids):
        m_ds = dataset_ids == ds
        for t in np.unique(labels[m_ds]):
            m = m_ds & (labels == t)
            if m.sum() >= min_cells:
                out.append((str(ds), str(t), m))
    return out


def signed_rvar_by_group(real, decoded, groups):
    vals = []
    for _, _, m in groups:
        r = r_var_per_gene_expressed(real[m], decoded[m])["expressed"]
        v = r["median_signed_ratio"]
        if np.isfinite(v):
            vals.append(float(v))
    return np.array(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    L.log("=== Tier 3: Arm-B replication of F3 (deterministic + Gaussian matched noise) ===")
    panel = json.loads((CACHE / "panel.json").read_text())
    gin = np.asarray(panel["gin_idx"], dtype=int)
    gout = np.asarray(panel["gout_idx"], dtype=int)
    assert np.intersect1d(gin, gout).size == 0, "G_in and G_out must be disjoint"
    L.log(f"  Arm B: encoder sees {gin.size} G_in genes; decoder predicts {gout.size} unseen G_out genes")

    data = L.load_corpus()
    logX = data["logX"]
    tr, va, te = data["tr"], data["va"], data["te"]
    lab, ds = data["labels"], data["dataset_ids"]

    Xin = logX[:, gin]
    Yout = logX[:, gout]                       # log1p target: the genes no encoder saw
    real_te = Yout[te]
    groups = group_masks(lab[te], ds[te])
    L.log(f"  {len(groups)} (dataset x cell type) test groups")

    per_seed = {}
    for seed in SEEDS:
        L.log(f"--- seed {seed} ---")
        Z = PCA(n_components=512, random_state=0).fit(Xin[tr]).transform(Xin).astype(np.float32)
        # ONE training run: predict val+test together, then split (retraining twice would double
        # the GPU cost for identical weights).
        nva = int(va.sum()) if va.dtype == bool else len(va)
        Z_arg = np.vstack([Z[va], Z[te]])
        pred, _, _, _ = train_decoder(Z[tr], Yout[tr], Z[va], Yout[va], Z_arg,
                                      seed=seed, device=args.device)
        pred = np.asarray(pred, dtype=np.float32)
        det_va, det_te = pred[:nva], pred[nva:]

        # label-free matched-noise control: per-gene residual std, in LOG1P space (spec §15 bug)
        resid_std = (Yout[va] - det_va).std(axis=0)
        rng = np.random.default_rng(seed + 1)
        matched = det_te + rng.normal(0.0, 1.0, size=det_te.shape).astype(np.float32) * resid_std
        assert_log1p_scale(matched, real_te, "matched_noise")   # guards the count-space bug

        dv = signed_rvar_by_group(real_te, det_te, groups)
        mv = signed_rvar_by_group(real_te, matched, groups)
        per_seed[str(seed)] = {
            "deterministic_signed_rvar_median": float(np.median(dv)),
            "matched_noise_signed_rvar_median": float(np.median(mv)),
            "deterministic_frac_groups_below_one": float(np.mean(dv < 1.0)),
            "matched_noise_frac_groups_below_one": float(np.mean(mv < 1.0)),
            "n_groups": int(dv.size),
        }
        s = per_seed[str(seed)]
        L.log(f"  deterministic signed r_var median = {s['deterministic_signed_rvar_median']:.4f} "
              f"({100*s['deterministic_frac_groups_below_one']:.1f}% of groups < 1)")
        L.log(f"  matched_noise  signed r_var median = {s['matched_noise_signed_rvar_median']:.4f} "
              f"({100*s['matched_noise_frac_groups_below_one']:.1f}% of groups < 1)")

    det_med = [per_seed[str(s)]["deterministic_signed_rvar_median"] for s in SEEDS]
    mat_med = [per_seed[str(s)]["matched_noise_signed_rvar_median"] for s in SEEDS]
    gates = {
        "det_destroys_variance_every_seed": bool(all(v < 0.5 for v in det_med)),
        "gaussian_under_inflates_every_seed": bool(all(v < 1.0 for v in mat_med)),
        "noise_restores_some_variance_every_seed": bool(all(m > d for m, d in zip(mat_med, det_med))),
    }

    out = {
        "purpose": "Arm-B replication of F3: does deterministic decoding still destroy within-type "
                   "variance, and does Gaussian matched noise still under-inflate, when the decoder "
                   "predicts genes no encoder ever saw?",
        "arm": "B (encoder sees G_in only; decoder predicts disjoint G_out)",
        "g_in": int(gin.size), "g_out": int(gout.size), "seeds": SEEDS,
        "statistic": "expression-restricted signed r_var = var(decoded)/var(real), median over genes then groups",
        "per_seed": per_seed,
        "deterministic_median_across_seeds": float(np.median(det_med)),
        "matched_noise_median_across_seeds": float(np.median(mat_med)),
        "claims": gates,
    }
    (RESULTS / "armb_f3.json").write_text(json.dumps(out, indent=2))
    L.log("CLAIM GATES:")
    for k, v in gates.items():
        L.log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    L.log(f"=== done -> {RESULTS / 'armb_f3.json'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
