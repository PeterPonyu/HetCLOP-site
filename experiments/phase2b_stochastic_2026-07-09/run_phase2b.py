"""Phase 2b: does a LEARNED stochastic decoder beat a TRIVIAL matched-noise
control on distributional fidelity, and does encoder identity start to matter
once decoding is stochastic?

Setting: Arm A geometry (encoders see all 2500 panel genes; decoder predicts
those same 2500 genes) -- the analogue of a real generation pipeline. All metrics
on held-out TEST cells (split==2).

4 encoders x 3 decoder variants:
  encoders : pca_512, randproj_512 (fixed seed), scgpt_512 (cached),
             geneformer (cached per-dataset npz, Arm A)
  variants : (1) deterministic MLP (MSE point estimate)     -> the deterministic baseline
             (2) learned heteroscedastic Gaussian (NLL, SAMPLED at eval)
             (3) trivial matched-noise control: det pred + per-gene residual std
                 (estimated on VALIDATION) * eps  -- restores marginal variance
                 by construction, learns nothing about how variance depends on
                 the cell. THE NULL variant (2) must beat.

Metrics (all on test cells):
  A pointwise   : mean per-gene Pearson r, MSE (stochastic reported on mu AND sampled)
  B distributional (per dataset x cell type, >=20 cells, median+IQR across groups):
      r_var (per-gene within-type variance ratio), within-type energy distance,
      gene-gene covariance preservation  <- the metric independent noise CANNOT fake
  C the C3 trap (report only): decoded tightness under BOTH estimators; expected
      to INFLATE under sampling for BOTH the learned and the trivial decoder.

Fixed seeds throughout. Reuses decoder_eval.train_decoder,
decoder_eval.reconstruction_metrics, parent_metric_probe.parent_tightness_ratio_
per_dataset, and src.diagnostics.tightness.per_dataset_tightness_ratio.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HETCLOP_ROOT = HERE.parent.parent
PHASE2A = HETCLOP_ROOT / "experiments" / "phase2a_decoder_2026-07-08"
sys.path.insert(0, str(HETCLOP_ROOT))
sys.path.insert(0, str(PHASE2A))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.architecture.stochastic_decoder import sample_from, train_stochastic_decoder  # noqa: E402
from src.diagnostics.tightness import per_dataset_tightness_ratio  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402

from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402
from parent_metric_probe import parent_tightness_ratio_per_dataset  # noqa: E402

CACHE = PHASE2A / "cache"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DEVICE = "cuda"
SEED = 42
NOISE_SEED = 123  # fixes both the matched-noise draw and the stochastic sample


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_geneformer_armA(ordered):
    """Concatenate cached per-dataset Arm A Geneformer embeddings (same
    cache-path resolution as f2_under_parent_metric.py). Returns None if any
    dataset's cache is missing/mismatched -> caller EXCLUDES loudly."""
    cache_dirs = [
        (HETCLOP_ROOT / "experiments/real_phase0_2026-07-05/results", "{fm}_{ds}.npz"),
        (HETCLOP_ROOT / "experiments/real_phase0_extended_2026-07-08/results", "{fm}_{ds}.npz"),
        (HETCLOP_ROOT / "data/embeddings/fm_emb", "{fm}_{fm}_{ds}.npz"),
    ]
    parts = []
    for name in ordered:
        p = None
        for d, pat in cache_dirs:
            cand = d / pat.format(fm="geneformer", ds=name)
            if cand.exists():
                p = cand
                break
        if p is None:
            return None
        parts.append(np.load(p)["X"])
    return np.vstack(parts).astype(np.float32)


def main() -> None:
    t0 = time.time()
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    ordered = list(meta["ordered"].astype(str))
    tr, va, te = split == 0, split == 1, split == 2
    assert tr.sum() and va.sum() and te.sum(), "empty split mask"
    nva = int(va.sum())

    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    target = logX  # Arm A: decode the same 2500 panel genes
    raw_te, lab_te, ds_te = logX[te], labels[te], dataset_ids[te]
    log(f"logX={logX.shape} train={tr.sum()} val={nva} test={te.sum()}")

    # HetCLOP per-dataset estimator needs each dataset's raw reference; use the
    # shared 2500-gene panel sliced per dataset (self-consistent with the Arm A
    # decode target, and no dependency on the full real-corpus assembly). This
    # affects only metric C (the trap demonstration), never a success criterion.
    raw_by_ds_test = {d: logX[te][ds_te == d] for d in np.unique(ds_te)}

    # --- Encoders (4) --------------------------------------------------------
    rng = np.random.default_rng(0)
    enc: dict[str, np.ndarray] = {}
    log("fitting PCA-512 on train cells")
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(logX[tr]).transform(logX).astype(np.float32)
    W = rng.normal(0, 1 / np.sqrt(512), size=(logX.shape[1], 512)).astype(np.float32)
    enc["randproj_512"] = (logX @ W).astype(np.float32)
    p = CACHE / "enc_A_scgpt.npy"
    if p.exists():
        z = np.load(p)
        if z.shape[0] == logX.shape[0]:
            enc["scgpt_512"] = z.astype(np.float32)
            log(f"loaded cached scgpt_512 {z.shape}")
        else:
            log(f"EXCLUDED scgpt_512: row mismatch {z.shape[0]} != {logX.shape[0]}")
    else:
        log("EXCLUDED scgpt_512: cache enc_A_scgpt.npy missing")
    gf = load_geneformer_armA(ordered)
    if gf is not None and gf.shape[0] == logX.shape[0]:
        enc["geneformer"] = gf
        log(f"loaded cached geneformer (Arm A) {gf.shape}")
    else:
        log("EXCLUDED geneformer: cached per-dataset Arm A embeddings missing/mismatched")

    def parent_t(pred):
        med, iqr, n = parent_tightness_ratio_per_dataset(pred, raw_te, lab_te, ds_te)
        return {"median": med, "iqr": list(iqr), "n": n}

    def hetclop_t(pred):
        try:
            med, iqr, vals = per_dataset_tightness_ratio(pred, raw_by_ds_test, lab_te, ds_te)
            return {"median": med, "iqr": list(iqr), "n": int(len(vals))}
        except Exception as e:  # noqa: BLE001
            return {"median": float("nan"), "iqr": [float("nan"), float("nan")], "n": 0, "err": str(e)}

    def distributional(pred):
        return aggregate_over_groups(raw_te, pred, lab_te, ds_te,
                                     min_cells=20, energy_subsample=300, top_k=200, seed=0)

    rows = []
    for name, Z in enc.items():
        log(f"=== encoder {name} dim={Z.shape[1]} ===")
        Ztr, Yva_t = Z[tr], target[va]
        # concat val+test so we get deterministic predictions on BOTH in one call
        Ztest_arg = np.vstack([Z[va], Z[te]])
        pred_all, det_val, _, det_np = train_decoder(
            Ztr, target[tr], Z[va], Yva_t, Ztest_arg, seed=SEED, device=DEVICE)
        pred_val = pred_all[:nva]
        det_test = pred_all[nva:]
        log(f"  [det] val_mse={det_val:.4f} params={det_np}")

        # (3) trivial matched-noise control: per-gene residual std on VALIDATION
        s_g = (target[va] - pred_val).std(axis=0).astype(np.float32)
        eps = np.random.default_rng(NOISE_SEED).standard_normal(det_test.shape).astype(np.float32)
        ctrl_test = (det_test + s_g * eps).astype(np.float32)

        # (2) learned heteroscedastic Gaussian
        mu_test, lv_test, stoch_val, _, stoch_np = train_stochastic_decoder(
            Ztr, target[tr], Z[va], Yva_t, Z[te], seed=SEED, device=DEVICE)
        stoch_test = sample_from(mu_test, lv_test, seed=NOISE_SEED)
        log(f"  [stoch] val_nll={stoch_val:.4f} params={stoch_np} "
            f"logvar[min={lv_test.min():.2f},max={lv_test.max():.2f}]")

        variants = {
            "deterministic": {"decoded": det_test, "point": det_test,
                              "val_loss": det_val, "loss": "MSE", "params": det_np},
            "stochastic": {"decoded": stoch_test, "point": mu_test,
                           "val_loss": stoch_val, "loss": "GaussianNLL", "params": stoch_np},
            "matched_noise": {"decoded": ctrl_test, "point": det_test,
                              "val_loss": det_val, "loss": "none(control)", "params": 0},
        }

        for vname, v in variants.items():
            recA_point = reconstruction_metrics(v["point"], raw_te)
            recA_sampled = reconstruction_metrics(v["decoded"], raw_te)
            dist = distributional(v["decoded"])
            row = {
                "encoder": name, "dim": int(Z.shape[1]), "variant": vname,
                "loss": v["loss"], "params": v["params"], "val_loss": v["val_loss"],
                # A pointwise
                "recon_pearson_point": recA_point["mean_gene_pearson"],
                "recon_mse_point": recA_point["mse"],
                "recon_pearson_sampled": recA_sampled["mean_gene_pearson"],
                "recon_mse_sampled": recA_sampled["mse"],
                # B distributional (the success criteria)
                "n_groups": dist["n_groups"],
                "r_var_signed_median": dist["r_var_signed"]["median"],
                "r_var_signed_iqr": dist["r_var_signed"]["iqr"],
                "r_var_abs_log2_median": dist["r_var_abs_log2"]["median"],
                "r_var_abs_log2_iqr": dist["r_var_abs_log2"]["iqr"],
                "energy_distance_median": dist["energy_distance"]["median"],
                "energy_distance_iqr": dist["energy_distance"]["iqr"],
                "cov_preservation_median": dist["cov_preservation"]["median"],
                "cov_preservation_iqr": dist["cov_preservation"]["iqr"],
                # C the trap (report only)
                "decoded_tightness_parent": parent_t(v["decoded"]),
                "decoded_tightness_hetclop": hetclop_t(v["decoded"]),
            }
            rows.append(row)
            log(f"  [{vname:13s}] r_pt={row['recon_pearson_point']:.4f} "
                f"r_smp={row['recon_pearson_sampled']:.4f} | "
                f"r_var={row['r_var_signed_median']:.3f} "
                f"|log2|={row['r_var_abs_log2_median']:.3f} "
                f"energy={row['energy_distance_median']:.3f} "
                f"cov={row['cov_preservation_median']:.3f} | "
                f"tight(parent)={row['decoded_tightness_parent']['median']:.3f} "
                f"tight(het)={row['decoded_tightness_hetclop']['median']:.3f}")

    out = {
        "purpose": "Phase 2b: learned stochastic decoder vs trivial matched-noise control on "
                   "distributional fidelity (Arm A geometry, test cells only)",
        "setting": "Arm A: encoders see 2500 panel genes; decoder predicts those same genes",
        "n_test_cells": int(te.sum()), "n_val_cells": nva, "n_train_cells": int(tr.sum()),
        "encoders": list(enc.keys()), "seed": SEED, "noise_seed": NOISE_SEED,
        "notes": {
            "success_criteria": "distributional metrics B only; tightness (C) reported to "
                                "DEMONSTRATE the C3 trap, never as a success criterion",
            "cov_preservation": "the metric a trivial per-gene noise injection should FAIL",
            "hetclop_tightness_raw_reference": "2500-gene panel sliced per dataset (self-consistent "
                                               "with Arm A decode target)",
            "energy_subsample": 300, "cov_top_k": 200, "min_cells_per_group": 20,
        },
        "results": rows,
    }
    (RESULTS / "phase2b_results.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'phase2b_results.json'} ===")


if __name__ == "__main__":
    main()
