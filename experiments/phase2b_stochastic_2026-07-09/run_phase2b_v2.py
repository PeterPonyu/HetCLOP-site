"""Phase 2b v2: FAIR re-run of "does a LEARNED stochastic decoder beat trivial
matched-noise controls on distributional fidelity?".

The v1 run (run_phase2b.py / phase2b_results.json) was INVALID: the
heteroscedastic Gaussian-NLL decoder collapsed (never fit the mean -- mu Pearson
~0.445 vs 0.742 deterministic on the identical trunk; log_var saturated at the
+10 clamp). That is the documented failure mode of heteroscedastic NLL (Seitzer
et al., ICLR 2022). Comparing a collapsed model to a working control and
concluding "learned stochastic decoding loses" proves nothing.

v2 fixes three things and re-tells the truth:
  * Task 1: the stochastic decoder now uses beta-NLL (beta=0.5, fallback 1.0),
    a mean warm-up (15 MSE-only epochs), a sane log_var clamp [-8, 3], and a
    lower LR on the log_var head. An ACCEPTANCE GATE requires mu Pearson within
    0.02 of the deterministic decoder; if it fails the distributional numbers
    are declared meaningless (not presented as a fair comparison).
  * Task 2: two metrics that independent noise CANNOT fake --
    correlation_attenuation and r_var_dispersion -- replace the discredited
    scale-invariant covariance_preservation as the discriminators (cov kept for
    continuity).
  * Task 3: a 4th variant, matched_noise_per_type, whose residual std is
    estimated per (cell type, gene) on validation cells. It USES the true labels
    (which the learned decoder does not), so it is deliberately advantaged.

4 encoders x 4 decoder variants. All metrics on held-out TEST cells (split==2),
Arm A geometry (encoders see all 2500 panel genes; decoder predicts those same
genes). Fixed seeds throughout.

Reuses decoder_eval.train_decoder / reconstruction_metrics,
centroid_metric_probe.centroid_tightness_ratio_per_dataset, and
src.diagnostics.tightness.per_dataset_tightness_ratio."""
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
from src.evaluation.acceptance import mean_acceptance_gate  # noqa: E402
from src.diagnostics.tightness import per_dataset_tightness_ratio  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402

from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402
from centroid_metric_probe import centroid_tightness_ratio_per_dataset  # noqa: E402

CACHE = PHASE2A / "cache"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DEVICE = "cuda"
SEED = 42
NOISE_SEED = 123  # fixes the matched-noise draws and the stochastic sample
GATE_TOL = 0.02   # stochastic mu Pearson must be within this of deterministic
MIN_VAL_CELLS_PER_TYPE = 20  # per-type control falls back below this
LOGVAR_MIN, LOGVAR_MAX = -8.0, 3.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_geneformer_armA(ordered):
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


def logvar_stats(lv: np.ndarray) -> dict:
    """min/median/max plus fraction of elements pinned at each clamp boundary."""
    lv = np.asarray(lv, dtype=np.float64)
    n = lv.size
    at_min = float(np.mean(lv <= LOGVAR_MIN + 1e-4))
    at_max = float(np.mean(lv >= LOGVAR_MAX - 1e-4))
    return {
        "min": float(lv.min()), "median": float(np.median(lv)), "max": float(lv.max()),
        "frac_at_min_clamp": at_min, "frac_at_max_clamp": at_max,
        "sampling_std_min": float(np.exp(0.5 * lv.min())),
        "sampling_std_median": float(np.exp(0.5 * np.median(lv))),
        "sampling_std_max": float(np.exp(0.5 * lv.max())),
    }


def fit_stochastic_with_gate(Ztr, Ytr, Zva, Yva, Zte, det_pearson, raw_te):
    """Train the beta-NLL decoder and enforce the acceptance gate. Try beta=0.5
    first; if mu Pearson is not within GATE_TOL of the deterministic decoder, try
    beta=1.0. Returns (mu, logvar, meta_dict)."""
    attempts = []
    for beta in (0.5, 1.0):
        mu, lv, val, hist, npar = train_stochastic_decoder(
            Ztr, Ytr, Zva, Yva, Zte, seed=SEED, device=DEVICE,
            logvar_min=LOGVAR_MIN, logvar_max=LOGVAR_MAX,
            beta=beta, warmup_epochs=15, logvar_lr_scale=0.1)
        mu_pearson = reconstruction_metrics(mu, raw_te)["mean_gene_pearson"]
        gap = det_pearson - mu_pearson
        # one-sided underfit guard: FAIL only if the stochastic mean is WORSE than
        # deterministic by > GATE_TOL (see src/evaluation/acceptance.py). A model
        # whose mean fits BETTER than deterministic is healthy and PASSES.
        passed = mean_acceptance_gate(det_pearson, mu_pearson, GATE_TOL)
        lvs = logvar_stats(lv)
        attempts.append({"beta": beta, "mu_pearson": mu_pearson, "gap_vs_det": gap,
                         "gate_passed": bool(passed), "val_loss": val,
                         "params": npar, "logvar_stats": lvs})
        log(f"  [stoch beta={beta}] mu_r={mu_pearson:.4f} (det={det_pearson:.4f} "
            f"gap={gap:+.4f}) gate={'PASS' if passed else 'FAIL'} "
            f"logvar[min={lvs['min']:.2f} med={lvs['median']:.2f} max={lvs['max']:.2f} "
            f"%@min={lvs['frac_at_min_clamp']:.1%} %@max={lvs['frac_at_max_clamp']:.1%}]")
        if passed:
            return mu, lv, {"beta_used": beta, "gate_passed": True, "attempts": attempts}
    # no beta passed -> return the best (highest mu Pearson) attempt, gate failed
    best = max(range(len(attempts)), key=lambda i: attempts[i]["mu_pearson"])
    beta = attempts[best]["beta"]
    mu, lv, val, hist, npar = train_stochastic_decoder(
        Ztr, Ytr, Zva, Yva, Zte, seed=SEED, device=DEVICE,
        logvar_min=LOGVAR_MIN, logvar_max=LOGVAR_MAX,
        beta=beta, warmup_epochs=15, logvar_lr_scale=0.1)
    log(f"  [stoch] GATE FAILED at all beta; reporting best beta={beta} "
        f"(gap={attempts[best]['gap_vs_det']:+.4f}) -- distributional numbers NOT a fair comparison")
    return mu, lv, {"beta_used": beta, "gate_passed": False, "attempts": attempts}


def per_type_residual_std(target_val, pred_val, lab_val, s_g_global, num_genes):
    """Per (cell type, gene) residual std on validation cells; fall back to the
    global per-gene std for types with < MIN_VAL_CELLS_PER_TYPE val cells.
    Returns (std_by_type dict, n_types_ok, n_types_fallback)."""
    resid = (target_val - pred_val)
    std_by_type = {}
    n_ok = n_fb = 0
    for t in np.unique(lab_val):
        m = lab_val == t
        if int(m.sum()) >= MIN_VAL_CELLS_PER_TYPE:
            std_by_type[t] = resid[m].std(axis=0).astype(np.float32)
            n_ok += 1
        else:
            std_by_type[t] = s_g_global
            n_fb += 1
    return std_by_type, n_ok, n_fb


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
    lab_va = labels[va]
    log(f"logX={logX.shape} train={tr.sum()} val={nva} test={te.sum()}")

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

    def centroid_t(pred):
        med, iqr, n = centroid_tightness_ratio_per_dataset(pred, raw_te, lab_te, ds_te)
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
    gate_summary = {}
    for name, Z in enc.items():
        log(f"=== encoder {name} dim={Z.shape[1]} ===")
        Ztr, Yva_t = Z[tr], target[va]
        Ztest_arg = np.vstack([Z[va], Z[te]])
        pred_all, det_val, _, det_np = train_decoder(
            Ztr, target[tr], Z[va], Yva_t, Ztest_arg, seed=SEED, device=DEVICE)
        pred_val = pred_all[:nva]
        det_test = pred_all[nva:]
        det_pearson = reconstruction_metrics(det_test, raw_te)["mean_gene_pearson"]
        log(f"  [det] val_mse={det_val:.4f} params={det_np} r_pt={det_pearson:.4f}")

        # (3a) global matched-noise: per-gene residual std on VALIDATION
        s_g = (target[va] - pred_val).std(axis=0).astype(np.float32)
        eps = np.random.default_rng(NOISE_SEED).standard_normal(det_test.shape).astype(np.float32)
        ctrl_test = (det_test + s_g * eps).astype(np.float32)

        # (3b) per-type matched-noise: per (cell type, gene) residual std on VAL,
        # fall back to global per-gene std for sparse types. Uses TRUE labels.
        std_by_type, n_ok, n_fb = per_type_residual_std(target[va], pred_val, lab_va, s_g, Z.shape[1])
        std_test = np.vstack([std_by_type[t] for t in lab_te]).astype(np.float32)
        n_fb_cells = int(sum(1 for t in lab_te if (lab_va == t).sum() < MIN_VAL_CELLS_PER_TYPE))
        ctrl_pt_test = (det_test + std_test * eps).astype(np.float32)
        log(f"  [ctrl_per_type] types_estimated={n_ok} types_fallback={n_fb} "
            f"test_cells_on_fallback={n_fb_cells}/{len(lab_te)} ({n_fb_cells/len(lab_te):.1%})")

        # (2) learned heteroscedastic beta-NLL with acceptance gate
        mu_test, lv_test, gate = fit_stochastic_with_gate(
            Ztr, target[tr], Z[va], Yva_t, Z[te], det_pearson, raw_te)
        stoch_test = sample_from(mu_test, lv_test, seed=NOISE_SEED)
        gate_summary[name] = gate

        variants = {
            "deterministic": {"decoded": det_test, "point": det_test,
                              "val_loss": det_val, "loss": "MSE", "params": det_np},
            "stochastic_betanll": {"decoded": stoch_test, "point": mu_test,
                                   "val_loss": gate["attempts"][-1]["val_loss"],
                                   "loss": f"betaNLL(beta={gate['beta_used']})",
                                   "params": gate["attempts"][-1]["params"],
                                   "gate": gate},
            "matched_noise": {"decoded": ctrl_test, "point": det_test,
                              "val_loss": det_val, "loss": "none(global control)", "params": 0},
            "matched_noise_per_type": {"decoded": ctrl_pt_test, "point": det_test,
                                       "val_loss": det_val,
                                       "loss": "none(per-type control, uses labels)", "params": 0,
                                       "types_estimated": n_ok, "types_fallback": n_fb,
                                       "test_cells_on_fallback": n_fb_cells},
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
                # B distributional
                "n_groups": dist["n_groups"],
                "r_var_signed_median": dist["r_var_signed"]["median"],
                "r_var_signed_iqr": dist["r_var_signed"]["iqr"],
                "r_var_dispersion": dist["r_var_dispersion"],
                "r_var_abs_log2_median": dist["r_var_abs_log2"]["median"],
                "r_var_abs_log2_iqr": dist["r_var_abs_log2"]["iqr"],
                "energy_distance_median": dist["energy_distance"]["median"],
                "energy_distance_iqr": dist["energy_distance"]["iqr"],
                "cov_preservation_median": dist["cov_preservation"]["median"],
                "cov_preservation_iqr": dist["cov_preservation"]["iqr"],
                "correlation_attenuation_median": dist["correlation_attenuation"]["median"],
                "correlation_attenuation_iqr": dist["correlation_attenuation"]["iqr"],
                # C the trap (report only)
                "decoded_tightness_centroid": centroid_t(v["decoded"]),
                "decoded_tightness_hetclop": hetclop_t(v["decoded"]),
            }
            if vname == "stochastic_betanll":
                row["gate_passed"] = v["gate"]["gate_passed"]
                row["beta_used"] = v["gate"]["beta_used"]
                row["logvar_stats"] = v["gate"]["attempts"][-1]["logvar_stats"]
            if vname == "matched_noise_per_type":
                row["types_estimated"] = v["types_estimated"]
                row["types_fallback"] = v["types_fallback"]
                row["test_cells_on_fallback"] = v["test_cells_on_fallback"]
            rows.append(row)
            log(f"  [{vname:22s}] r_pt={row['recon_pearson_point']:.4f} "
                f"r_smp={row['recon_pearson_sampled']:.4f} | "
                f"r_var={row['r_var_signed_median']:.3f} disp={row['r_var_dispersion']:.3f} "
                f"energy={row['energy_distance_median']:.3f} "
                f"corr_att={row['correlation_attenuation_median']:.3f} "
                f"cov={row['cov_preservation_median']:.3f} | "
                f"tight(centroid)={row['decoded_tightness_centroid']['median']:.3f} "
                f"tight(het)={row['decoded_tightness_hetclop']['median']:.3f}")

    out = {
        "purpose": "Phase 2b v2 (FAIR re-run): learned beta-NLL stochastic decoder vs global AND "
                   "per-type matched-noise controls on distributional fidelity (Arm A, test cells only). "
                   "Supersedes phase2b_results.json, whose stochastic arm collapsed (invalid null).",
        "setting": "Arm A: encoders see 2500 panel genes; decoder predicts those same genes",
        "n_test_cells": int(te.sum()), "n_val_cells": nva, "n_train_cells": int(tr.sum()),
        "encoders": list(enc.keys()), "seed": SEED, "noise_seed": NOISE_SEED,
        "acceptance_gate": {
            "rule": "stochastic mu Pearson within %.2f of deterministic; else its distributional "
                    "metrics are NOT a fair comparison" % GATE_TOL,
            "tol": GATE_TOL, "per_encoder": gate_summary,
        },
        "notes": {
            "fixes_vs_v1": "beta-NLL (beta=0.5, fallback 1.0), 15-epoch MSE mean warm-up, "
                           "log_var clamp [-8,3], 0.1x LR on log_var head",
            "discriminating_metrics": "correlation_attenuation and r_var_dispersion (independent "
                                      "noise CANNOT fake these). cov_preservation is scale-invariant "
                                      "and fails to discriminate -- reported for continuity only.",
            "matched_noise_per_type": "per (cell type, gene) residual std on val, USES true labels; "
                                      "deliberately advantaged vs the learned decoder",
            "tightness": "metric C reported ONLY to demonstrate the C3 trap, never a success criterion",
            "energy_subsample": 300, "cov_top_k": 200, "min_cells_per_group": 20,
            "min_val_cells_per_type": MIN_VAL_CELLS_PER_TYPE,
        },
        "results": rows,
    }
    (RESULTS / "phase2b_v2_results.json").write_text(json.dumps(out, indent=2))
    log(f"=== done in {time.time()-t0:.0f}s -> {RESULTS / 'phase2b_v2_results.json'} ===")


if __name__ == "__main__":
    main()
