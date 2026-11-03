"""Phase 2d E3 -- ERROR BARS on the two Phase 2a headline claims.

Phase 2a reported, single-seed, Arm B geometry (encoders see G_in=1250 genes;
decoder predicts G_out=1250 DISJOINT genes no encoder saw), two headline claims:
  (a) every encoder decodes G_out within ~0.003 test mean-gene Pearson of every
      other (encoder identity barely matters in the circularity-broken arm);
  (b) deterministic decoding retains only 12-26% of real within-type variance
      (signed r_var median ~0.117-0.259).
Both were SINGLE SEED, no error bars.

E3 puts error bars on both. Encoders (all restricted to G_in): oracle_1250,
pca_512, pca_50, randproj_512, randproj_50, scgpt_512, geneformer -- the exact Arm
B encoder set from armb_512_probe.py. Encoders are FIXED; 3 seeds drive only the
decoder init/shuffle. Per encoder x seed: deterministic MLP decoder G_in->G_out,
then (a) test mean-gene Pearson vs Yout[te], and (b) signed r_var median of the
deterministic decoder over the 314 (dataset x cell type) groups (metric on
log1p(G_out); real = Yout[te] = log1p(gout counts)). energy uses <=300 cells/side.

Reported as mean +/- sd across the 3 seeds. Then, explicitly:
  * BETWEEN-ENCODER Pearson spread (max-min of per-encoder means) vs the mean
    ACROSS-SEED sd. If the between-encoder spread is SMALLER than the across-seed
    sd, the 0.003 encoder-irrelevance claim is STRENGTHENED (encoder choice moves
    the number less than reshuffling the decoder does) -- reported that way.

No re-embedding, no tightness metric (not needed for these two claims), no
assemble_real_corpus dependency. Reuses only cached logX + the panel gene split
and the shared decoder/eval/distributional code.

"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2B = HERE.parent / "phase2b_stochastic_2026-07-09"
PHASE2A = HERE.parent / "phase2a_decoder_2026-07-08"
sys.path.insert(0, str(PHASE2B))
sys.path.insert(0, str(PHASE2A))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from run_phase2b_v2 import CACHE, DEVICE  # noqa: E402
from src.evaluation.distributional import aggregate_over_groups  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
# order matches armb_512_probe.py so encoder identities line up with Phase 2a
ENCODER_ORDER = ["oracle_1250", "pca_512", "randproj_512", "pca_50",
                 "randproj_50", "scgpt_512", "geneformer"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_armb_encoders(Xin, tr):
    """Arm B encoders on G_in, byte-for-byte armb_512_probe.py's construction
    (rng seed 0; randproj_512 drawn before randproj_50)."""
    enc = {}
    enc["oracle_1250"] = Xin
    enc["pca_512"] = PCA(n_components=512, random_state=0).fit(
        Xin[tr]).transform(Xin).astype(np.float32)
    enc["pca_50"] = PCA(n_components=50, random_state=0).fit(
        Xin[tr]).transform(Xin).astype(np.float32)
    rng = np.random.default_rng(0)
    for dim in (512, 50):
        W = rng.normal(0, 1 / np.sqrt(dim), size=(Xin.shape[1], dim)).astype(np.float32)
        enc[f"randproj_{dim}"] = (Xin @ W).astype(np.float32)
    for nm, f in [("scgpt_512", "enc_B_scgpt.npy"),
                  ("geneformer", "enc_B_geneformer.npy")]:
        p = CACHE / f
        if p.exists():
            enc[nm] = np.load(p).astype(np.float32)
            log(f"loaded cached Arm B {nm}: {enc[nm].shape}")
        else:
            log(f"MISSING cached Arm B {nm} ({f}) -> excluded")
    return enc


def main() -> None:
    t0 = time.time()
    panel = json.loads((CACHE / "panel.json").read_text())
    gin_idx = np.array(panel["gin_idx"])
    gout_idx = np.array(panel["gout_idx"])

    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    labels = meta["labels"].astype(str)
    dataset_ids = meta["dataset_ids"].astype(str)
    split = meta["split"]
    tr, va, te = split == 0, split == 1, split == 2
    nva = int(va.sum())

    logX = np.load(CACHE / "logX_full.npy").astype(np.float32)
    Xin = logX[:, gin_idx].astype(np.float32)      # encoders may see
    Yout = logX[:, gout_idx].astype(np.float32)    # decoders must predict
    lab_te, ds_te = labels[te], dataset_ids[te]
    Yout_te = Yout[te]
    log(f"Arm B G_in={Xin.shape} G_out={Yout.shape} test={int(te.sum())}")

    enc = build_armb_encoders(Xin, tr)
    encoders = [e for e in ENCODER_ORDER if e in enc]

    # per_enc[name] = {"pearson": [...seeds], "rvar_signed": [...seeds], ...}
    per_enc = {}
    for name in encoders:
        Z = enc[name]
        pearsons, rvars, rvar_iqrs, energies = [], [], [], []
        expr_rvars = []  # expression-restricted context (deterministic mean is
        # mean-regressed so unrestricted signed r_var is NOT sparsity-pinned here;
        # restricted value reported for continuity with E1/E4)
        for seed in SEEDS:
            pred, vloss, _, npar = train_decoder(
                Z[tr], Yout[tr], Z[va], Yout[va], Z[te], seed=seed, device=DEVICE)
            rec = reconstruction_metrics(pred, Yout_te)
            agg = aggregate_over_groups(
                Yout_te, pred, lab_te, ds_te,
                min_cells=20, energy_subsample=300, top_k=200, seed=0)
            pearsons.append(float(rec["mean_gene_pearson"]))
            rvars.append(float(agg["r_var_signed"]["median"]))
            rvar_iqrs.append([float(x) for x in agg["r_var_signed"]["iqr"]])
            expr_rvars.append(float(agg["r_var_signed_expressed"]["median"]))
            energies.append(float(agg["energy_distance"]["median"]))
            log(f"  {name:14s} seed={seed} dim={Z.shape[1]:5d} "
                f"pearson={rec['mean_gene_pearson']:.4f} "
                f"signed_rvar={agg['r_var_signed']['median']:.4f} "
                f"expr_signed_rvar={agg['r_var_signed_expressed']['median']:.4f} "
                f"n_groups={agg['n_groups']}")
        per_enc[name] = {
            "dim": int(Z.shape[1]),
            "pearson_per_seed": pearsons,
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_sd": float(np.std(pearsons, ddof=1)),
            "rvar_signed_per_seed": rvars,
            "rvar_signed_mean": float(np.mean(rvars)),
            "rvar_signed_sd": float(np.std(rvars, ddof=1)),
            "rvar_signed_iqr_per_seed": rvar_iqrs,
            "rvar_signed_expressed_per_seed": expr_rvars,
            "rvar_signed_expressed_mean": float(np.mean(expr_rvars)),
            "rvar_signed_expressed_sd": float(np.std(expr_rvars, ddof=1)),
            "energy_per_seed": energies,
        }
        log(f"  [{name}] pearson={per_enc[name]['pearson_mean']:.4f}"
            f"+/-{per_enc[name]['pearson_sd']:.4f}  "
            f"signed_rvar={per_enc[name]['rvar_signed_mean']:.4f}"
            f"+/-{per_enc[name]['rvar_signed_sd']:.4f}")

    # --- Claim (a): encoder-irrelevance vs seed noise -----------------------
    means = [per_enc[n]["pearson_mean"] for n in encoders]
    between_spread = float(max(means) - min(means))
    mean_across_seed_sd = float(np.mean([per_enc[n]["pearson_sd"] for n in encoders]))
    max_across_seed_sd = float(np.max([per_enc[n]["pearson_sd"] for n in encoders]))
    # exclude oracle (uncompressed upper bound) for a "real encoders" view too
    real = [n for n in encoders if n != "oracle_1250"]
    real_means = [per_enc[n]["pearson_mean"] for n in real]
    real_between_spread = float(max(real_means) - min(real_means))

    claim_a = {
        "between_encoder_pearson_spread_all": between_spread,
        "between_encoder_pearson_spread_excl_oracle": real_between_spread,
        "mean_across_seed_sd": mean_across_seed_sd,
        "max_across_seed_sd": max_across_seed_sd,
        "spread_smaller_than_mean_seed_sd_all": bool(between_spread < mean_across_seed_sd),
        "spread_smaller_than_mean_seed_sd_excl_oracle": bool(
            real_between_spread < mean_across_seed_sd),
        "interpretation": (
            "If the between-encoder spread is SMALLER than the across-seed sd, "
            "encoder choice moves the decoded Pearson less than reshuffling the "
            "decoder does -> the 0.003 encoder-irrelevance claim is STRENGTHENED."),
    }
    claim_b = {
        "rvar_signed_range_over_encoders_mean": [
            float(min(per_enc[n]["rvar_signed_mean"] for n in encoders)),
            float(max(per_enc[n]["rvar_signed_mean"] for n in encoders))],
        "note": "Phase 2a single-seed reported signed r_var median 0.117-0.259; "
                "means +/- sd per encoder above put error bars on the retained "
                "within-type variance fraction of deterministic decoding.",
    }
    log(f"[CLAIM a] between-encoder Pearson spread (all)={between_spread:.5f} "
        f"(excl oracle={real_between_spread:.5f}); mean across-seed sd="
        f"{mean_across_seed_sd:.5f}; strengthened(all)="
        f"{claim_a['spread_smaller_than_mean_seed_sd_all']} "
        f"excl_oracle={claim_a['spread_smaller_than_mean_seed_sd_excl_oracle']}")

    out = {
        "purpose": "Phase 2d E3: error bars (3 seeds, decoder init/shuffle) on the "
                   "two Phase 2a Arm B headline claims -- encoder-irrelevance in "
                   "Pearson and the retained within-type variance fraction (signed "
                   "r_var) of deterministic decoding.",
        "geometry": "Arm B: encoders see G_in=1250; decoder predicts G_out=1250 "
                    "disjoint genes; metric on log1p(G_out).",
        "seeds": SEEDS, "encoders": encoders,
        "g_in": int(len(gin_idx)), "g_out": int(len(gout_idx)),
        "energy_subsample": 300,
        "per_encoder": per_enc,
        "claim_a_encoder_irrelevance": claim_a,
        "claim_b_retained_variance": claim_b,
    }
    (RESULTS / "e3_error_bars.json").write_text(json.dumps(out, indent=2))
    log(f"=== E3 done in {time.time()-t0:.0f}s -> {RESULTS / 'e3_error_bars.json'} ===")


if __name__ == "__main__":
    main()
