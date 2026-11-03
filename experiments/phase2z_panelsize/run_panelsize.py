"""Phase 2z -- F2 panel-size sensitivity (review panel 2026-07-19, SYNTHESIS #10 context).

F2's near-null (six encoders span ~0.0045 decoded Pearson) is measured in Arm B:
encoders see HALF of the 2,500-gene shared panel (1,250 G_in genes, ~6% of FM
native panels) and decode the disjoint 1,250 G_out genes (circularity broken).
The referee caveat: does the near-null survive at other panel sizes?

This script recomputes the Arm-B reconstruction Pearson for the three encoders
that need NO refitting of the encoder itself -- PCA-50, random-projection-50,
and the ORACLE (identity on the input genes) -- at input-panel sizes
{250, 625, 1250} genes: nested prefixes of the original G_in set (G_in was drawn
by fixed seed 1234, so prefixes are deterministic, effectively random subsets).
G_out is held FIXED at the original 1,250 genes so the readout is comparable
across sizes; circularity stays broken at every size. Sizes are reported as
INPUT genes; in the review's "total panel" terms this ladder is
{500, 1250, 2500}. Larger ladders (input 2,500 / panel 5,000) are INFEASIBLE on
this corpus: the shared panel caps at 2,500 genes, and circularity-broken
disjointness leaves at most 1,250 input genes.

Decoder: the exact train_decoder/reconstruction_metrics of phase2a
(decoder_eval.py, MLP 1024-2048, 50 epochs, early stop, seed 42), on CPU.
The size-1250 rows are anchors: they must reproduce phase2a's Arm-B pca /
randproj and oracle_control's oracle Pearson.

Boundary rule: all code under HetCLOP/; phase2a code is imported, not modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHASE2A = REPO / "experiments" / "phase2a_decoder_2026-07-08"
for p in (str(REPO), str(PHASE2A)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.decomposition import PCA  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

CACHE = PHASE2A / "cache"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

INPUT_SIZES = [250, 625, 1250]   # nested prefixes of G_in; 1250 = the paper's Arm B
DEVICE = "cuda"   # phase2a's own device; the decoder fits are minutes on GPU, hours on CPU


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def randproj(X, n_comp=50, seed=7):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((X.shape[1], n_comp)).astype(np.float32) / np.sqrt(X.shape[1])
    return (X @ R).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        return merge_shards()

    t0 = time.time()
    panel = json.loads((CACHE / "panel.json").read_text())
    gin_idx = np.array(panel["gin_idx"], dtype=int)
    gout_idx = np.array(panel["gout_idx"], dtype=int)
    meta = np.load(CACHE / "meta.npz", allow_pickle=True)
    split = meta["split"]
    tr, va, te = split == 0, split == 1, split == 2
    logX = np.load(CACHE / "logX_full.npy")
    target = logX[:, gout_idx].astype(np.float32)
    log(f"corpus {logX.shape}; G_out fixed at {len(gout_idx)} genes; "
        f"train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")

    # anchors from the original runs
    anchors = {}
    p2a = json.loads((PHASE2A / "results" / "phase2a_results.json").read_text())
    for r in p2a["results"]:
        if r["arm"] == "B" and not r.get("capacity_control"):
            anchors[f"armB_{r['encoder']}"] = r["reconstruction"]["mean_gene_pearson"]
    oc = json.loads((PHASE2A / "results" / "oracle_control.json").read_text())
    for r in oc["results"]:
        anchors[r["encoder"]] = r["reconstruction"]["mean_gene_pearson"]
    log(f"anchors: {anchors}")

    jobs = [(k, e) for k in INPUT_SIZES for e in ("oracle_identity", "pca_50", "randproj_50")]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]
    log(f"shard {args.shard}/{args.n_shards}: jobs {jobs}")

    rows = []
    for k, enc_name in jobs:
        gin_k = gin_idx[:k]
        X_in = logX[:, gin_k].astype(np.float32)
        if enc_name == "oracle_identity":
            Z = X_in
        elif enc_name == "pca_50":
            Z = PCA(n_components=50, random_state=0).fit(X_in[tr]).transform(X_in).astype(np.float32)
        else:
            Z = randproj(X_in, 50, seed=7)
        pred, vloss, hist, nparams = train_decoder(
            Z[tr], target[tr], Z[va], target[va], Z[te], device=DEVICE)
        rec = reconstruction_metrics(pred, target[te])
        rows.append({
            "input_genes": int(k), "encoder": enc_name, "embed_dim": int(Z.shape[1]),
            "recon_pearson": rec["mean_gene_pearson"], "val_mse": vloss,
            "best_epoch": int(max(hist, key=lambda h: -h["val"])["epoch"]) if hist else None,
        })
        log(f"  input={k:5d} {enc_name:16s} pearson={rec['mean_gene_pearson']:.4f} "
            f"val_mse={vloss:.4f}")

    part = RESULTS / f"panelsize_shard{args.shard}.json"
    part.write_text(json.dumps({"shard": args.shard, "rows": rows}, indent=2))
    log(f"shard done -> {part} ({time.time()-t0:.0f}s)")
    return 0


def merge_shards() -> int:
    t0 = time.time()
    panel = json.loads((CACHE / "panel.json").read_text())
    gout_idx = np.array(panel["gout_idx"], dtype=int)
    anchors = {}
    p2a = json.loads((PHASE2A / "results" / "phase2a_results.json").read_text())
    for r in p2a["results"]:
        if r["arm"] == "B" and not r.get("capacity_control"):
            anchors[f"armB_{r['encoder']}"] = r["reconstruction"]["mean_gene_pearson"]
    oc = json.loads((PHASE2A / "results" / "oracle_control.json").read_text())
    for r in oc["results"]:
        anchors[r["encoder"]] = r["reconstruction"]["mean_gene_pearson"]

    rows = []
    for f in sorted(RESULTS.glob("panelsize_shard*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    have = {(r["input_genes"], r["encoder"]) for r in rows}
    need = {(k, e) for k in INPUT_SIZES for e in ("oracle_identity", "pca_50", "randproj_50")}
    missing = need - have
    assert not missing, f"missing shard rows: {missing}"

    span_by_size = {}
    for k in INPUT_SIZES:
        rows_k = {r["encoder"]: r["recon_pearson"] for r in rows if r["input_genes"] == k}
        vals = list(rows_k.values())
        # F2's convention: the span is over NON-DEGENERATE encoders (the paper excludes
        # the degenerate random-projection-50 fallback from its 0.0045 span); here the
        # refit-free non-degenerate pair is the identity oracle and PCA-50.
        nd = [rows_k["oracle_identity"], rows_k["pca_50"]]
        span_by_size[str(k)] = {
            "min": float(min(vals)), "max": float(max(vals)),
            "span_all_encoders": float(max(vals) - min(vals)),
            "span": float(max(nd) - min(nd)),
            "span_definition": "non-degenerate (oracle vs PCA-50), matching the paper's "
                               "exclusion of the degenerate randproj-50 fallback",
        }
        log(f"  input={k}: span_nd={span_by_size[str(k)]['span']:.4f} "
            f"span_all={span_by_size[str(k)]['span_all_encoders']:.4f}")

    out = {
        "purpose": "F2 panel-size sensitivity: circularity-broken (Arm B) reconstruction "
                   "Pearson for refit-free encoders across input-panel sizes",
        "design": "G_out fixed at the original 1,250 genes; encoders see nested prefixes "
                  "of G_in (deterministic; G_in drawn by gene_split_seed 1234); decoder "
                  "retrained per (encoder, size) with phase2a's train_decoder (seed 42, CPU)",
        "input_sizes": INPUT_SIZES,
        "panel_terms": "input {250,625,1250} = total panel {500,1250,2500}; input 2500 / "
                       "panel 5000 infeasible: corpus caps at a 2,500-gene shared panel and "
                       "circularity-broken disjointness leaves at most 1,250 input genes",
        "paper_reference_span": 0.0045,
        "anchors_phase2a": anchors,
        "rows": rows,
        "span_by_input_size": span_by_size,
        "elapsed_s": time.time() - t0,
    }
    (RESULTS / "panelsize_sensitivity.json").write_text(json.dumps(out, indent=2))
    log(f"=== merged -> {RESULTS / 'panelsize_sensitivity.json'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
