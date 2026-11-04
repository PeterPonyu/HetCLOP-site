"""Phase 2a: does the Phase-0 encoder-side heterogeneity ranking survive
decoding into gene space through a retrained, ENCODER-AGNOSTIC MLP decoder?

Two arms:
  Arm A (in-panel, CONFOUNDED in PCA's favour): every encoder decodes the full
    shared 2500-gene panel. PCA's decoder is ~its own linear inverse -> circular.
  Arm B (circularity broken, LOAD-BEARING): the 2500 panel is split by a fixed
    seed into G_in (1250) and G_out (1250). Every encoder sees only G_in; every
    decoder predicts G_out (genes no encoder saw). PCA has no inverse advantage.

Encoders: PCA-50, RandomProj-50 (null), scGPT-512, Geneformer-768,
scFoundation-3072, UCE-1280. Arm A reuses cached full-gene FM embeddings; Arm B
re-embeds on G_in-restricted data (scFoundation/UCE have a hard >=2000-gene
floor and are EXPECTED to fail Arm B -- logged, excluded, reported).

Stages (resumable): prep -> encode (GPU) -> decode. Run with --stage all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np

from src.data.real_corpus import assemble_real_corpus
from src.data.shared_gene_panel import canonicalize_symbols, select_shared_panel
from src.diagnostics.tightness import per_dataset_tightness_ratio
from src.orthologs.mapping import load_ortholog_table
from src.utils.paths import CORPUS_SOURCE_ROOT, ORTHOLOG_TABLE_PATH

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from combine_datasets import ALL_DATASETS, fm_cache_path  # noqa: E402
from decoder_eval import reconstruction_metrics, train_decoder  # noqa: E402

CACHE = HERE / "cache"
RESULTS = HERE / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

PANEL_SIZE = 2500
SPLIT_SEED = 42
GENE_SPLIT_SEED = 1234
DEVICE = "cuda"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# Data reprojection (raw counts onto the shared panel; log1p is applied later) #
# --------------------------------------------------------------------------- #
def reproject_raw(counts, dataset_human_symbols, panel_index):
    out = np.zeros((counts.shape[0], len(panel_index)), dtype=np.float32)
    for col, sym in enumerate(dataset_human_symbols):
        if sym is None:
            continue
        j = panel_index.get(sym)
        if j is not None:
            out[:, j] += counts[:, col]
    return out


# --------------------------------------------------------------------------- #
# Stage 1: prep                                                               #
# --------------------------------------------------------------------------- #
def stage_prep():
    log("PREP: assembling 50-dataset corpus")
    corpus = assemble_real_corpus(ALL_DATASETS, CORPUS_SOURCE_ROOT)
    ordered = list(corpus["adata_by_dataset"].keys())
    ot = load_ortholog_table(ORTHOLOG_TABLE_PATH)
    log(f"PREP: {len(ordered)} datasets, {len(corpus['skipped'])} skipped")

    # Canonicalize each dataset's genes to human symbols; build shared panel.
    human_syms = {}
    for name in ordered:
        a = corpus["adata_by_dataset"][name]
        human_syms[name] = canonicalize_symbols(
            list(a.var_names), corpus["species_by_dataset"][name], ot
        )
    panel, pstats = select_shared_panel(human_syms, target_size=PANEL_SIZE)
    panel_index = {s: j for j, s in enumerate(panel)}
    log(f"PREP: panel {len(panel)} genes (min/med/max ds-freq "
        f"{pstats['panel_min_dataset_freq']}/{pstats['panel_median_dataset_freq']}/"
        f"{pstats['panel_max_dataset_freq']})")

    # Fixed disjoint gene split.
    grng = np.random.default_rng(GENE_SPLIT_SEED)
    perm = grng.permutation(len(panel))
    gin_idx = np.sort(perm[: len(panel) // 2]).astype(int)
    gout_idx = np.sort(perm[len(panel) // 2:]).astype(int)

    n_total = int(sum(corpus["adata_by_dataset"][n].n_obs for n in ordered))
    raw_full = np.zeros((n_total, len(panel)), dtype=np.float32)
    dataset_ids = np.empty(n_total, dtype=object)
    labels = np.empty(n_total, dtype=object)
    within_pos = np.zeros(n_total, dtype=np.int64)
    split = np.zeros(n_total, dtype=np.int8)  # 0 train, 1 val, 2 test

    srng = np.random.default_rng(SPLIT_SEED)
    cur = 0
    for name in ordered:
        a = corpus["adata_by_dataset"][name]
        counts = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        counts = counts.astype(np.float32)
        rp = reproject_raw(counts, human_syms[name], panel_index)
        nd = a.n_obs
        raw_full[cur:cur + nd] = rp
        dataset_ids[cur:cur + nd] = name
        labels[cur:cur + nd] = corpus["labels"][corpus["dataset_ids"] == name]
        within_pos[cur:cur + nd] = np.arange(nd)
        # 70/15/15 split within this dataset
        p = srng.permutation(nd)
        n_tr, n_va = int(0.70 * nd), int(0.15 * nd)
        s = np.zeros(nd, dtype=np.int8)
        s[p[n_tr:n_tr + n_va]] = 1
        s[p[n_tr + n_va:]] = 2
        split[cur:cur + nd] = s
        cur += nd
    assert cur == n_total

    logX_full = np.log1p(raw_full).astype(np.float32)

    np.save(CACHE / "raw_full.npy", raw_full)
    np.save(CACHE / "logX_full.npy", logX_full)
    np.savez(CACHE / "meta.npz", dataset_ids=dataset_ids, labels=labels,
             within_pos=within_pos, split=split, ordered=np.array(ordered, dtype=object))
    (CACHE / "panel.json").write_text(json.dumps({
        "panel": panel, "gin_idx": gin_idx.tolist(), "gout_idx": gout_idx.tolist(),
        "gene_split_seed": GENE_SPLIT_SEED, "split_seed": SPLIT_SEED,
        "panel_stats": {k: v for k, v in pstats.items() if k != "vocab_freq_histogram"},
        "g_in_genes": [panel[i] for i in gin_idx], "g_out_genes": [panel[i] for i in gout_idx],
    }, indent=2))
    n_tr = int((split == 0).sum()); n_va = int((split == 1).sum()); n_te = int((split == 2).sum())
    log(f"PREP done: N={n_total} train/val/test={n_tr}/{n_va}/{n_te}  raw_full={raw_full.shape}")


# --------------------------------------------------------------------------- #
# Stage 2: encode (GPU)                                                       #
# --------------------------------------------------------------------------- #
def _load_meta():
    m = np.load(CACHE / "meta.npz", allow_pickle=True)
    panel_info = json.loads((CACHE / "panel.json").read_text())
    return m, panel_info


def _build_adata(raw_block, var_names):
    a = ad.AnnData(X=raw_block.copy())
    a.var_names = [str(v) for v in var_names]
    a.layers["counts"] = raw_block.copy()
    a.obs["cell_type"] = "unused"
    a.obs["batch"] = "b"
    return a


def stage_encode():
    m, panel_info = _load_meta()
    dataset_ids = m["dataset_ids"]
    ordered = list(m["ordered"])
    panel = panel_info["panel"]
    gin_idx = np.array(panel_info["gin_idx"], dtype=int)
    gin_genes = [panel[i] for i in gin_idx]
    raw_full = np.load(CACHE / "raw_full.npy", mmap_mode="r")

    # ---- scGPT : Arm A (full panel) + Arm B (G_in)
    from src.embedders.scgpt_embed import ScGPTCellEncoder
    for arm, cols, var, tag in [
        ("A", None, panel, "enc_A_scgpt.npy"),
        ("B", gin_idx, gin_genes, "enc_B_scgpt.npy"),
    ]:
        outp = CACHE / tag
        if outp.exists():
            log(f"scGPT arm {arm}: cached {tag}, skip")
            continue
        log(f"scGPT arm {arm}: encoding {len(dataset_ids)} cells "
            f"({'2500' if cols is None else '1250 G_in'} genes)")
        enc = ScGPTCellEncoder(device=DEVICE, batch_size=64)
        emb = np.zeros((len(dataset_ids), enc.embed_dim), dtype=np.float32)
        t0 = time.time()
        for name in ordered:
            mask = dataset_ids == name
            block = np.asarray(raw_full[mask])
            if cols is not None:
                block = block[:, cols]
            a = _build_adata(block, var)
            emb[mask] = enc.encode(a)
        np.save(outp, emb)
        log(f"scGPT arm {arm} done: {emb.shape} in {time.time()-t0:.1f}s -> {tag}")

    # ---- Geneformer Arm B (re-embed on G_in-restricted raw counts)
    outp = CACHE / "enc_B_geneformer.npy"
    if not outp.exists():
        try:
            from src.embedders.geneformer_embed import embed as gf_embed
            log("Geneformer arm B: re-embedding on G_in (1250 genes)")
            hidden = 768
            emb = np.zeros((len(dataset_ids), hidden), dtype=np.float32)
            t0 = time.time()
            failed = []
            for name in ordered:
                mask = dataset_ids == name
                block = np.asarray(raw_full[mask])[:, gin_idx]
                a = _build_adata(block, gin_genes)
                tmp = str(CACHE / "_gf_tmp.npz")
                try:
                    gf_embed(a, "cell_type", "batch", tmp, species="human", ortholog_table=None)
                    emb[mask] = np.load(tmp)["X"]
                except Exception as e:
                    failed.append((name, str(e)[:120]))
            if failed:
                log(f"Geneformer arm B per-dataset failures: {failed}")
            np.save(outp, emb)
            log(f"Geneformer arm B done: {emb.shape} in {time.time()-t0:.1f}s "
                f"({len(ordered)-len(failed)}/{len(ordered)} datasets ok)")
        except Exception:
            log("Geneformer arm B FAILED:\n" + traceback.format_exc())
    else:
        log("Geneformer arm B: cached, skip")

    # ---- scFoundation / UCE Arm B feasibility probe (expected to hit the
    #      >=2000-gene floor on a 1250-gene G_in). Attempt on ONE dataset to
    #      capture the real error, then mark excluded.
    probe = {}
    name0 = ordered[0]
    mask0 = dataset_ids == name0
    block0 = np.asarray(raw_full[mask0])[:, gin_idx]
    a0 = _build_adata(block0, gin_genes)
    try:
        from src.embedders.scfoundation_embed import embed as scf_embed
        scf_embed(a0, "cell_type", "batch", str(CACHE / "_scf_tmp.npz"),
                  species="human", ortholog_table=None)
        probe["scfoundation"] = "unexpectedly succeeded on G_in"
    except Exception as e:
        probe["scfoundation"] = f"EXCLUDED: {str(e)[:200]}"
    try:
        from src.embedders.uce_embed import run as uce_run
        tmp_h5 = str(CACHE / "_uce_probe.h5ad")
        a0.write(tmp_h5)
        uce_run(tmp_h5, "cell_type", "batch", "uce_probe_gin", species="human")
        probe["uce"] = "unexpectedly succeeded on G_in"
    except Exception as e:
        probe["uce"] = f"EXCLUDED: {str(e)[:200]}"
    (CACHE / "armB_fm_probe.json").write_text(json.dumps(probe, indent=2))
    log(f"Arm B scFoundation/UCE feasibility: {probe}")
    log("ENCODE stage done")


# --------------------------------------------------------------------------- #
# Stage 3: decode + metrics                                                   #
# --------------------------------------------------------------------------- #
def _pca_fit_transform(logX, train_mask, n_comp=50, seed=0):
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(n_comp, logX.shape[1]), random_state=seed)
    p.fit(logX[train_mask])
    return p.transform(logX).astype(np.float32)


def _randproj(logX, n_comp=50, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((logX.shape[1], n_comp)).astype(np.float32) / np.sqrt(logX.shape[1])
    return (logX @ R).astype(np.float32)


def _latent_tightness(Z, raw_by_ds_test, labels_te, ds_te):
    try:
        med, iqr, vals = per_dataset_tightness_ratio(Z, raw_by_ds_test, labels_te, ds_te)
        return {"median": med, "iqr": iqr, "n_datasets": len(vals)}
    except Exception as e:
        return {"error": str(e)[:150]}


def stage_decode():
    m, panel_info = _load_meta()
    dataset_ids = m["dataset_ids"]; labels = m["labels"]
    within_pos = m["within_pos"]; split = m["split"]; ordered = list(m["ordered"])
    panel = panel_info["panel"]
    gin_idx = np.array(panel_info["gin_idx"], dtype=int)
    gout_idx = np.array(panel_info["gout_idx"], dtype=int)
    logX_full = np.load(CACHE / "logX_full.npy")

    tr = split == 0; va = split == 1; te = split == 2
    labels_te = labels[te]; ds_te = dataset_ids[te]

    # Native per-dataset raw denominators (Phase-0 convention), subset to TEST
    # cells in the SAME within-dataset order as the encoders/targets.
    log("DECODE: re-assembling corpus for native raw denominators")
    corpus = assemble_real_corpus(ALL_DATASETS, CORPUS_SOURCE_ROOT)
    # Sanity: identical cell order to prep.
    assert list(corpus["adata_by_dataset"].keys()) == ordered
    assert np.array_equal(corpus["dataset_ids"].astype(str), dataset_ids.astype(str))
    raw_by_ds = corpus["raw_X_by_dataset"]
    raw_by_ds_test = {}
    for d in ordered:
        dmask_te = te & (dataset_ids == d)
        raw_by_ds_test[d] = raw_by_ds[d][within_pos[dmask_te]]

    # Decode targets
    target_full = logX_full            # Arm A target = all 2500 genes
    target_gout = logX_full[:, gout_idx]  # Arm B target = G_out (1250)

    def eval_encoder(Z, target, arm, enc_name, capacity_control=False):
        Ztr, Zva, Zte = Z[tr], Z[va], Z[te]
        Ytr, Yva = target[tr], target[va]
        Yte = target[te]
        pred, vloss, hist, nparams = train_decoder(
            Ztr, Ytr, Zva, Yva, Zte, device=DEVICE)
        rec = reconstruction_metrics(pred, Yte)
        lat = _latent_tightness(Zte, raw_by_ds_test, labels_te, ds_te)
        try:
            dmed, diqr, dvals = per_dataset_tightness_ratio(
                pred, raw_by_ds_test, labels_te, ds_te)
            dec = {"median": dmed, "iqr": diqr, "n_datasets": len(dvals)}
        except Exception as e:
            dec = {"error": str(e)[:150]}
        row = {
            "arm": arm, "encoder": enc_name, "embed_dim": int(Z.shape[1]),
            "decoder_params": nparams, "val_mse": vloss,
            "reconstruction": rec, "latent_tightness": lat, "decoded_tightness": dec,
            "capacity_control": capacity_control,
        }
        cc = "[cap50]" if capacity_control else ""
        log(f"  {arm} {enc_name}{cc}: dim={Z.shape[1]} pearson={rec['mean_gene_pearson']:.4f} "
            f"mse={rec['mse']:.4f} latentT={lat.get('median','NA')} decodedT={dec.get('median','NA')}")
        return row

    results = []

    # ---------------- Arm A (in-panel, confounded) ----------------
    log("=== ARM A (in-panel, 2500-gene decode; CONFOUNDED in PCA's favour) ===")
    armA = {}
    armA["pca"] = _pca_fit_transform(logX_full, tr, 50)
    armA["randproj"] = _randproj(logX_full, 50, seed=7)
    if (CACHE / "enc_A_scgpt.npy").exists():
        armA["scgpt"] = np.load(CACHE / "enc_A_scgpt.npy")
    for fm in ["geneformer", "scfoundation", "uce"]:
        try:
            parts = [np.load(fm_cache_path(fm, n))["X"] for n in ordered]
            arr = np.vstack(parts)
            if arr.shape[0] == len(dataset_ids):
                armA[fm] = arr.astype(np.float32)
            else:
                log(f"  ArmA {fm}: cached rows {arr.shape[0]} != {len(dataset_ids)}, skip")
        except Exception as e:
            log(f"  ArmA {fm}: cache load failed ({str(e)[:100]}), skip")

    for enc_name, Z in armA.items():
        results.append(eval_encoder(Z, target_full, "A", enc_name))
    # capacity control: project every encoder to 50-d (PCA on train) then decode
    for enc_name, Z in armA.items():
        if Z.shape[1] <= 50:
            continue
        Z50 = _pca_fit_transform(Z, tr, 50)
        results.append(eval_encoder(Z50, target_full, "A", enc_name, capacity_control=True))

    # ---------------- Arm B (circularity broken) ----------------
    log("=== ARM B (G_in -> G_out; circularity broken; LOAD-BEARING) ===")
    logX_gin = logX_full[:, gin_idx]
    armB = {}
    armB["pca"] = _pca_fit_transform(logX_gin, tr, 50)
    armB["randproj"] = _randproj(logX_gin, 50, seed=7)
    if (CACHE / "enc_B_scgpt.npy").exists():
        armB["scgpt"] = np.load(CACHE / "enc_B_scgpt.npy")
    if (CACHE / "enc_B_geneformer.npy").exists():
        gf = np.load(CACHE / "enc_B_geneformer.npy")
        # exclude all-zero rows (per-dataset geneformer failures) datasets
        armB["geneformer"] = gf
    for enc_name, Z in armB.items():
        results.append(eval_encoder(Z, target_gout, "B", enc_name))
    for enc_name, Z in armB.items():
        if Z.shape[1] <= 50:
            continue
        Z50 = _pca_fit_transform(Z, tr, 50)
        results.append(eval_encoder(Z50, target_gout, "B", enc_name, capacity_control=True))

    probe = {}
    if (CACHE / "armB_fm_probe.json").exists():
        probe = json.loads((CACHE / "armB_fm_probe.json").read_text())

    out = {
        "n_cells": int(len(dataset_ids)),
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "panel_size": len(panel), "g_in": int(len(gin_idx)), "g_out": int(len(gout_idx)),
        "armB_fm_feasibility": probe,
        "results": results,
    }
    (RESULTS / "phase2a_results.json").write_text(json.dumps(out, indent=2))
    log(f"DECODE done -> {RESULTS / 'phase2a_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "encode", "decode", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("prep", "all"):
        if (CACHE / "raw_full.npy").exists() and args.stage == "all":
            log("PREP: cache exists, skip")
        else:
            stage_prep()
    if args.stage in ("encode", "all"):
        stage_encode()
    if args.stage in ("decode", "all"):
        stage_decode()


if __name__ == "__main__":
    main()
