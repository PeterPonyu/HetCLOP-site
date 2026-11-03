"""Phase 2t -- a DIRECT single-cell view, and what it FAILS to show.

Every other figure in the paper plots a summary statistic; a reader never sees a cell. This builds
the missing single-cell figure: a joint UMAP of REAL held-out cells and the three mean-fixed
synthetic draws.

The result inverts the intent, and that is the finding. The three draws share an IDENTICAL
conditional mean by construction, and a UMAP layout is driven by the mean -- so the embedding is
nearly BLIND to the dispersion. The global-dispersion draw, which destroys marker genes and halves
rare-cell-type recovery, looks almost indistinguishable from the correct one. We measure the
blindness rather than assert it: the relative damage visible to kNN purity in the embedding is
~5x smaller than the damage the marker-gene readout reports. The field's most-used single-cell
figure cannot see this defect -- the same lesson as F1 (a geometry ratio rewards not learning) and
F5 (decoded tightness crowns noise injection), now demonstrated on a UMAP.

DESIGN (identical to the biological readout, so the figure and the numbers describe the same run):
one trained scVI `gene-cell` model supplies mu_ic and theta_ic; three draws share that IDENTICAL
mean and differ only in the dispersion parameterisation:

    poisson        ~ Poisson(mu)                     [no dispersion]
    nb_global      ~ NB(mu, theta_g)                 [theta_g = per-gene median of theta_ic]
    nb_conditional ~ NB(mu, theta_ic)                [the law's prescription]

EMBEDDING (one shared coordinate system, so panels are comparable):
  * subsample test cells, stratified by cell type so RARE types survive the subsample;
  * fit PCA-50 on the REAL log1p cells only (the synthetic draws must not define the space);
  * project real + all three synthetic sets into that same PCA space;
  * run ONE UMAP on the stacked coordinates -> every panel shares axes.

Panels are real / poisson / nb_global / nb_conditional, coloured by cell type, with rare types
(< RARE_MAX real test cells) flagged. Each panel carries its kNN label purity, so the reader can
see that the numbers barely move while the biology collapses.

Outputs coords to JSON; the figure itself is drawn by the R pipeline (rasterised scatter), keeping
the "no Python in the figure path" convention.

"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
import umap  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

import phase2j_lib as L  # noqa: E402
from run_scvi import fit_and_extract  # noqa: E402
from src.architecture.stochastic_decoder import sample_nb, sample_nb_fixed_theta  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
CACHE = Path(__file__).resolve().parent / "cache_mu_theta.npz"

N_CELLS = 2500      # subsampled real test cells (each synthetic set mirrors these same cells)
RARE_MAX = 50       # a cell type is "rare" if it has fewer than this many real test cells
N_PCA = 50
TOP_TYPES = 8       # colour the most abundant types; the rest collapse to "other"


def _log1p(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def stratified_subsample(labels, n_total, rng):
    """Proportional subsample that guarantees rare types are represented."""
    types, counts = np.unique(labels, return_counts=True)
    idx = []
    for t, c in zip(types, counts):
        pool = np.where(labels == t)[0]
        take = max(5, int(round(n_total * c / labels.size)))   # floor of 5 keeps rare types alive
        take = min(take, pool.size)
        idx.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(idx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-param-samples", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    L.log("=== Phase 2t: joint UMAP of real vs mean-fixed synthetic draws ===")
    data = L.load_corpus()
    te, lab_all = data["te"], data["labels"]

    if CACHE.exists():
        L.log(f"reusing cached mu/theta ({CACHE.name})")
        z = np.load(CACHE)
        mu, theta_ic = z["mu"], z["theta"]
    else:
        L.log("fitting scVI gene-cell for its own mu and amortised theta ...")
        mu, theta_ic, _ = fit_and_extract(data, "gene-cell", args.seed, args.n_param_samples, None, args.device)
        mu = np.clip(np.asarray(mu, np.float64), 0.0, None)
        theta_ic = np.clip(np.asarray(theta_ic, np.float64), 1e-8, None)
        np.savez_compressed(CACHE, mu=mu, theta=theta_ic)
        L.log(f"cached -> {CACHE.name}")

    theta_g = np.median(theta_ic, axis=0)
    lab_te = lab_all[te]

    rng = np.random.default_rng(args.seed)
    sub = stratified_subsample(lab_te, N_CELLS, rng)
    L.log(f"  subsampled {sub.size} of {lab_te.size} test cells (stratified)")

    real = _log1p(data["raw"][te][sub])
    mu_s, th_s = mu[sub], theta_ic[sub]
    syn = {
        "poisson": _log1p(sample_nb_fixed_theta(mu_s, float("inf"), seed=args.seed + 1)),
        "nb_global": _log1p(sample_nb(mu_s, np.broadcast_to(theta_g, mu_s.shape).copy(), seed=args.seed + 1)),
        "nb_conditional": _log1p(sample_nb(mu_s, th_s, seed=args.seed + 1)),
    }
    lab = lab_te[sub]

    # rare types, defined on the FULL real test set (not the subsample)
    types, counts = np.unique(lab_te, return_counts=True)
    rare = {t for t, c in zip(types, counts) if c < RARE_MAX}
    top = [t for t, _ in sorted(zip(types, counts), key=lambda x: -x[1])[:TOP_TYPES]]
    L.log(f"  {len(rare)} rare types (<{RARE_MAX} real cells); colouring top {len(top)} types")

    # PCA fitted on REAL cells only, then all sets projected into it
    L.log("  fitting PCA-50 on real cells; projecting all sets ...")
    pca = PCA(n_components=N_PCA, random_state=0).fit(real)
    blocks = [("real", pca.transform(real))] + [(k, pca.transform(v)) for k, v in syn.items()]
    stacked = np.vstack([b for _, b in blocks])

    L.log(f"  running one UMAP on {stacked.shape[0]} stacked points (shared coordinates) ...")
    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=args.seed).fit_transform(stacked)

    out_pts, off = [], 0
    for name, b in blocks:
        n = b.shape[0]
        e = emb[off:off + n]
        off += n
        for i in range(n):
            t = str(lab[i])
            out_pts.append({
                "variant": name,
                "x": round(float(e[i, 0]), 3), "y": round(float(e[i, 1]), 3),
                "type": t if t in top else "other",
                "rare": bool(t in rare),
            })

    # ---- what the PICTURE actually shows, quantified -------------------------------------
    # The three synthetic draws share an IDENTICAL conditional mean by construction, and a UMAP
    # layout is driven by that mean. So the embedding is expected to be nearly blind to the
    # dispersion parameterisation -- which is the point: the field's standard single-cell figure
    # cannot see a defect that destroys marker genes. We measure that blindness rather than assert it.
    def panel_stats(block_emb, labels):
        sp = []
        for t in set(labels):
            if t == "other":
                continue
            m = labels == t
            if m.sum() < 5:
                continue
            c = block_emb[m].mean(0)
            sp.append(float(np.linalg.norm(block_emb[m] - c, axis=1).mean()))
        nn = NearestNeighbors(n_neighbors=16).fit(block_emb)
        _, idx = nn.kneighbors(block_emb)
        pur = float(np.mean([np.mean(labels[idx[i, 1:]] == labels[i]) for i in range(len(block_emb))]))
        return {"within_type_spread": float(np.mean(sp)), "knn_purity_k15": pur}

    lab_disp = np.array([t if t in top else "other" for t in lab])
    stats, off2 = {}, 0
    for name, b in blocks:
        n = b.shape[0]
        stats[name] = panel_stats(emb[off2:off2 + n], lab_disp)
        off2 += n
    for k, v in stats.items():
        L.log(f"  [{k:15s}] knn_purity={v['knn_purity_k15']:.3f} within_type_spread={v['within_type_spread']:.3f}")

    # how much does the embedding UNDER-REPORT the damage, relative to the biological readout?
    bio = json.loads((Path(__file__).resolve().parents[1] /
                      "phase2q_biological_2026-07-10" / "results" / "biological_agg.json").read_text())
    mk_cond = bio["marker_recall_mean_sd"]["nb_conditional"][0]
    mk_glob = bio["marker_recall_mean_sd"]["nb_global"][0]
    pur_cond = stats["nb_conditional"]["knn_purity_k15"]
    pur_glob = stats["nb_global"]["knn_purity_k15"]
    rel_marker = (mk_cond - mk_glob) / mk_cond          # relative damage the BIOLOGY sees
    rel_purity = (pur_cond - pur_glob) / pur_cond       # relative damage the PICTURE sees
    blindness = rel_marker / rel_purity if rel_purity > 0 else float("inf")
    L.log(f"  relative damage seen by marker recall = {rel_marker:.3f}; by UMAP kNN purity = {rel_purity:.3f} "
          f"-> the embedding under-reports it {blindness:.1f}x")

    out = {
        "panel_stats": stats,
        "relative_damage_marker_recall": rel_marker,
        "relative_damage_umap_purity": rel_purity,
        "umap_underreports_damage_factor": blindness,
        "purpose": "Joint UMAP of real held-out cells and three mean-fixed synthetic draws "
                   "(identical mu; only the dispersion parameterisation differs), so the "
                   "biological effect of mis-placed dispersion is directly visible.",
        "design": "PCA-50 fitted on REAL cells only; all sets projected into it; ONE UMAP on the "
                  "stacked coordinates so all panels share axes.",
        "seed": args.seed, "n_cells_per_panel": int(sub.size),
        "rare_max": RARE_MAX, "n_rare_types": len(rare), "top_types": top,
        "variants": ["real"] + list(syn),
        "points": out_pts,
    }
    (RESULTS / "umap.json").write_text(json.dumps(out))
    L.log(f"  wrote {len(out_pts)} points -> {RESULTS / 'umap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
