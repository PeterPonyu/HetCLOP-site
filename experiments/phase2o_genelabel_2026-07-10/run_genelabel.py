"""Phase 2o (Tier 2) -- scVI dispersion="gene-label": the INTERMEDIATE conditionality rung.

The theta probe (spec §36) showed factor (b) as a BINARY contrast: a global gene-wise theta
(scVI `gene`, CFGen) loses to Poisson, while a per-cell amortised theta (scVI `gene-cell`)
beats it. But conditionality is not binary. scVI also offers `gene-label`: one theta per gene
per CELL-TYPE label -- conditional on the group, between global and per-cell.

This makes factor (b) a GRADED axis:  global (gene)  <  per-group (gene-label)  <  per-cell (gene-cell).

Prediction from the law: because the within-type variance deficit differs BY CELL TYPE, a theta
conditioned on the cell-type label should already place the missing variance where it is missing
-- so `gene-label` should calibrate like `gene-cell` (NB beats Poisson), not like `gene`
(Poisson wins). If instead `gene-label` behaves like `gene`, the law's factor (b) needs the
finer per-cell conditioning, which would be a sharper (and reportable) claim.

Mechanics: identical to the deployed-model harness, except setup_anndata also gets a
labels_key (cell type) so scVI can fit a per-(gene,label) dispersion. mu is the model's own
fitted count mean; the within-decoder Poisson-vs-NB control holds mu fixed.

"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scvi  # noqa: E402

import phase2j_lib as L  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def fit_and_extract_genelabel(data, seed, n_param_samples, max_epochs, device):
    """Train SCVI with dispersion='gene-label' (one theta per gene per cell type)."""
    scvi.settings.seed = seed

    def build(mask):
        A = ad.AnnData(
            sp.csr_matrix(data["raw"][mask].astype(np.float32)),
            obs={"dataset": data["dataset_ids"][mask].astype(str),
                 "cell_type": data["labels"][mask].astype(str)},
        )
        A.layers["counts"] = A.X.copy()
        return A

    A_tr = build(data["tr"])
    scvi.model.SCVI.setup_anndata(A_tr, layer="counts", batch_key="dataset", labels_key="cell_type")
    model = scvi.model.SCVI(A_tr, gene_likelihood="nb", dispersion="gene-label")
    accel = "gpu" if device == "cuda" else "cpu"
    kw = dict(accelerator=accel, enable_progress_bar=False, early_stopping=True)
    if max_epochs is not None:
        kw["max_epochs"] = max_epochs
    model.train(**kw)

    A_te = build(data["te"])
    params = model.get_likelihood_parameters(adata=A_te, n_samples=n_param_samples, give_mean=True)
    mu = np.asarray(params["mean"], dtype=np.float64)
    theta = np.asarray(params["dispersions"], dtype=np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, mu.shape).copy()
    n_epochs = int(len(model.history["elbo_train"])) if "elbo_train" in model.history else -1
    # how many distinct (gene,label) dispersion values did it actually fit?
    n_labels = int(np.unique(data["labels"][data["tr"]]).size)
    return mu, theta, {"n_epochs_trained": n_epochs, "dispersion": "gene-label", "n_labels": n_labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-param-samples", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    L.log(f"gene-label conditionality rung; seeds={seeds}")
    data = L.load_corpus()
    det = L.deterministic_baseline_log_pearson(data, seed=42, device=args.device)
    L.log(f"deterministic PCA-512 baseline log1p Pearson = {det:.4f}")

    out = {
        "purpose": "scVI dispersion='gene-label' -- theta conditional on CELL TYPE (the "
                   "intermediate rung between global 'gene' and per-cell 'gene-cell'). Tests "
                   "whether factor (b) is graded: does per-group conditioning already flip the "
                   "winner to NB, like per-cell does?",
        "prediction": "gene-label calibrates like gene-cell (NB beats Poisson), because the "
                      "variance deficit differs by cell type and a per-label theta places it per group.",
        "deterministic_baseline_log_pearson": det, "seeds": seeds, "results": {},
    }
    for seed in seeds:
        L.log(f"=== scVI gene-label seed {seed} ===")
        mu, theta, meta = fit_and_extract_genelabel(
            data, seed, args.n_param_samples, args.max_epochs, args.device)
        res = L.evaluate_dispersion_control("scvi_gene_label", mu, theta, data, det, seed=seed)
        res["train_meta"] = meta
        out["results"][str(seed)] = res
        c = res["contrasts"]["rvar_expressed"]
        L.log(f"  [gene-label seed{seed}] poisson_win_groups={c['poisson_win_fraction_groups']:.3f} "
              f"clustered_p={c['clustered_wilcoxon_p']:.2e}  "
              f"=> {'NB wins (like gene-cell)' if c['poisson_win_fraction_groups'] < 0.5 else 'Poisson wins (like gene)'}")
        (RESULTS / "scvi_genelabel.json").write_text(json.dumps(out, indent=2))
    L.log(f"=== gene-label done -> {RESULTS / 'scvi_genelabel.json'} ===")


if __name__ == "__main__":
    main()
