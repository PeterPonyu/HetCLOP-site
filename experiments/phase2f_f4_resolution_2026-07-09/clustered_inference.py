"""Dataset-clustered inference machinery for Phase 2f (F4 resolution).

The Phase 2c NB arm ran a paired Wilcoxon over the 314 (dataset x cell type)
groups, treating them as independent. They are not: the 314 groups come from
only 50 datasets, and every cell type inside a dataset shares the SAME trained
decoder, the same fitted mean, and the same library-size distribution. That is
pseudoreplication -- the effective number of independent units is ~50, not 314.

This module recomputes significance for a paired (nb_stochastic vs control)
contrast while HONOURING the nesting. All functions operate on per-group paired
differences ``diff = a - b`` (a = nb_stochastic metric, b = control metric), each
tagged with its dataset id, so the caller decides the metric and sign convention
once.

Sign convention downstream: for energy_distance and |log2 r_var| (both
lower-is-better), ``diff = nb - control``; diff > 0 means NB is WORSE (control
wins), diff < 0 means NB is BETTER. ``nb_win_fraction`` = fraction of groups with
diff < 0.

Three inferential views are provided:

  * ``naive_paired``            -- paired Wilcoxon over all groups (reproduces the
    Phase 2c number; the discredited baseline, reported for contrast only).
  * ``clustered_dataset_medians`` -- collapse the group diffs to one median per
    dataset (~50 values), then a one-sample Wilcoxon + a sign test on those
    dataset-level medians. This is the PRIMARY corrected inference.
  * ``dataset_signflip_permutation`` -- a cluster-level sign-flip permutation:
    the exchangeable unit under the null is the DATASET, so every group inside a
    dataset flips sign together. This is the permutation analogue that actually
    fixes the pseudoreplication (a per-group flip would not). A literal
    within-dataset per-group flip is provided too (``within_dataset_signflip_
    permutation``) for completeness.

The permutation STATISTIC is the Wilcoxon SIGNED-RANK sum of the per-dataset
median diffs (``sum_d sign(m_d) * rank(|m_d|)``), NOT a raw mean. This is
deliberate: energy-distance median diffs are heavy-tailed across datasets (a
single dataset can have |median| ~ 9 while the next is ~1), so a mean-based
statistic is dominated by one outlier dataset and is uninformative -- it can read
p ~ 0.5 even when 90% of datasets agree in sign. The rank statistic is robust to
that magnitude heterogeneity and matches the rank logic of the primary clustered
Wilcoxon, so the permutation is an exact-permutation confirmation of it.

"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.stats import binomtest, wilcoxon


def paired_group_diffs(a_map, b_map, metric):
    """Return aligned per-group paired differences for one metric.

    ``a_map`` / ``b_map`` map (dataset, cell_type) -> per-group record dict (as
    produced by ``aggregate_over_groups``'s ``per_group``). Only groups present in
    both maps with finite values on ``metric`` are kept, in sorted-key order so
    the result is deterministic and reproduces the naive test's pairing.

    Returns a list of dicts: {"dataset", "cell_type", "a", "b", "diff"=a-b}.
    """
    keys = sorted(set(a_map) & set(b_map))
    out = []
    for k in keys:
        av = a_map[k][metric]
        bv = b_map[k][metric]
        if np.isfinite(av) and np.isfinite(bv):
            out.append({
                "dataset": k[0], "cell_type": k[1],
                "a": float(av), "b": float(bv), "diff": float(av) - float(bv),
            })
    return out


def naive_paired(records):
    """Paired Wilcoxon over ALL groups (the pseudoreplicated baseline).

    Reproduces the Phase 2c ``paired`` statistic exactly (two-sided, zero_method
    'wilcox'). Reported only for contrast against the clustered views."""
    a = np.array([r["a"] for r in records], dtype=np.float64)
    b = np.array([r["b"] for r in records], dtype=np.float64)
    diff = a - b
    if np.count_nonzero(diff) >= 1:
        stat, pval = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        stat, pval = float(stat), float(pval)
    else:
        stat, pval = float("nan"), 1.0
    return {
        "n_groups": int(a.size),
        "median_paired_diff": float(np.median(diff)),
        "nb_win_fraction": float(np.mean(a < b)),
        "wilcoxon_stat": stat,
        "p_value": pval,
    }


def dataset_median_diffs(records):
    """Collapse per-group diffs to one median-diff per dataset.

    Returns (datasets, medians) as parallel arrays, dataset order deterministic
    (first appearance)."""
    by_ds = OrderedDict()
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r["diff"])
    datasets = list(by_ds.keys())
    medians = np.array([np.median(by_ds[d]) for d in datasets], dtype=np.float64)
    return datasets, medians


def clustered_dataset_medians(records):
    """PRIMARY corrected inference: collapse to per-dataset median diffs, then a
    one-sample Wilcoxon (signed-rank vs 0) and a sign test on those ~50 values.

    ``nb_win_fraction_datasets`` = fraction of datasets whose median diff < 0
    (NB better). The sign test's binomial counts datasets with diff < 0 among
    non-zero-median datasets."""
    datasets, medians = dataset_median_diffs(records)
    n = int(medians.size)
    nz = medians[medians != 0.0]
    if nz.size >= 1:
        wstat, wp = wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        wstat, wp = float(wstat), float(wp)
    else:
        wstat, wp = float("nan"), 1.0
    n_neg = int(np.sum(nz < 0))  # datasets where NB is better
    n_eff = int(nz.size)
    if n_eff >= 1:
        sp = float(binomtest(n_neg, n_eff, 0.5, alternative="two-sided").pvalue)
    else:
        sp = 1.0
    return {
        "n_datasets": n,
        "n_datasets_nonzero": n_eff,
        "median_of_dataset_medians": float(np.median(medians)),
        "nb_win_fraction_datasets": float(np.mean(medians < 0)),
        "wilcoxon_stat": wstat,
        "wilcoxon_p": wp,
        "sign_test_n_nb_better": n_neg,
        "sign_test_p": sp,
        "dataset_medians": medians.tolist(),
        "datasets": datasets,
    }


def _signed_rank_stat(values):
    """Wilcoxon signed-rank sum ``sum sign(v) * rank(|v|)`` (average ranks for
    ties, zeros contribute 0). Robust to magnitude outliers; symmetric around 0
    under sign-flip. Used as the permutation statistic instead of a raw mean,
    which energy's heavy-tailed between-dataset magnitudes would otherwise
    dominate."""
    v = np.asarray(values, dtype=np.float64)
    nz = v[v != 0.0]
    if nz.size == 0:
        return 0.0
    order = np.argsort(np.abs(nz), kind="mergesort")
    ranks = np.empty(nz.size, dtype=np.float64)
    ranks[order] = np.arange(1, nz.size + 1)
    # average-rank tie handling on |v|
    absv = np.abs(nz)
    _, inv, counts = np.unique(absv, return_inverse=True, return_counts=True)
    start = np.zeros(counts.size)
    start[1:] = np.cumsum(counts)[:-1]
    avg = start + (counts + 1) / 2.0  # 1-based average rank per tie group
    ranks = avg[inv]
    return float(np.sum(np.sign(nz) * ranks))


def dataset_signflip_permutation(records, n_perm=10000, seed=0):
    """Cluster-level sign-flip permutation (the nesting-honouring permutation).

    Under the null the exchangeable unit is the DATASET: each dataset's whole
    block of group diffs flips sign together, which (for a dataset-level median)
    just flips the sign of that dataset's median diff. Statistic = signed-rank sum
    of the per-dataset median diffs; ranks of |m_d| are invariant to sign flips,
    so this is the exact Wilcoxon signed-rank permutation distribution. Two-sided
    p = fraction of permutations with |stat| >= |observed| (observed included)."""
    datasets, medians = dataset_median_diffs(records)
    n = medians.size
    absr = np.abs(medians)
    nz = absr != 0.0
    ranks = np.zeros(n)
    if nz.any():
        _, inv, counts = np.unique(absr[nz], return_inverse=True, return_counts=True)
        start = np.zeros(counts.size)
        start[1:] = np.cumsum(counts)[:-1]
        avg = start + (counts + 1) / 2.0
        ranks[nz] = avg[inv]
    obs = abs(float(np.sum(np.sign(medians) * ranks)))
    rng = np.random.default_rng(seed)
    ge = 1  # include observed
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if abs(float(np.sum(signs * ranks))) >= obs - 1e-9:
            ge += 1
    return {
        "level": "dataset (cluster-level sign flip; honours nesting)",
        "statistic": "wilcoxon_signed_rank_sum_of_dataset_medians",
        "n_datasets": int(n),
        "n_perm": int(n_perm),
        "observed_stat": float(np.sum(np.sign(medians) * ranks)),
        "p_value": float(ge / (n_perm + 1)),
    }


def within_dataset_signflip_permutation(records, n_perm=10000, seed=0):
    """Literal within-dataset per-group sign flip, blocked by dataset.

    Each group's NB/Poisson label is flipped independently within its dataset;
    the per-dataset medians are recomputed each permutation and the same robust
    signed-rank statistic is taken over the ~50 datasets. Reported alongside the
    cluster-level flip; because within-dataset flips can change a dataset's median
    magnitude it is a slightly different null."""
    by_ds = OrderedDict()
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r["diff"])
    blocks = [np.array(v, dtype=np.float64) for v in by_ds.values()]
    obs_medians = np.array([np.median(b) for b in blocks], dtype=np.float64)
    obs = abs(_signed_rank_stat(obs_medians))
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_perm):
        meds = np.empty(len(blocks), dtype=np.float64)
        for i, b in enumerate(blocks):
            signs = rng.choice(np.array([-1.0, 1.0]), size=b.size)
            meds[i] = np.median(signs * b)
        if abs(_signed_rank_stat(meds)) >= obs - 1e-9:
            ge += 1
    return {
        "level": "within-dataset per-group sign flip (blocked by dataset)",
        "statistic": "wilcoxon_signed_rank_sum_of_dataset_medians",
        "n_perm": int(n_perm),
        "observed_stat": _signed_rank_stat(obs_medians),
        "p_value": float(ge / (n_perm + 1)),
    }
