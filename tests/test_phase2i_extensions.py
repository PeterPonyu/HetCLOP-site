"""Pins the load-bearing conclusions of phase2i G3/G4/G5 (spec §31).

These three results feed the post-probe F4 rewrite (G3 broadens the E1 over-inflation +
energy-plateau claim to a second gate-passing encoder; G4 upgrades the mean-fit-gap
monotonicity leg from borderline to significant; G5 adds the ZINB rung to the
"more noise parameters => worse Arm-A calibration" ladder). Pinning them here prevents
silent drift before they enter the manuscript. Pure JSON re-reads; no GPU.

Crucial statistic hygiene (defect #14): G3's over-inflation direction MUST be read from the
EXPRESSION-RESTRICTED signed r_var (`expr_signed_median`). The unrestricted field pins at
1.000 (sparsity point mass) and is directionally meaningless.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "phase2i_extensions_2026-07-09" / "results"
G3 = RES / "g3_theta_sweep.json"
G4 = RES / "g4_monotonicity.json"
G5 = RES / "g5_zinb.json"

pytestmark = pytest.mark.skipif(
    not (G3.exists() and G4.exists() and G5.exists()),
    reason="phase2i G3/G4/G5 results absent",
)


def _seed_mean(sweep_by_seed, i, key):
    return st.mean(sweep_by_seed[s]["theta_sweep"][i][key] for s in sweep_by_seed)


def test_g3_randproj_reproduces_pca_overinflation():
    """randproj_512 (a SECOND gate-passing encoder) over-inflates at the fitted theta and
    the ratio never crosses 1 down to the Poisson limit -- reproducing the pca_512 E1
    pattern on the EXPRESSION-RESTRICTED statistic."""
    d = json.loads(G3.read_text())
    seeds = d["per_encoder"]["randproj_512"]["per_seed"]
    sweep = seeds["42"]["theta_sweep"]
    fitted_i = next(i for i, r in enumerate(sweep) if r["is_fitted_median"])
    inf_i = next(i for i, r in enumerate(sweep) if r["theta"] in ("inf", float("inf")))
    fitted = _seed_mean(seeds, fitted_i, "expr_signed_median")
    pois = _seed_mean(seeds, inf_i, "expr_signed_median")
    assert fitted > 1.0, f"fitted-theta expr_signed should over-inflate, got {fitted}"
    assert pois > 1.0, f"Poisson-limit expr_signed should still be >1 (never crosses), got {pois}"
    assert fitted > pois, "over-inflation should decrease monotonically toward Poisson"


def test_g3_no_distinguishable_finite_optimum():
    """§16's exact claim (NOT "finite optimum"): the best finite-theta energy is within noise
    of the Poisson limit -- energy_argmin_over_sweep differs from the Poisson-limit energy by
    less than ~2x the energy CI sd. Holds even when a shallow interior min exists (scgpt's
    argmin is theta=20, ~1.2 sd below the Poisson value)."""
    d = json.loads(G3.read_text())
    for enc in ("randproj_512", "scgpt_512"):
        seeds = d["per_encoder"][enc]["per_seed"]
        sweep = seeds["42"]["theta_sweep"]
        energies, sds, inf_i = [], [], None
        for i, r in enumerate(sweep):
            energies.append(_seed_mean(seeds, i, "energy_distance_median"))
            sds.append(_seed_mean(seeds, i, "energy_ci_sd"))
            if r["theta"] in ("inf", float("inf")):
                inf_i = i
        assert inf_i is not None, f"{enc}: sweep has no Poisson (theta=inf) row"
        argmin_i = min(range(len(energies)), key=lambda i: energies[i])
        gap = abs(energies[argmin_i] - energies[inf_i])
        tol = 2.0 * max(sds[argmin_i], sds[inf_i])
        assert gap < tol, (f"{enc}: finite argmin distinguishable from Poisson "
                           f"(gap {gap:.4f} >= {tol:.4f})")


def test_g4_monotonicity_reaches_significance():
    d = json.loads(G4.read_text())
    s = d["statistics"]["gap_vs_energy_effect"]
    assert s["spearman_rho"] < -0.5, "expected a strong negative monotone relation"
    assert s["exact_permutation_p"]["p_two_sided"] < 0.05, "n=7 leg must clear 0.05"
    assert d["statistics"]["n_encoders"] == 7


def test_g4_identity_cross_check_reproduces_phase2f():
    d = json.loads(G4.read_text())
    cc = d["pca_512_identity_cross_check"]
    for k in ("mean_fit_gap", "energy_effect", "rvar_expr_effect"):
        assert abs(cc["recomputed"][k] - cc["phase2f"][k]) < 1e-9, f"{k} must reproduce phase2f"


def test_g5_zinb_hurts_ordering_poisson_gt_nb_gt_zinb():
    """Arm A: adding dispersion then zero-inflation monotonically worsens calibration.
    ZINB loses to NB, NB loses to Poisson-on-mean -- every rung, all seeds, clustered."""
    d = json.loads(G5.read_text())
    v = d["verdict"]
    # ZINB strictly worse than NB (b = NB favored) on both metrics
    assert v["zinb_stochastic_vs_nb_stochastic__energy_distance"]["b_favored_all_seeds"]
    assert v["zinb_stochastic_vs_nb_stochastic__r_var_abs_log2_expr"]["b_favored_all_seeds"]
    # NB strictly worse than Poisson-on-mean (b = Poisson favored) on both
    assert v["nb_stochastic_vs_poisson_on_nb_mean__energy_distance"]["b_favored_all_seeds"]
    assert v["nb_stochastic_vs_poisson_on_nb_mean__r_var_abs_log2_expr"]["b_favored_all_seeds"]
    # ZINB wins almost no groups vs NB on energy
    assert v["zinb_stochastic_vs_nb_stochastic__energy_distance"]["mean_a_win_fraction"] < 0.10
    # learned zero-inflation is nonzero yet useless
    assert 0.0 < d["overall"]["learned_pi_median_across_seeds"] < 0.2
    assert d["overall"]["zinb_worse_than_nb_energy_all_seeds"] is True
