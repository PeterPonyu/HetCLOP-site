"""Unit tests for the one-sided mean acceptance gate.

The gate is an UNDERFIT guard, not a two-sided equivalence test: a decoder whose
mean fits BETTER than the deterministic baseline must PASS (it exhibits none of
the heteroscedastic-collapse pathology the gate exists to catch), and one whose
mean is WORSE by more than tol must FAIL. A two-sided ``abs(gap) <= tol`` version
wrongly excluded a healthy NB decoder whose count likelihood produced a better
mean (randproj_512: NB mean 0.785 vs det 0.752). These tests lock the semantics.
"""
from src.evaluation.acceptance import mean_acceptance_gate


def test_gate_passes_when_model_better_by_a_lot():
    # model mean 0.05 BETTER than deterministic -> healthy fit -> PASS
    assert mean_acceptance_gate(det_pearson=0.75, model_pearson=0.80, tol=0.02) is True


def test_gate_fails_when_model_worse_by_a_lot():
    # model mean 0.05 WORSE than deterministic -> underfit pathology -> FAIL
    assert mean_acceptance_gate(det_pearson=0.75, model_pearson=0.70, tol=0.02) is False


def test_gate_passes_within_tolerance_either_direction():
    assert mean_acceptance_gate(0.75, 0.74, tol=0.02) is True   # worse by 0.01 <= tol
    assert mean_acceptance_gate(0.75, 0.76, tol=0.02) is True   # better by 0.01


def test_gate_boundary_just_inside_and_outside_tol():
    # worse by just under tol -> PASS; worse by just over tol -> FAIL
    assert mean_acceptance_gate(0.75, 0.731, tol=0.02) is True    # gap ~0.019
    assert mean_acceptance_gate(0.75, 0.729, tol=0.02) is False   # gap ~0.021


def test_gate_matches_known_phase_decisions():
    # pca_512 NB: gap -0.0060 (better) -> PASS
    assert mean_acceptance_gate(0.7420, 0.7479, 0.02) is True
    # randproj_512 NB: gap -0.0329 (better) -> PASS (the false-exclusion case)
    assert mean_acceptance_gate(0.7522, 0.7851, 0.02) is True
    # scgpt_512 NB: gap +0.0216 (worse) -> FAIL (exclusion stands)
    assert mean_acceptance_gate(0.6900, 0.6684, 0.02) is False
