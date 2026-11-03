import numpy as np
import pytest

from src.eval.r_var_split import r_var_decoded, r_var_latent


def test_r_var_decoded_perfect_linear_relationship():
    real_expr = np.array(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 5.0, 10.0, 3.0], [3.0, 9.0, 19.0, 5.0]]
    )
    # gen_var = 3 * real_var exactly (uniform positive scaling of every
    # column) -> Pearson r of the two per-gene-variance vectors is exactly 1.
    gen_expr = real_expr * np.sqrt(3.0)
    r = r_var_decoded(real_expr, gen_expr)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_r_var_latent_perfect_linear_relationship_pooled_across_groups():
    real_latent = np.array(
        [
            [1.0, 2.0], [2.0, 5.0], [3.0, 9.0],  # group A
            [10.0, 1.0], [12.0, 4.0], [15.0, 8.0],  # group B
        ]
    )
    labels = ["A", "A", "A", "B", "B", "B"]
    gen_latent = real_latent * np.sqrt(3.0)
    r = r_var_latent(real_latent, gen_latent, labels)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_r_var_latent_requires_at_least_two_cells_per_used_group():
    real_latent = np.array([[1.0], [2.0], [3.0]])
    labels = ["A", "A", "B"]  # group B has only 1 cell, must be skipped not crash
    gen_latent = real_latent * 2.0
    # Only group A (2 cells) contributes. With a single latent dimension (1-D),
    # the pooled variance arrays have length 1 after concatenation. Since
    # pearsonr requires length >= 2 to compute correlation, this case is
    # mathematically undefined and should return nan instead of crashing.
    r = r_var_latent(real_latent, gen_latent, labels)
    assert isinstance(r, float) and np.isnan(r)
