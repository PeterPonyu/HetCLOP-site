import numpy as np
import pytest
import scipy.sparse as sp

from src.embedders.validation import assert_raw_counts


def test_assert_raw_counts_accepts_raw_integer_counts():
    X = np.array([[0.0, 3.0, 12.0], [5.0, 0.0, 1.0]])
    assert_raw_counts(X)  # must not raise


def test_assert_raw_counts_rejects_negative_values():
    X = np.array([[0.0, -1.0, 12.0]])
    with pytest.raises(ValueError, match="negative"):
        assert_raw_counts(X)


def test_assert_raw_counts_rejects_log_normalized_data():
    # log1p-normalized data: nonzero values are fractional, not integer counts
    X = np.array([[0.0, 1.386294, 2.564949], [0.693147, 0.0, 1.098612]])
    with pytest.raises(ValueError, match="raw"):
        assert_raw_counts(X)


def test_assert_raw_counts_accepts_sparse_raw_integer_counts():
    # Every real call site (geneformer_embed.py, scfoundation_embed.py,
    # uce_embed.py) passes adata.X, which is sparse for virtually every real
    # single-cell dataset -- exercise the sp.issparse branch, not just dense.
    X = sp.csr_matrix(np.array([[0.0, 3.0, 12.0], [5.0, 0.0, 1.0]]))
    assert_raw_counts(X)  # must not raise


def test_assert_raw_counts_rejects_sparse_non_integer_values():
    X = sp.csr_matrix(np.array([[0.0, 1.386294, 2.564949], [0.693147, 0.0, 1.098612]]))
    with pytest.raises(ValueError, match="raw"):
        assert_raw_counts(X)
