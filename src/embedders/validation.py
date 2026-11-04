"""Shared input validation for embedder ports. None of the three embedders (Geneformer, scFoundation,
UCE) asserted their input is raw counts before performing their own
normalization (norm-to-10k/gene-median for Geneformer; log1p(v/total*1e4)
for scFoundation) -- silently producing plausible-looking but wrong
embeddings if fed already-log-normalized data instead of raw counts."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def assert_raw_counts(X, context: str = "") -> None:
    prefix = f"{context}: " if context else ""
    sample = X[:1000] if X.shape[0] > 1000 else X
    dense = sample.toarray() if sp.issparse(sample) else np.asarray(sample)

    if (dense < 0).any():
        raise ValueError(
            f"{prefix}input contains negative values, expected raw non-negative counts"
        )
    nonzero = dense[dense != 0]
    if nonzero.size and not np.allclose(nonzero, np.round(nonzero), atol=1e-6):
        raise ValueError(
            f"{prefix}input does not look like raw integer counts (found non-integer "
            "nonzero values) -- do not pass log-normalized or CPM-normalized data"
        )
