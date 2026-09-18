"""
Unit tests for animal_id.common.embeddings.
"""

import numpy as np

from animal_id.common.embeddings import normalize_embeddings


class TestNormalizeEmbeddings:
    def test_unit_vectors_unchanged(self):
        """Already-normalized rows stay unchanged."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((8, 16)).astype(np.float32)
        normed = normalize_embeddings(raw)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-6)

    def test_zero_row_safe(self):
        """A zero-norm row must not produce NaN."""
        emb = np.zeros((3, 4), dtype=np.float32)
        emb[1] = [1.0, 0.0, 0.0, 0.0]
        result = normalize_embeddings(emb)
        assert not np.any(np.isnan(result))
        # Non-zero row is still normalized.
        np.testing.assert_allclose(np.linalg.norm(result[1]), 1.0, atol=1e-6)
