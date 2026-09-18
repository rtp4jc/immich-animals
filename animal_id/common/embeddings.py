"""Shared vector operations on embedding arrays."""

import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row of *embeddings*; zero-norm rows are left as-is."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / safe_norms
