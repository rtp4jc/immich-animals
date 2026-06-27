"""Cluster embeddings with cosine-distance DBSCAN, mirroring Immich face grouping.

``eps`` maps to Immich's ``maxDistance`` and ``min_samples`` to ``minFaces``; label
``-1`` is noise/unassigned. Embeddings are L2-normalized for parity with Immich
(cosine distance is norm-invariant, so this doesn't change the result).
"""

import numpy as np
from sklearn.cluster import DBSCAN


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row of *embeddings*; zero-norm rows are left as-is."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / safe_norms


def cluster(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 3,
) -> np.ndarray:
    """Cluster ``(N, D)`` embeddings with cosine-distance DBSCAN.

    ``eps`` is the max cosine distance between neighbors, ``min_samples`` the min
    neighborhood size for a core point. Returns a length-N label array; ``-1`` is
    noise/unassigned.
    """
    normed = normalize_embeddings(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(normed)
