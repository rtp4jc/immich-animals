"""Immich-style identity clustering for animal embeddings.

Public API: ``embed_gallery``, ``normalize_embeddings``, ``cluster``,
``cluster_quality``.
"""

from .clusterer import cluster, normalize_embeddings
from .embedder import embed_gallery
from .metrics import cluster_quality

__all__ = [
    "embed_gallery",
    "normalize_embeddings",
    "cluster",
    "cluster_quality",
]
