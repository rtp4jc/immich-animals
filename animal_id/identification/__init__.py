"""Immich-style identity clustering for animal embeddings.

Public API: ``embed_gallery``, ``cluster``, ``cluster_quality``.
"""

from .clusterer import cluster
from .embedder import embed_gallery
from .metrics import cluster_quality

__all__ = [
    "embed_gallery",
    "cluster",
    "cluster_quality",
]
