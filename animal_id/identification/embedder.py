"""Embed a gallery of labeled images via a pipeline's ``generate_embedding``."""

from typing import Any

import numpy as np
from tqdm import tqdm


def embed_gallery(
    pipeline: Any,
    items: list[dict],
    show_progress: bool = True,
) -> tuple[np.ndarray, list, list[str]]:
    """Embed labeled images, returning aligned ``(embeddings, labels, paths)``.

    ``pipeline`` is any object with ``generate_embedding(image_path) -> ndarray |
    None``; ``items`` are dicts with ``"image_path"`` / ``"identity_label"`` keys.
    Items whose embedding is ``None`` are dropped, keeping the three outputs
    aligned. ``embeddings`` is an ``(N, D)`` float32 array.
    """
    embeddings: list[np.ndarray] = []
    labels: list = []
    paths: list[str] = []

    iterable = tqdm(items, desc="Embedding gallery") if show_progress else items
    for item in iterable:
        image_path: str = item["image_path"]
        embedding = pipeline.generate_embedding(image_path)
        if embedding is None:
            continue
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        labels.append(item["identity_label"])
        paths.append(image_path)

    if embeddings:
        return np.stack(embeddings, axis=0), labels, paths

    return np.empty((0,), dtype=np.float32), labels, paths
