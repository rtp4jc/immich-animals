"""Protocols for the detection, keypoint and embedding stages of the pipeline."""

from enum import Enum
from typing import Any, Protocol

import numpy as np


class AnimalClass(Enum):
    """Animal classification results."""

    NO_ANIMAL = "no_animal"
    DOG = "dog"
    CAT = "cat"
    BIRD = "bird"
    OTHER = "other"


class DetectionModel(Protocol):
    """Protocol for animal detection models."""

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect animals in an RGB image.

        Returns one dict per detection with 'bbox' [x1, y1, x2, y2], 'confidence'
        and 'class'.
        """
        ...


class KeypointModel(Protocol):
    """Protocol for animal keypoint models."""

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect keypoints in a cropped animal image.

        Returns one dict per detection with 'keypoints' [[x, y, conf], ...] and
        'confidence'.
        """
        ...


class EmbeddingPipeline(Protocol):
    """Protocol for end-to-end pipelines that embed an image file."""

    def generate_embedding(self, image_path: str) -> np.ndarray | None:
        """Embed the image at image_path, or None if no target animal was detected."""
        ...


class EmbeddingModel(Protocol):
    """Protocol for animal embedding models."""

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate an embedding vector for an animal image crop."""
        ...
