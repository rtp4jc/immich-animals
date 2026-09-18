"""
Base model protocols for animal identification pipeline.

Defines interfaces for detection, keypoint, and embedding models.
"""

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
        """
        Detect animals in image.

        Args:
            image: RGB image as numpy array

        Returns:
            List of detections with 'bbox' [x1, y1, x2, y2], 'confidence', and 'class'
        """
        ...


class KeypointModel(Protocol):
    """Protocol for animal keypoint models."""

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect keypoints in cropped animal image.

        Args:
            image: RGB image crop containing animal

        Returns:
            List of keypoint detections with 'keypoints' [[x, y, conf], ...] and 'confidence'
        """
        ...


class EmbeddingPipeline(Protocol):
    """Protocol for end-to-end pipelines that embed an image file."""

    def generate_embedding(self, image_path: str) -> np.ndarray | None:
        """
        Generate an embedding for the image at image_path.

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector, or None if no target animal was detected
        """
        ...


class EmbeddingModel(Protocol):
    """Protocol for animal embedding models."""

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding for animal image crop.

        Args:
            image: RGB image crop containing animal

        Returns:
            Embedding vector as numpy array
        """
        ...
