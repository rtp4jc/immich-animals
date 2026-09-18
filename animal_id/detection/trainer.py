"""
Training utilities for YOLO detection models.

Provides configuration and training functionality for dog detection models.
"""

import logging
import sys
from typing import Any

import torch
from ultralytics import YOLO

from animal_id.common.constants import DETECTOR_PROJECT_DIR, DETECTOR_RUN_NAME

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """Trainer for YOLO detection models."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        config: dict[str, Any] | None = None,
        seed: int = 42,
    ):
        """Initialize trainer with model and configuration."""
        self.model_name = model_name
        self.seed = seed
        self.config = config or self._get_default_config()
        self.model = None

    def _get_default_config(self) -> dict[str, Any]:
        """Get default training configuration."""
        return {
            "data": "data/detector/dogs_detection.yaml",
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "project": str(DETECTOR_PROJECT_DIR),
            "name": DETECTOR_RUN_NAME,
            "device": 0,
            "patience": 10,
            "save_period": 5,
            "cache": False,
            "dropout": 0.1,
            "seed": self.seed,
            # Augmentation settings
            "fliplr": 0.5,
            "degrees": 15.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 15.0,
            "perspective": 0.001,
        }

    def verify_prerequisites(self) -> bool:
        """Verify system prerequisites for training."""
        logger.info("Verifying prerequisites...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if self.config.get("device") == "cpu":
            logger.warning("Training on CPU as requested.")
            return True

        if not torch.cuda.is_available():
            logger.error("CUDA not available. Training requires GPU.")
            return False

        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        )
        logger.info(f"Device: {device_name}")
        return True

    def load_model(self) -> None:
        """Load the YOLO model."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)

    def train(self) -> Any:
        """Train the detection model."""
        if not self.verify_prerequisites():
            raise RuntimeError("Prerequisites not met.")

        if self.model is None:
            self.load_model()

        logger.info(f"\nStarting detector training with config: {self.config}")

        # Train the model
        results = self.model.train(**self.config)

        logger.info("\nTraining completed.")
        logger.info(f"Results saved to {results.save_dir}")

        return results

    def update_config(self, **kwargs) -> None:
        """Update training configuration."""
        self.config.update(kwargs)


def train_detector(model_name: str = "yolo11n.pt", **config_overrides) -> Any:
    """Convenience function to train a detector with optional config overrides."""
    trainer = DetectionTrainer(model_name)
    if config_overrides:
        trainer.update_config(**config_overrides)
    return trainer.train()
