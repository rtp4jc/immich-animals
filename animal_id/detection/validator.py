"""
Validation utilities for YOLO detection models.

Provides model validation and inference functionality for dog detection models.
"""

import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from ultralytics import YOLO


class DetectionValidator:
    """Validator for YOLO detection models."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize validator with optional model path."""
        self.model_path = model_path
        self.model = None

    def find_latest_detector_model(self, model_dir: str = "models/phase1") -> str:
        """Find the latest model weights (best.pt) from the detector_run directory."""
        # Find the latest directory inside models/phase1 that starts with detector_run
        list_of_dirs = glob.glob(os.path.join(model_dir, "detector_run*"))
        if not list_of_dirs:
            raise FileNotFoundError(f"No detector run found in {model_dir}")
        latest_dir = max(list_of_dirs, key=os.path.getmtime)

        model_path = os.path.join(latest_dir, "weights", "best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No best.pt model found in {latest_dir}")

        return model_path

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load model for validation."""
        if model_path:
            self.model_path = model_path
        elif not self.model_path:
            self.model_path = self.find_latest_detector_model()

        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)

    def validate_model(
        self, data_yaml: str = "data/detector/dogs_detection.yaml"
    ) -> Tuple[YOLO, Dict[str, float]]:
        """Run validation on the model and return metrics."""
        if self.model is None:
            self.load_model()

        print("Running validation on the detector model...")
        results = self.model.val(data=data_yaml, verbose=True)

        # Metrics for an object detector
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

        return self.model, metrics

    def run_inference(
        self,
        image_paths: list,
        output_dir: str = "outputs/detection_inference",
        conf_threshold: float = 0.5,
    ) -> list:
        """Run inference on a list of images."""
        if self.model is None:
            self.load_model()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for image_path in image_paths:
            result = self.model(image_path, conf=conf_threshold)
            # Save visualization
            result[0].save(output_path / f"result_{Path(image_path).name}")
            results.append(result[0])

        print(f"Inference results saved to: {output_path}")
        return results


def validate_latest_detector(
    data_yaml: str = "data/detector/dogs_detection.yaml",
) -> Tuple[YOLO, Dict[str, float]]:
    """Convenience function to validate the latest trained detector."""
    validator = DetectionValidator()
    return validator.validate_model(data_yaml)
