"""
Training utilities for YOLO keypoint models.

Provides configuration and training functionality for dog keypoint detection models.
"""

import sys
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, Optional

from dog_id.common.constants import KEYPOINT_PROJECT_DIR, KEYPOINT_RUN_NAME


class KeypointTrainer:
    """Trainer for YOLO keypoint models."""
    
    def __init__(self, model_name: str = 'yolo11n-dog-pose.yaml', config: Optional[Dict[str, Any]] = None):
        """Initialize trainer with model and configuration."""
        self.model_name = model_name
        self.config = config or self._get_default_config()
        self.model = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'data': 'data/keypoints/dogs_keypoints_only.yaml',
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'project': str(KEYPOINT_PROJECT_DIR),
            'name': KEYPOINT_RUN_NAME,
            'device': 0,
            'patience': 10,
            'save_period': 5,
            'cache': False,
            # Augmentation settings (more conservative for keypoints)
            'fliplr': 0.5,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 10.0,
            'perspective': 0.0005,
        }
    
    def verify_prerequisites(self) -> bool:
        """Verify system prerequisites for training."""
        print("Verifying prerequisites...")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("Error: CUDA not available. Training requires GPU.")
            return False

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        print(f"Device: {device_name}")
        return True
    
    def load_model(self) -> None:
        """Load the YOLO model."""
        print(f"Loading model: {self.model_name}")
        # Load the model architecture from custom YAML file
        # This creates a new model from scratch with the correct number of keypoints
        self.model = YOLO(self.model_name)
    
    def train(self) -> Any:
        """Train the keypoint model."""
        if not self.verify_prerequisites():
            sys.exit("Prerequisites not met.")
        
        if self.model is None:
            self.load_model()
        
        print(f"\nStarting keypoint model training with config: {self.config}")
        
        # Train the model
        # Note: Since we are building from a YAML, this will train from scratch.
        # For transfer learning, you would use .load('yolov11n-pose.pt') after init.
        results = self.model.train(**self.config)
        
        print(f"\nTraining completed.")
        print(f"Results saved to {results.save_dir}")
        
        return results
    
    def update_config(self, **kwargs) -> None:
        """Update training configuration."""
        self.config.update(kwargs)


def train_keypoint_model(model_name: str = 'yolo11n-dog-pose.yaml', **config_overrides) -> Any:
    """Convenience function to train a keypoint model with optional config overrides."""
    trainer = KeypointTrainer(model_name)
    if config_overrides:
        trainer.update_config(**config_overrides)
    return trainer.train()
