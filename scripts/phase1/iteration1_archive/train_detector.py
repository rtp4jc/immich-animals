"""
YOLOv8 Pose Model Training Script for Dog Keypoints Detection

This script trains a YOLOv8 nano pose model on the dogs_keypoints dataset.
It uses Ultralytics YOLO API for training with minimal configuration.

Prerequisites:
- Python 3.12+ in conda env 'python312'
- PyTorch with CUDA support
- Ultralytics YOLOv8 installed
- Data file: data/dogs_keypoints.yaml (containing train/val/test paths)

Usage:
    python scripts/phase1/train_detector.py
"""

import sys
import torch
from ultralytics import YOLO


def main():
    """Main training function."""

    # Verify prerequisites
    print("Verifying prerequisites...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        sys.exit("Error: CUDA not available. Training requires GPU.")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"Device: {device_name}")

    # Load pretrained YOLOv8 pose model
    model = YOLO('yolov11n-pose.yaml')

    # Training configuration for prototype
    # Small epochs for initial testing - user can extend
    train_config = {
        'data': 'data/dogs_keypoints.yaml',
        'epochs': 100,  # Increased epochs for better learning
        'imgsz': 640,
        'batch': 8,
        'project': 'models/phase1',
        'name': 'pose_run_augmented', # New name for the run
        'device': 0,
        'patience': 10, # Increased patience
        'save_period': 5,
        'cache': False,
        # Augmentation settings from https://docs.ultralytics.com/guides/yolo-data-augmentation
        'fliplr': 0.5,   # Horizontal flip (50% probability)
        'degrees': 10,   # Rotation augmentation (+/- 10 degrees)
        'translate': 0.1, # Translation augmentation (+/- 10%)
        'scale': 0.5,    # Scale augmentation (+/- 20%)
        'shear': 10.0,    # Shear augmentation (+/- 10 degrees)
        'perspective': 0.0005,

    }

    print(f"\nStarting training with config: {train_config}")

    # Train the model
    results = model.train(**train_config)

    print(f"\nTraining completed.")
    print(f"Results: {results}")

    # The results object contains paths to results
    # User should check results in the project directory (models/phase1/pose_run1/)
    # and run validation separately for detailed metrics


if __name__ == '__main__':
    main()