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
    model = YOLO('yolov8n-pose.pt')

    # Training configuration for prototype
    # Small epochs for initial testing - user can extend
    train_config = {
        'data': 'data/dogs_keypoints.yaml',
        'epochs': 10,  # Prototype: small epochs
        'imgsz': 640,
        'batch': 8,    # Small batch for testing
        'project': 'models/phase1',
        'name': 'pose_run1',
        'device': 0,   # Use first GPU
        'patience': 5, # Early stopping patience
        'save_period': 5,  # Save every 5 epochs
        'cache': False,    # Disable caching for debugging
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