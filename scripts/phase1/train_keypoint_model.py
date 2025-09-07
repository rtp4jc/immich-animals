"""
Training Script for YOLOv11 Keypoint Estimator

This script trains a YOLOv11 nano pose model on the cropped keypoints dataset.
It is configured to learn our custom 4-point schema by building the model
from a custom YAML configuration file.
"""

import sys
import torch
from ultralytics import YOLO

from dog_id.common.constants import KEYPOINT_PROJECT_DIR, KEYPOINT_RUN_NAME

def main():
    """Main training function."""

    print("Verifying prerequisites...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        sys.exit("Error: CUDA not available. Training requires GPU.")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"Device: {device_name}")

    # Load the model architecture from our custom YAML file.
    # This creates a new model from scratch with the correct number of classes and keypoints.
    model = YOLO('yolo11n-dog-pose.yaml')

    # Training configuration for the keypoint estimator
    train_config = {
        'data': 'data/keypoints/dogs_keypoints_only.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'project': str(KEYPOINT_PROJECT_DIR),
        'name': KEYPOINT_RUN_NAME, # New name for this run
        'device': 0,
        'patience': 10,
        'save_period': 5,
        'cache': False,
        # Augmentation settings
        'fliplr': 0.5,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 10.0,
        'perspective': 0.0005,
    }

    print(f"\nStarting keypoint model training with config: {train_config}")

    # Train the model
    # Note: Since we are building from a YAML, this will train from scratch.
    # For transfer learning, you would use .load('yolov11n-pose.pt') after init.
    results = model.train(**train_config)

    print(f"\nTraining completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()
