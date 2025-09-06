"""
Training Script for YOLOv11 Object Detector

This script trains a YOLOv11 nano object detector on the dogs_detection dataset.
It is configured to only train for object detection (bounding boxes).
"""

import sys
import torch
from ultralytics import YOLO

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

    # Load a standard pretrained YOLOv11 nano model
    model = YOLO('yolo11n.pt')

    # Training configuration for the detector
    train_config = {
        'data': 'data/detector/dogs_detection.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'project': 'models/phase1',
        'name': 'detector_run',
        'device': 0,
        'patience': 10,
        'save_period': 5,
        'cache': False,
        'dropout': 0.1,
        # Augmentation settings
        'fliplr': 0.5,
        'degrees': 15.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 15.0,
        'perspective': 0.001,
    }

    print(f"\nStarting detector training with config: {train_config}")

    # Train the model
    results = model.train(**train_config)

    print(f"\nTraining completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()
