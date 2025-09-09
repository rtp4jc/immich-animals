#!/usr/bin/env python3
"""
Train Detection Model

Trains YOLO detection model using the prepared dataset.
Uses extracted dog_id.detection.trainer module.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dog_id.detection.trainer import DetectionTrainer


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train detection model")
    parser.add_argument("--model", default="yolo11n.pt", 
                       help="Base model to use (default: yolo11n.pt)")
    parser.add_argument("--data", default="data/detector/dogs_detection.yaml",
                       help="Dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    args = parser.parse_args()

    print("=" * 60)
    print("Training Detection Model")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")

    # Create trainer with custom config
    trainer = DetectionTrainer(args.model)
    trainer.update_config(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )

    # Train the model
    results = trainer.train()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
