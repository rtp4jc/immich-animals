#!/usr/bin/env python3
"""
export_model.py - Export the trained YOLOv8 pose model to other formats.

Prompt 7: Export the trained model to ONNX and TFLite (basic)

This script finds the latest trained model (best.pt) and exports it to
ONNX and TensorFlow Lite formats for deployment.
"""

import os
import glob
from pathlib import Path
from ultralytics import YOLO

def find_latest_model():
    """Find the latest model weights (best.pt) from phase1 runs"""
    model_dir = "models/phase1"
    pattern = os.path.join(model_dir, "*", "weights", "best.pt")
    model_paths = glob.glob(pattern)
    if not model_paths:
        raise FileNotFoundError("No best.pt model found in models/phase1/")
    model_paths.sort(key=os.path.getmtime, reverse=True)
    return model_paths[0]

def main():
    """Main export function."""
    print("=" * 60)
    print("Exporting the latest trained model to ONNX and TFLite")
    print("=" * 60)

    try:
        model_path = find_latest_model()
        print(f"Found latest model: {model_path}")
    except FileNotFoundError as e:
        print(e)
        return

    model = YOLO(model_path)

    export_dir = Path("models/phase1/export")
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to directory: {export_dir.resolve()}")

    # Export to ONNX
    try:
        print("\nExporting to ONNX format...")
        onnx_path = model.export(format='onnx', imgsz=640)
        print(f"ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

    # Export to TFLite
    try:
        print("\nExporting to TFLite format...")
        tflite_path = model.export(format='tflite', imgsz=640)
        print(f"TFLite export successful: {tflite_path}")
    except Exception as e:
        print(f"Error during TFLite export: {e}")

    print(f"\nExport complete. Models saved in {export_dir.resolve()}")
    print("=" * 60)

if __name__ == '__main__':
    main()
