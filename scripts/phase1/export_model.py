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
import shutil

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
    print(f"Ensuring export directory exists: {export_dir.resolve()}")

    # Export to ONNX
    try:
        print("\nExporting to ONNX format...")
        # The export function saves next to the original model, so we move it after.
        source_onnx_path_str = model.export(format='onnx', imgsz=640)
        source_onnx_path = Path(source_onnx_path_str)
        
        destination_onnx_path = export_dir / source_onnx_path.name
        
        # Move the file to our desired export directory
        shutil.move(str(source_onnx_path), str(destination_onnx_path))
        
        print(f"ONNX export successful. Model moved to: {destination_onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

    # Export to TFLite (currently failing, but let's keep the logic)
    try:
        print("\nExporting to TFLite format...")
        source_tflite_path_str = model.export(format='tflite', imgsz=640)
        source_tflite_path = Path(source_tflite_path_str)
        
        destination_tflite_path = export_dir / source_tflite_path.name
        
        shutil.move(str(source_tflite_path), str(destination_tflite_path))
        print(f"TFLite export successful. Model moved to: {destination_tflite_path}")
    except Exception as e:
        print(f"Error during TFLite export: {e}")

    print(f"\nExport complete. Check {export_dir.resolve()} for models.")
    print("=" * 60)

if __name__ == '__main__':
    main()