#!/usr/bin/env python3
"""
export_model.py - Export trained YOLOv8 pose model to ONNX and TFLite formats.

This script loads the best trained model weights and exports them to ONNX and TFLite
for deployment purposes. It verifies CUDA availability and exports to the specified
save directory.
"""

import os
import sys
from ultralytics import YOLO
import torch

def main():
    # Verify CUDA availability early
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Export may fail or be slow.")
        sys.exit(1)
    print("CUDA available. Proceeding with export.")

    # Model path (use the latest pose_run14 based on available models)
    model_path = 'models/phase1/pose_run14/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found.")
        sys.exit(1)

    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # Export directory
    save_dir = 'models/phase1/export'

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Export to ONNX
    try:
        print("Exporting to ONNX...")
        model.export(format='onnx', imgsz=640, save_dir=save_dir)
        print("ONNX export completed.")
    except Exception as e:
        print(f"ERROR exporting to ONNX: {e}")

    # Export to TFLite
    try:
        print("Exporting to TFLite...")
        model.export(format='tflite', imgsz=640, save_dir=save_dir)
        print("TFLite export completed.")
    except Exception as e:
        print(f"ERROR exporting to TFLite: {e}")
        print("Note: TFLite export may fail due to unsupported ops; ONNX model saved.")

    # Verify exports
    onnx_path = os.path.join(save_dir, 'model.onnx')
    tflite_path = os.path.join(save_dir, 'model.tflite')

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX model saved at {onnx_path} ({size_mb:.2f} MB)")
    else:
        print("WARNING: ONNX model not found after export.")

    if os.path.exists(tflite_path):
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"TFLite model saved at {tflite_path} ({size_mb:.2f} MB)")
    else:
        print("WARNING: TFLite model not found after export.")

if __name__ == '__main__':
    main()