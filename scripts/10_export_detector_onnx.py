#!/usr/bin/env python
"""
Export Script for YOLOv11 Detector Model to ONNX

This script loads the best-performing detector model from the Phase 1 training
and exports it to the ONNX format for use in the inference pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

from dog_id.common.constants import (
    DETECTOR_PROJECT_DIR,
    DETECTOR_RUN_NAME,
    ONNX_DETECTOR_PATH,
)
from dog_id.common.utils import find_latest_run

def main():
    """Main export function."""
    print("--- YOLOv11 Detector ONNX Exporter ---")

    # --- Find latest training run ---
    print(f"Searching for latest run in: {DETECTOR_PROJECT_DIR}")
    latest_run_dir = find_latest_run(DETECTOR_PROJECT_DIR, DETECTOR_RUN_NAME)
    if not latest_run_dir:
        print(f"[ERROR] No training runs found for '{DETECTOR_RUN_NAME}' in '{DETECTOR_PROJECT_DIR}'.")
        sys.exit(1)
    
    model_checkpoint = latest_run_dir / "weights/best.pt"
    print(f"Found latest run: {latest_run_dir.name}")

    # --- Verification ---
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {YOLO._version}")

    if not model_checkpoint.exists():
        print(f"\n[ERROR] Model checkpoint not found at: {model_checkpoint}")
        print("Please ensure that the detector model has been trained and the path is correct.")
        sys.exit(1)

    print(f"\nLoading model from: {model_checkpoint}")

    # --- Load Model ---
    try:
        model = YOLO(model_checkpoint)
    except Exception as e:
        print(f"\n[ERROR] Failed to load the YOLO model: {e}")
        sys.exit(1)

    # --- Export to ONNX ---
    print(f"\nExporting model to ONNX format...")
    # This version of ultralytics saves the export in the same directory as the model
    try:
        exported_path_str = model.export(format="onnx", opset=12)
        exported_path = Path(exported_path_str)
        print("\nExport successful!")
    except Exception as e:
        print(f"\n[ERROR] Failed to export the model: {e}")
        sys.exit(1)

    # --- Move and Verify ---
    if not exported_path.exists():
        print(f"\n[ERROR] Export process completed, but the intermediate file was not found at {exported_path}.")
        sys.exit(1)
    
    # Ensure the final directory exists
    ONNX_DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Moving exported model to final destination: {ONNX_DETECTOR_PATH}")
    try:
        os.rename(exported_path, ONNX_DETECTOR_PATH)
    except OSError as e:
        print(f"\n[ERROR] Failed to move file: {e}")
        sys.exit(1)

    if ONNX_DETECTOR_PATH.exists():
        print(f"Verified: ONNX model created at {ONNX_DETECTOR_PATH}")
        print(f"File size: {ONNX_DETECTOR_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("\n[ERROR] Move process completed, but the final file was not found.")
        sys.exit(1)

    print("\n--- Exporter finished ---")



if __name__ == "__main__":
    main()
