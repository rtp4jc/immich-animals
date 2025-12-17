#!/usr/bin/env python
"""
Export Script for YOLOv11 Detector Model to ONNX

This script finds the latest trained detector model and calls the centralized
export function to convert it to ONNX format.
"""
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.train_master import run_detector_export
from animal_id.common.constants import DETECTOR_PROJECT_DIR, DETECTOR_RUN_NAME
from animal_id.common.utils import find_latest_run

def main():
    """Main export function."""
    print("--- YOLOv11 Detector ONNX Exporter ---")

    # --- Find latest training run ---
    print(f"Searching for latest run in: {DETECTOR_PROJECT_DIR}")
    latest_run_dir = find_latest_run(DETECTOR_PROJECT_DIR, DETECTOR_RUN_NAME)
    if not latest_run_dir:
        raise FileNotFoundError(
            f"No training runs found for '{DETECTOR_RUN_NAME}' in '{DETECTOR_PROJECT_DIR}'."
        )

    model_checkpoint = latest_run_dir / "weights/best.pt"
    print(f"Found latest model checkpoint: {model_checkpoint}")

    # Call the centralized export function
    run_detector_export(model_checkpoint)

    print("\n--- Exporter finished ---")


if __name__ == "__main__":
    main()

