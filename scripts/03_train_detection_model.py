#!/usr/bin/env python
"""
Runs the full detection model training and export pipeline.

This script is a simple, focused entry point that calls the master detection
pipeline function. All core logic is centralized in `train_master.py`.
"""
import sys
from pathlib import Path

# Add project root to Python path to allow importing from `scripts`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.train_master import run_detection_pipeline

if __name__ == "__main__":
    # All the complex logic for data prep, training, and export is in one place.
    # This script is just a simple entry point to run only the detection pipeline.
    run_detection_pipeline()
    print("Detection pipeline completed successfully.")
