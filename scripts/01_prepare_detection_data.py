#!/usr/bin/env python
"""
Runs the data preparation pipeline for the detection model.

This script is a simple, focused entry point that calls the master data prep
function. All core logic is centralized in `train_master.py`.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow importing from `scripts`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_master import run_detection_data_prep  # noqa: E402

if __name__ == "__main__":
    # All the complex logic for data prep is now in one place.
    # This script is just a simple entry point to run only the detection data prep.
    run_detection_data_prep()
    print("Detection data preparation completed successfully.")
