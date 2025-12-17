#!/usr/bin/env python
"""
Runs the data preparation pipeline for the embedding model.

This script is a simple, focused entry point that calls the master data prep
function. All core logic is centralized in `train_master.py`.
"""

# Add project root to Python path to allow importing from `scripts`

from scripts.train_master import run_embedding_data_prep

if __name__ == "__main__":
    # All the complex logic for data prep is now in one place.
    # This script is just a simple entry point to run only the embedding data prep.
    run_embedding_data_prep()
    print("Embedding data preparation completed successfully.")
