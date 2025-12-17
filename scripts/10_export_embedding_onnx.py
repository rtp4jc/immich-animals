#!/usr/bin/env python
"""
Export Script for Embedding Model to ONNX

This script finds the latest trained embedding model and calls the centralized
export function to evaluate and convert it to ONNX format.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.train_master import run_embedding_export
from animal_id.common.constants import MODELS_DIR
from animal_id.common.utils import find_latest_timestamped_run
from animal_id.embedding.config import DATA_CONFIG, DEFAULT_BACKBONE
from animal_id.embedding.backbones import BackboneType
from animal_id.common.datasets import DogIdentityDataset


def main(args):
    """Main export function."""
    print("--- Embedding Model ONNX Exporter ---")

    # --- Find latest training run ---
    latest_run = find_latest_timestamped_run()
    model_path = None

    if latest_run:
        model_path = latest_run / "best_model.pt"
        if not model_path.exists():
            model_path = None

    # Fall back to old location if not found
    if model_path is None:
        model_path = MODELS_DIR / "dog_embedding_best.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found. Checked runs/*/best_model.pt and {MODELS_DIR / 'dog_embedding_best.pt'}"
            )
    
    print(f"Found latest model checkpoint: {model_path}")

    # The export function requires a dataloader and num_classes for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = DogIdentityDataset(
        json_path=DATA_CONFIG["VAL_JSON_PATH"],
        img_size=DATA_CONFIG["IMG_SIZE"],
        is_training=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=DATA_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2
    )
    
    # We also need num_classes from the *training* set to initialize the model
    train_dataset = DogIdentityDataset(
        json_path=DATA_CONFIG["TRAIN_JSON_PATH"],
        img_size=DATA_CONFIG["IMG_SIZE"],
        is_training=True,
    )
    num_classes = train_dataset.num_classes
    
    # Call the centralized export function
    run_embedding_export(model_path, val_loader, device, num_classes)

    print("\n--- Exporter finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained embedding model to ONNX."
    )
    parser.add_argument(
        "--backbone",
        type=BackboneType,
        default=DEFAULT_BACKBONE,
        choices=list(BackboneType),
        help=f"Backbone of the trained model to export. Default: {DEFAULT_BACKBONE.value}",
    )
    args = parser.parse_args()
    main(args)

