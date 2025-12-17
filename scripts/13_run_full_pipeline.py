#!/usr/bin/env python
"""
Runs the full pipeline benchmark.

This script is a simple, focused entry point that calls the master benchmark
function. All core logic is centralized in `train_master.py`.
"""
import argparse
import sys
from pathlib import Path

# Add project root to Python path to allow importing from `scripts`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.train_master import run_full_pipeline_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark AmbidextrousAxolotl pipeline."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to process from the validation set. If not specified, uses full dataset.",
    )
    parser.add_argument(
        "--include-additional",
        action="store_true",
        help="Include additional identities from data/additional_identities",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for the WandB run",
    )
    args = parser.parse_args()

    # All the complex logic for benchmarking is now in one place.
    run_full_pipeline_benchmark(
        num_images=args.num_images,
        include_additional=args.include_additional,
        tag=args.tag,
        no_wandb=args.no_wandb,
    )
    
    print("\nBenchmark completed successfully.")
