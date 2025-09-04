#!/usr/bin/env python3
"""
Creates a sample_images directory from the validation set.

This script selects a small number of images from the validation set
(as defined in data/coco_keypoints/annotations_val.json) and copies them to
outputs/phase1/sample_images/. These images are intended for quick visual
inspection of model inference.
"""

import os
import random
import shutil
import json
from pathlib import Path

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
COCO_ANNOTATIONS_VAL = DATA_ROOT / "coco_keypoints" / "annotations_val.json"
SAMPLE_IMAGES_DIR = REPO_ROOT / "outputs" / "phase1" / "sample_images"

NUM_SAMPLES_PER_DATASET = 10 # Number of samples to take from each source dataset

def main():
    print("=" * 60)
    print("Creating sample images from validation set")
    print("=" * 60)

    if not COCO_ANNOTATIONS_VAL.exists():
        print(f"Error: Validation annotations file not found: {COCO_ANNOTATIONS_VAL}")
        print("Please run 'python scripts/phase1/convert_to_coco_keypoints.py' first.")
        return

    # Clear existing sample images
    if SAMPLE_IMAGES_DIR.exists():
        print(f"Clearing existing sample images in {SAMPLE_IMAGES_DIR}...")
        shutil.rmtree(SAMPLE_IMAGES_DIR)
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Read validation image paths from COCO JSON
    with open(COCO_ANNOTATIONS_VAL, 'r') as f:
        coco_data = json.load(f)
    
    val_images_info = coco_data['images']

    if not val_images_info:
        print("Error: No image information found in annotations_val.json.")
        return

    # Group images by source dataset dynamically
    images_by_source = {}
    for img_info in val_images_info:
        source = img_info.get('source_dataset', 'unknown')
        if source not in images_by_source:
            images_by_source[source] = []
        images_by_source[source].append(img_info['file_name'])
        # Add other sources if necessary

    copied_count = 0
    for source, paths in images_by_source.items():
        if not paths:
            print(f"No {source} images found in validation set.")
            continue

        # Select random samples
        selected_samples = random.sample(paths, min(len(paths), NUM_SAMPLES_PER_DATASET))
        print(f"Copying {len(selected_samples)} samples from {source}...")

        for i, relative_img_path_str in enumerate(selected_samples):
            # Construct absolute source path
            src_path = DATA_ROOT / relative_img_path_str
            # Construct destination path with sequential naming
            dst_path = SAMPLE_IMAGES_DIR / f"{source}_{i+1:03d}{Path(relative_img_path_str).suffix}"
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Warning: Could not copy {src_path}: {e}")

    print(f"\nSuccessfully copied {copied_count} sample images to {SAMPLE_IMAGES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
