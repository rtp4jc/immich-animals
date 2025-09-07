#!/usr/bin/env python
"""
Prepares the embedding training dataset from DogFaceNet.

What it's for:
This script is the first step in the embedding model pipeline. It takes the raw
DogFaceNet dataset and converts it into a unified format that our PyTorch Dataset 
class can easily consume.

What it does:
1. Scans the DogFaceNet directory, treating each sub-folder as a unique dog identity.
2. Filters out identities that have fewer than a specified number of images.
3. Assigns a unique integer `identity_label` to each dog.
4. Splits the data by identity into training and validation sets to prevent data leakage.
5. Saves the results as JSON files that can be loaded by the PyTorch Dataset.

How to run it:
- Ensure DogFaceNet dataset is downloaded to the `data/` directory.
- Run the script from the root of the project:
  `python scripts/05_prepare_embedding_data.py`
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Configuration ---
DOGFACENET_PATH = 'data/dogfacenet/DogFaceNet_224resized/after_4_bis'
OUTPUT_TRAIN_JSON = 'data/identity_train.json'
OUTPUT_VAL_JSON = 'data/identity_val.json'
VAL_SPLIT_RATIO = 0.15
MIN_IMAGES_PER_IDENTITY = 5  # Only include identities with at least this many images

def create_identity_dataset():
    """
    Scans DogFaceNet for identities and creates train/validation JSON files.
    """
    print("--- Embedding Data Preparation ---")
    
    if not os.path.exists(DOGFACENET_PATH):
        print(f"[ERROR] DogFaceNet path not found: {DOGFACENET_PATH}")
        print("Please download the DogFaceNet dataset.")
        return

    print("Scanning DogFaceNet for dog identities...")
    
    # Scan DogFaceNet directory structure
    filtered_identities = {}
    for identity_folder in os.listdir(DOGFACENET_PATH):
        identity_path = os.path.join(DOGFACENET_PATH, identity_folder)
        if os.path.isdir(identity_path):
            image_files = [f for f in os.listdir(identity_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_files) >= MIN_IMAGES_PER_IDENTITY:
                filtered_identities[identity_folder] = [
                    os.path.join(identity_path, img) for img in image_files
                ]
    
    print(f"Found {len(filtered_identities)} identities with >= {MIN_IMAGES_PER_IDENTITY} images.")

    # Create final data list, grouped by identity
    data_by_identity = defaultdict(list)
    identity_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_identities.keys())}

    for old_identity_id, image_paths in filtered_identities.items():
        new_identity_id = identity_id_map[old_identity_id]
        for path in image_paths:
            data_by_identity[new_identity_id].append({
                'file_path': path.replace('\\', '/'),
                'identity_label': new_identity_id,
                'breed_label': 'unknown'  # No breed information needed
            })

    # Split data by identity to prevent data leakage
    print("\nSplitting data by identity to prevent leakage...")
    all_identity_ids = list(data_by_identity.keys())
    random.shuffle(all_identity_ids)

    train_data = []
    val_data = []
    total_images = sum(len(imgs) for imgs in data_by_identity.values())
    val_target_count = int(total_images * VAL_SPLIT_RATIO)

    for identity_id in all_identity_ids:
        # Add identities to validation set until we reach the desired ratio
        if len(val_data) < val_target_count:
            val_data.extend(data_by_identity[identity_id])
        else:
            train_data.extend(data_by_identity[identity_id])

    print(f"Total images: {total_images}")
    print(f"Target validation images: ~{val_target_count}")

    # Save to JSON
    print(f"Writing {len(train_data)} training samples to {OUTPUT_TRAIN_JSON}")
    with open(OUTPUT_TRAIN_JSON, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"Writing {len(val_data)} validation samples to {OUTPUT_VAL_JSON}")
    with open(OUTPUT_VAL_JSON, 'w') as f:
        json.dump(val_data, f, indent=2)

    print("Dataset preparation complete!")
    print(f"Training identities: {len([id for id in all_identity_ids if any(item['identity_label'] == id for item in train_data)])}")
    print(f"Validation identities: {len([id for id in all_identity_ids if any(item['identity_label'] == id for item in val_data)])}")

if __name__ == "__main__":
    create_identity_dataset()
