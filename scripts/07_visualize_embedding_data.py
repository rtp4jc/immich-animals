"""
Visualizes the validation identity dataset and shows statistics for both train and val sets.

What it's for:
This script is a crucial debugging and verification tool. It allows you to inspect the
dataset statistics for both training and validation sets, and visually inspect the
validation dataset to ensure label integrity.

What it does:
1. For both `identity_train.json` and `identity_val.json`, it calculates and prints:
   - Total number of unique identities.
   - Distribution of samples per identity.
   - Min, max, average, and median samples per identity.
2. For the validation set, it calls the centralized `visualize_identity_dataset` 
   function to generate a non-interactive plot, saving it to the `outputs` directory.

How to run it:
- This script should be run after `06_prepare_embedding_data.py`.
- Run from the root of the project:
  `python scripts/07_visualize_embedding_data.py`

How to interpret the results:
- The script will print statistics to the console for both datasets.
- It will save a plot for the validation set to `outputs/phase2_visualizations/identity_verification.png`.
- Each row in the plot is a unique dog, confirming dataset integrity.
"""

import json
import os
import sys
from collections import defaultdict, Counter
import numpy as np

# Adjust path to import from our new package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from animal_id.embedding.config import DATA_CONFIG
from animal_id.common.visualization import visualize_identity_dataset

# --- Configuration ---
NUM_IDENTITIES_TO_SHOW = 4
MIN_IMAGES_PER_ID = 3


def print_dataset_stats(json_path: str, dataset_name: str):
    """Loads a dataset JSON and prints statistics about its identities."""
    print(f"--- {dataset_name} Dataset Statistics ---")
    if not os.path.exists(json_path):
        print(f"Error: {dataset_name} JSON not found at {json_path}")
        print("Please run `scripts/06_prepare_embedding_data.py` first.")
        print("-------------------------------------\n")
        return

    print(f"Loading {dataset_name.lower()} dataset from {json_path}...")
    with open(json_path, "r") as f:
        annotations = json.load(f)
    print("Loading complete.")

    ids_to_images = defaultdict(list)
    for anno in annotations:
        ids_to_images[anno["identity_label"]].append(anno)

    num_identities = len(ids_to_images)
    print(f"Total unique identities: {num_identities}")

    sample_counts = [len(annos) for annos in ids_to_images.values()]

    if not sample_counts:
        print("No samples found to calculate stats.")
    else:
        count_distribution = Counter(sample_counts)
        print("\nDistribution of samples per identity:")
        for num_samples, count in sorted(count_distribution.items()):
            print(f"  - {count} identities have {num_samples} sample(s)")

        min_samples = min(sample_counts)
        max_samples = max(sample_counts)
        avg_samples = np.mean(sample_counts)
        median_samples = np.median(sample_counts)

        print(f"\nMin samples per identity: {min_samples}")
        print(f"Max samples per identity: {max_samples}")
        print(f"Average samples per identity: {avg_samples:.2f}")
        print(f"Median samples per identity: {median_samples}")
    print("-------------------------------------\n")


def main():
    """Main function to run the statistics and visualization."""
    train_json_path = DATA_CONFIG["TRAIN_JSON_PATH"]
    val_json_path = DATA_CONFIG["VAL_JSON_PATH"]

    # Process training set (stats only)
    print_dataset_stats(train_json_path, "Training")

    # Process validation set (stats and visualization)
    print_dataset_stats(val_json_path, "Validation")

    print("Generating identity verification plot for validation set...")
    # data_root is set to "." because file paths in the JSON are expected to be
    # absolute or relative to the project root.
    visualize_identity_dataset(
        identity_json_path=val_json_path,
        data_root=".",
        output_dir="outputs/phase2_visualizations",
        num_identities=NUM_IDENTITIES_TO_SHOW,
        min_images_per_id=MIN_IMAGES_PER_ID,
        display=False,  # Make it non-interactive
    )
    print("Visualization complete.")


if __name__ == "__main__":
    main()
