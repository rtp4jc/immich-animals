import json
import os
import random
from collections import defaultdict


class EmbeddingDatasetConverter:
    """
    Prepares the embedding training dataset from DogFaceNet.
    """

    def __init__(
        self,
        source_path: str,
        output_train_json: str,
        output_val_json: str,
        min_images_per_identity: int = 5,
        val_split_ratio: float = 0.15,
    ):
        self.source_path = source_path
        self.output_train_json = output_train_json
        self.output_val_json = output_val_json
        self.min_images_per_identity = min_images_per_identity
        self.val_split_ratio = val_split_ratio

    def convert(self):
        """
        Scans DogFaceNet for identities and creates train/validation JSON files.
        """
        print("--- Embedding Data Preparation ---")

        if not os.path.exists(self.source_path):
            print(f"[ERROR] DogFaceNet path not found: {self.source_path}")
            print("Please download the DogFaceNet dataset.")
            return

        print("Scanning DogFaceNet for dog identities...")

        # Scan DogFaceNet directory structure
        filtered_identities = {}
        for identity_folder in os.listdir(self.source_path):
            identity_path = os.path.join(self.source_path, identity_folder)
            if os.path.isdir(identity_path):
                image_files = [
                    f
                    for f in os.listdir(identity_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if len(image_files) >= self.min_images_per_identity:
                    filtered_identities[identity_folder] = [
                        os.path.join(identity_path, img) for img in image_files
                    ]

        print(
            f"Found {len(filtered_identities)} identities with >= {self.min_images_per_identity} images."
        )

        # Create final data list, grouped by identity
        data_by_identity = defaultdict(list)
        identity_id_map = {
            old_id: new_id for new_id, old_id in enumerate(filtered_identities.keys())
        }

        for old_identity_id, image_paths in filtered_identities.items():
            new_identity_id = identity_id_map[old_identity_id]
            for path in image_paths:
                data_by_identity[new_identity_id].append(
                    {
                        "file_path": path.replace("\\", "/"),
                        "identity_label": new_identity_id,
                        "breed_label": "unknown",  # No breed information needed
                    }
                )

        # Split data by identity to prevent data leakage
        print("\nSplitting data by identity to prevent leakage...")
        all_identity_ids = list(data_by_identity.keys())
        random.shuffle(all_identity_ids)

        train_data = []
        val_data = []
        total_images = sum(len(imgs) for imgs in data_by_identity.values())
        val_target_count = int(total_images * self.val_split_ratio)

        for identity_id in all_identity_ids:
            # Add identities to validation set until we reach the desired ratio
            if len(val_data) < val_target_count:
                val_data.extend(data_by_identity[identity_id])
            else:
                train_data.extend(data_by_identity[identity_id])

        print(f"Total images: {total_images}")
        print(f"Target validation images: ~{val_target_count}")

        # Ensure output directories exist
        os.makedirs(os.path.dirname(self.output_train_json), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_val_json), exist_ok=True)

        # Save to JSON
        print(f"Writing {len(train_data)} training samples to {self.output_train_json}")
        with open(self.output_train_json, "w") as f:
            json.dump(train_data, f, indent=2)

        print(f"Writing {len(val_data)} validation samples to {self.output_val_json}")
        with open(self.output_val_json, "w") as f:
            json.dump(val_data, f, indent=2)

        print("Dataset preparation complete!")
        print(
            f"Training identities: {len([id for id in all_identity_ids if any(item['identity_label'] == id for item in train_data)])}"
        )
        print(
            f"Validation identities: {len([id for id in all_identity_ids if any(item['identity_label'] == id for item in val_data)])}"
        )
