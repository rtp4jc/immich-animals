#!/usr/bin/env python3
"""
Converts a COCO JSON dataset to the YOLOv8 detection format.

This script reads the detector-specific COCO JSON annotation files and
generates a corresponding set of YOLO `.txt` label files and the final
`dogs_detection.yaml` file required for training.
"""

import os
import json
from pathlib import Path
import yaml
from tqdm import tqdm
import shutil
import glob

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
# Read from the new detector-specific COCO directory
COCO_ANNOTATIONS_DIR = DATA_ROOT / "detector/coco"

# The directory where the YOLO `.txt` label files will be created.
# This remains DATA_ROOT to ensure `labels` directories are parallel to `images`
LABELS_OUTPUT_DIR = DATA_ROOT

# Save the final YAML file in the new detector directory
YOLO_YAML_PATH = DATA_ROOT / "detector/dogs_detection.yaml"

def convert_split(split_name):
    """Processes a single split (e.g., 'train' or 'val')."""
    coco_json_path = COCO_ANNOTATIONS_DIR / f"annotations_{split_name}.json"

    if not coco_json_path.exists():
        print(f"Warning: Annotation file not found, skipping split '{split_name}': {coco_json_path}")
        return []

    print(f"Processing {split_name} split from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images_map = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    image_paths = []
    found_annotations_count = 0
    written_labels_count = 0
    for img_id, image_info in tqdm(images_map.items(), desc=f"Generating {split_name} labels"):
        img_height = image_info['height']
        img_width = image_info['width']
        relative_img_path = image_info['file_name']
        
        relative_img_path_str = str(Path(relative_img_path))
        # With the new standardized directory structure, we can use a single, simple replacement.
        relative_label_path_str = relative_img_path_str.replace('images', 'labels', 1)
        label_path = (LABELS_OUTPUT_DIR / Path(relative_label_path_str)).with_suffix('.txt')

        image_paths.append(str(DATA_ROOT / relative_img_path))
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, 'w') as f_label:
            if img_id in annotations_by_image:
                found_annotations_count += 1
                annotations = annotations_by_image[img_id]
                if len(annotations) > 0:
                    written_labels_count += 1
                for ann in annotations:
                    bbox = ann['bbox']
                    x, y, w, h = bbox

                    # Redundant clamping to definitively fix any stale data issues
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(img_width, x + w)
                    y2 = min(img_height, y + h)

                    # Log if clamping was necessary
                    if x1 != x or y1 != y or x2 != (x + w) or y2 != (y + h):
                        print(f"[WARN] Clamped bbox for {relative_img_path}. Original: {[x, y, w, h]}, Clamped: {[x1, y1, x2 - x1, y2 - y1]}")

                    final_w = x2 - x1
                    final_h = y2 - y1

                    if final_w <= 0 or final_h <= 0: continue

                    x_center_norm = (x1 + final_w / 2) / img_width
                    y_center_norm = (y1 + final_h / 2) / img_height
                    width_norm = final_w / img_width
                    height_norm = final_h / img_height

                    f_label.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

    return image_paths

def create_yaml_config(train_image_paths, val_image_paths):
    """Creates the final YOLOv8 dataset YAML configuration file for detection."""
    # Use new detector-specific train/val files in the detector directory
    train_txt_path = DATA_ROOT / "detector/train.txt"
    val_txt_path = DATA_ROOT / "detector/val.txt"

    with open(train_txt_path, 'w') as f:
        for path in sorted(train_image_paths):
            f.write(f"{Path(path).as_posix()}\n")
    print(f"Created {train_txt_path.name} with {len(train_image_paths)} image paths.")

    with open(val_txt_path, 'w') as f:
        for path in sorted(val_image_paths):
            f.write(f"{Path(path).as_posix()}\n")
    print(f"Created {val_txt_path.name} with {len(val_image_paths)} image paths.")

    # Create YAML for detection, pointing to the new train/val lists
    yaml_content = {
        'path': Path(DATA_ROOT.resolve()).as_posix(),
        'train': Path(train_txt_path.resolve()).as_posix(),
        'val': Path(val_txt_path.resolve()).as_posix(),
        'nc': 1,
        'names': ['dog'],
    }

    with open(YOLO_YAML_PATH, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"Successfully created YAML config for detection at: {YOLO_YAML_PATH}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Converting COCO Detector Dataset to YOLOv8 Detection Format")
    print("=" * 60)

    train_paths = convert_split('train')
    val_paths = convert_split('val')

    if not train_paths and not val_paths:
        print(f"Error: No data was processed. Check that your COCO JSON files exist in {COCO_ANNOTATIONS_DIR}")
        return

    create_yaml_config(train_paths, val_paths)

    print("\nConversion complete!")
    print("You are now ready to train the YOLOv8 detector model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
