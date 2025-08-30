#!/usr/bin/env python3
"""
Converts a dataset from COCO JSON keypoint format to the YOLOv8 pose format.

This script reads COCO JSON annotation files (for train and val splits) and
generates a corresponding set of YOLO `.txt` label files, where each file
corresponds to an image and contains the normalized bounding box and keypoint
data.

It also creates the final `dogs_keypoints.yaml` file required for training.
"""

import os
import json
from pathlib import Path
import yaml
from tqdm import tqdm

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
COCO_ANNOTATIONS_DIR = DATA_ROOT / "coco_keypoints"

# The directory where the YOLO `.txt` label files will be created.
# The structure will mirror the image directories.
LABELS_OUTPUT_DIR = DATA_ROOT

YOLO_YAML_PATH = DATA_ROOT / "dogs_keypoints.yaml"

def convert_split(split_name):
    """Processes a single split (e.g., 'train' or 'val')."""
    coco_json_path = COCO_ANNOTATIONS_DIR / f"annotations_{split_name}.json"

    if not coco_json_path.exists():
        print(f"Warning: Annotation file not found, skipping split '{split_name}': {coco_json_path}")
        return []

    print(f"Processing {split_name} split from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from image ID to image info (filename, width, height)
    images_map = {img['id']: img for img in coco_data['images']}
    
    # Clear existing label files in target directories to prevent stale data
    # (This is important if you re-run the script)
    existing_labels = set()

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    image_paths = []
    for img_id, annotations in tqdm(annotations_by_image.items(), desc=f"Generating {split_name} labels"):
        if img_id not in images_map:
            continue

        image_info = images_map[img_id]
        img_height = image_info['height']
        img_width = image_info['width']
        relative_img_path = image_info['file_name']
        
        # Determine the output path for the label file
        # It should be in a 'labels' subdirectory relative to the image
        label_path = (LABELS_OUTPUT_DIR / Path(relative_img_path)).parent.parent / "labels" / f"{Path(relative_img_path).stem}.txt"
        
        # Add to the list of processed images for the final txt file
        image_paths.append(str(DATA_ROOT / relative_img_path))

        # Ensure the output directory for the label file exists
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add to set of labels to clear later if needed (optional)
        existing_labels.add(label_path)

        with open(label_path, 'w') as f_label:
            for ann in annotations:
                # Normalize bounding box coordinates
                bbox = ann['bbox']
                x, y, w, h = bbox
                x_center_norm = (x + w / 2) / img_width
                y_center_norm = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height

                # Normalize keypoints
                keypoints = ann['keypoints']
                kpts_str = ""
                for i in range(0, len(keypoints), 3):
                    kpt_x = keypoints[i] / img_width
                    kpt_y = keypoints[i+1] / img_height
                    kpt_v = keypoints[i+2]
                    kpts_str += f" {kpt_x:.6f} {kpt_y:.6f} {kpt_v}"
                
                # Write to file
                # Class index is 0 since we only have one class ('dog')
                f_label.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}{kpts_str}\n")

    return image_paths

def create_yaml_config(train_image_paths, val_image_paths):
    """Creates the final YOLOv8 dataset YAML configuration file."""
    
    # Create train.txt and val.txt files with absolute image paths
    train_txt_path = DATA_ROOT / "train.txt"
    val_txt_path = DATA_ROOT / "val.txt"

    with open(train_txt_path, 'w') as f:
        for path in sorted(train_image_paths):
            f.write(f"{Path(path).as_posix()}\n")
    print(f"Created train.txt with {len(train_image_paths)} image paths.")

    with open(val_txt_path, 'w') as f:
        for path in sorted(val_image_paths):
            f.write(f"{Path(path).as_posix()}\n")
    print(f"Created val.txt with {len(val_image_paths)} image paths.")

    yaml_content = {
        'path': Path(DATA_ROOT.resolve()).as_posix(),
        'train': Path(train_txt_path.resolve()).as_posix(),
        'val': Path(val_txt_path.resolve()).as_posix(),
        'nc': 1,
        'names': ['dog'],
        'kpt_shape': [5, 3], # 5 keypoints, 3 dimensions (x, y, visibility)
    }

    with open(YOLO_YAML_PATH, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"Successfully created YAML config at: {YOLO_YAML_PATH}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Converting COCO JSON dataset to YOLOv8 Pose format")
    print("=" * 60)

    train_paths = convert_split('train')
    val_paths = convert_split('val')

    if not train_paths and not val_paths:
        print("Error: No data was processed. Check that your COCO JSON files exist.")
        return

    create_yaml_config(train_paths, val_paths)

    print("\nConversion complete!")
    print("You are now ready to train the YOLOv8 model.")
    print("=" * 60)

if __name__ == "__main__":
    main()
