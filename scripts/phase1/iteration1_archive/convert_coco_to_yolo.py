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
import shutil
import glob

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
COCO_ANNOTATIONS_DIR = DATA_ROOT / "coco_keypoints"

# The directory where the YOLO `.txt` label files will be created.
# The structure will mirror the image directories.
LABELS_OUTPUT_DIR = DATA_ROOT

YOLO_YAML_PATH = DATA_ROOT / "dogs_keypoints.yaml"

def convert_split(split_name, include_negatives=False):
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
    # Process all images, including those without annotations (negatives)
    for img_id, image_info in tqdm(images_map.items(), desc=f"Generating {split_name} labels"):
        img_height = image_info['height']
        img_width = image_info['width']
        relative_img_path = image_info['file_name']
        
        # Determine label path based on whether it's a COCO path (no 'images' dir) or not.
        relative_img_path_str = str(Path(relative_img_path))
        if 'images' in relative_img_path_str:
            # Standard case (Stanford, Oxford): replace 'images' with 'labels'.
            # e.g., stanford_dogs/images/subdir/file.jpg -> stanford_dogs/labels/subdir/file.txt
            relative_label_path_str = relative_img_path_str.replace('images', 'labels', 1)
            label_path = (LABELS_OUTPUT_DIR / Path(relative_label_path_str)).with_suffix('.txt')
        else:
            # COCO case: insert 'labels' directory to mirror the image directory structure.
            # e.g., coco/train2017/file.jpg -> coco/labels/train2017/file.txt
            p = Path(relative_img_path)
            label_path = LABELS_OUTPUT_DIR / p.parent.parent / 'labels' / p.parent.name / p.with_suffix('.txt').name

        image_paths.append(str(DATA_ROOT / relative_img_path))
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, 'w') as f_label:
            if img_id in annotations_by_image:
                annotations = annotations_by_image[img_id]
                for ann in annotations:
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    x_center_norm = (x + w / 2) / img_width
                    y_center_norm = (y + h / 2) / img_height
                    width_norm = w / img_width
                    height_norm = h / img_height

                    keypoints = ann.get('keypoints', [])
                    kpts_str = ""
                    num_kpts = 4 
                    if keypoints and ann.get('num_keypoints', 0) > 0:
                        for i in range(0, len(keypoints), 3):
                            kpt_x = keypoints[i] / img_width
                            kpt_y = keypoints[i+1] / img_height
                            kpt_v = keypoints[i+2]
                            if kpt_v > 0:
                                kpt_v = 2
                            kpts_str += f" {kpt_x:.6f} {kpt_y:.6f} {kpt_v}"
                    else:
                        kpts_str = " 0" * (num_kpts * 3)
                    
                    f_label.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}{kpts_str}\n")

    return image_paths

def create_yaml_config(train_image_paths, val_image_paths):
    """Creates the final YOLOv8 dataset YAML configuration file."""
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
        'kpt_shape': [4, 3],
        'flip_idx': [2, 3, 0, 1],
    }

    with open(YOLO_YAML_PATH, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"Successfully created YAML config at: {YOLO_YAML_PATH}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Clearing stale YOLO annotations...")
    
    # Find and delete all .txt files within any 'labels' subdirectories
    label_files = glob.glob(str(DATA_ROOT / "**/labels/**/*.txt"), recursive=True)
    if label_files:
        print(f"Deleting {len(label_files)} stale .txt label files...")
        for label_file in label_files:
            os.remove(label_file)

    # Delete existing train/val list files and dataset yaml
    for list_file in ["train.txt", "val.txt", "dogs_keypoints.yaml"]:
        file_path = DATA_ROOT / list_file
        if file_path.exists():
            print(f"Deleting stale list file: {file_path}")
            os.remove(file_path)
    
    print("\nConverting COCO JSON dataset to YOLOv8 Pose format")
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