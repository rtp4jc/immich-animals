#!/usr/bin/env python3
"""
Creates a cropped keypoint dataset in COCO format.

This script reads the StanfordExtra dataset, which contains keypoint annotations.
For each annotation, it crops the dog from the original image, applies padding,
and transforms the keypoint coordinates to be relative to the new cropped image.

It then saves these new cropped images and a new COCO-formatted annotation file.
"""

import os
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

# --- Configuration ---
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"

SOURCE_JSON = DATA_ROOT / "stanford_dogs" / "stanford_extra_keypoints.json"

# Output directories following the consistent structure
CROPPED_IMAGE_DIR = DATA_ROOT / "keypoints" / "images"
OUTPUT_COCO_DIR = DATA_ROOT / "keypoints" / "coco"

# --- Parameters ---
PADDING_FACTOR = 0.2  # 20% padding around the bounding box
TRAIN_VAL_SPLIT = 0.9 # 90% for training, 10% for validation


def map_and_transform_keypoints(joints, x_offset, y_offset):
    """
    Selects our 4 target keypoints from the 24 available, and transforms
    their coordinates from original image space to cropped space.
    """
    keypoint_map = {
        'nose': 16,
        'chin': 17,
        'left_ear_base': 14,
        'right_ear_base': 15,
    }
    
    output_keypoints = []
    # Iterate in our desired keypoint order
    for name in ['nose', 'chin', 'left_ear_base', 'right_ear_base']:
        idx = keypoint_map[name]
        
        # Check if the keypoint exists and is visible in the source data
        if len(joints) > idx and joints[idx][2] > 0:
            kpt = joints[idx]
            x, y, v = kpt[0], kpt[1], kpt[2]
            # Transform coordinates to be relative to the crop
            new_x = x - x_offset
            new_y = y - y_offset
            output_keypoints.extend([new_x, new_y, v])
        else:
            # If keypoint is not present or not visible, add zeros
            output_keypoints.extend([0, 0, 0])
            
    return output_keypoints

def main():
    print("=" * 60)
    print("Creating Cropped Keypoint Dataset in COCO Format")
    print("=" * 60)

    if not SOURCE_JSON.exists():
        print(f"Error: Source keypoint JSON not found: {SOURCE_JSON}")
        return

    # Setup output directories
    for dir_path in [CROPPED_IMAGE_DIR, OUTPUT_COCO_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)

    with open(SOURCE_JSON, 'r') as f:
        source_data = json.load(f)

    # Initialize COCO structure
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1, 'name': 'dog',
            'keypoints': ['nose', 'chin', 'left_ear_base', 'right_ear_base'],
            'skeleton': []
        }]
    }

    image_id_counter = 0
    annotation_id_counter = 0

    for entry in tqdm(source_data, desc="Processing and cropping images"):
        if entry.get('is_multiple_dogs', False):
            continue

        original_img_path = DATA_ROOT / "stanford_dogs" / "images" / entry['img_path']
        if not original_img_path.exists():
            continue

        img = cv2.imread(str(original_img_path))
        if img is None:
            continue

        # Calculate crop coordinates with padding
        x, y, w, h = [int(v) for v in entry['img_bbox']]
        pad_w = int(w * PADDING_FACTOR)
        pad_h = int(h * PADDING_FACTOR)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img.shape[1], x + pad_w)
        y2 = min(img.shape[0], y + pad_h)

        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        new_filename = f"{Path(entry['img_path']).stem}_{annotation_id_counter}.jpg"
        cropped_img_path = CROPPED_IMAGE_DIR / new_filename
        cv2.imwrite(str(cropped_img_path), cropped_img)

        new_height, new_width, _ = cropped_img.shape
        image_entry = {
            'id': image_id_counter,
            'width': new_width,
            'height': new_height,
            'file_name': str(cropped_img_path.relative_to(DATA_ROOT)),
            'source_dataset': 'stanford_extra_cropped'
        }
        coco_output['images'].append(image_entry)

        # FIX: Use the new mapping function to get the correct 4 keypoints
        transformed_kpts = map_and_transform_keypoints(entry['joints'], x1, y1)
        
        new_bbox_x = x - x1
        new_bbox_y = y - y1
        new_bbox = [new_bbox_x, new_bbox_y, w, h]

        annotation_entry = {
            'id': annotation_id_counter,
            'image_id': image_id_counter,
            'category_id': 1,
            'bbox': new_bbox,
            'keypoints': transformed_kpts,
            'num_keypoints': sum(1 for i in range(2, len(transformed_kpts), 3) if transformed_kpts[i] > 0),
            'area': w * h,
            'iscrowd': 0
        }
        coco_output['annotations'].append(annotation_entry)

        image_id_counter += 1
        annotation_id_counter += 1

    # Split data into train and validation sets
    random.seed(42)
    random.shuffle(coco_output['images'])
    split_idx = int(len(coco_output['images']) * TRAIN_VAL_SPLIT)

    train_images = coco_output['images'][:split_idx]
    val_images = coco_output['images'][split_idx:]

    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    train_annotations = [ann for ann in coco_output['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_output['annotations'] if ann['image_id'] in val_image_ids]

    train_coco = {'images': train_images, 'annotations': train_annotations, 'categories': coco_output['categories']}
    val_coco = {'images': val_images, 'annotations': val_annotations, 'categories': coco_output['categories']}

    with open(OUTPUT_COCO_DIR / 'annotations_train.json', 'w') as f:
        json.dump(train_coco, f, indent=2)
    with open(OUTPUT_COCO_DIR / 'annotations_val.json', 'w') as f:
        json.dump(val_coco, f, indent=2)

    print(f"\nSuccessfully created {len(train_images)} training samples and {len(val_images)} validation samples.")
    print(f"Cropped images saved to: {CROPPED_IMAGE_DIR}")
    print(f"COCO annotations saved to: {OUTPUT_COCO_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()