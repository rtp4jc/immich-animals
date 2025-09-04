"""
visualize_dataset.py

A script to visualize and validate the dog keypoints dataset.

This tool helps debug the data pipeline by:
1.  Randomly sampling images from the dataset.
2.  Drawing all bounding boxes and keypoints from both the source COCO JSON
    annotations and the final YOLO .txt labels for a given image.
3.  Printing a side-by-side comparison of the raw annotation data.

Usage:
    Random sampling:
    python scripts/phase1/visualize_dataset.py [--num_samples 5] [--source coco]

    Specific image:
    python scripts/phase1/visualize_dataset.py --image_path path/to/your/image.jpg
"""

import os
import json
import random
import cv2
import numpy as np
import argparse
from pathlib import Path

# --- Configuration ---
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
ANNOTATIONS_PATH = DATA_ROOT / "coco_keypoints" / "annotations_val.json"

# --- Drawing Constants ---
COCO_COLOR = (255, 128, 0)  # Blue
YOLO_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 0, 255)  # Red

def denormalize_yolo_bbox(x_center, y_center, w, h, img_width, img_height):
    """Converts normalized YOLO bbox coordinates to absolute pixel values."""
    w_abs = w * img_width
    h_abs = h * img_height
    x1 = int((x_center * img_width) - (w_abs / 2))
    y1 = int((y_center * img_height) - (h_abs / 2))
    x2 = int(x1 + w_abs)
    y2 = int(y1 + h_abs)
    return x1, y1, x2, y2

def denormalize_yolo_kpts(kpts, img_width, img_height):
    """Converts normalized YOLO keypoints to absolute pixel values."""
    denormalized = []
    for i in range(0, len(kpts), 3):
        x = int(kpts[i] * img_width)
        y = int(kpts[i+1] * img_height)
        v = int(kpts[i+2])
        denormalized.extend([x, y, v])
    return denormalized

def draw_single_annotation(image, coco_ann, yolo_data, img_width, img_height):
    """Draws a single annotation from both formats onto the image."""
    if coco_ann:
        x, y, w, h = [int(v) for v in coco_ann['bbox']]
        cv2.rectangle(image, (x, y), (x + w, y + h), COCO_COLOR, 2)
        cv2.putText(image, "COCO", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COCO_COLOR, 2)
        kpts = coco_ann.get('keypoints', [])
        for i in range(0, len(kpts), 3):
            kx, ky, v = kpts[i], kpts[i+1], kpts[i+2]
            if v > 0:
                cv2.circle(image, (int(kx), int(ky)), 5, COCO_COLOR, -1)

    if yolo_data:
        x1, y1, x2, y2 = denormalize_yolo_bbox(*yolo_data[1:5], img_width, img_height)
        cv2.rectangle(image, (x1, y1), (x2, y2), YOLO_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, "YOLO", (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YOLO_COLOR, 2)
        kpts = denormalize_yolo_kpts(yolo_data[5:], img_width, img_height)
        for i in range(0, len(kpts), 3):
            kx, ky, v = kpts[i], kpts[i+1], kpts[i+2]
            if v > 0:
                cv2.circle(image, (kx, ky), 5, YOLO_COLOR, -1)
                cv2.circle(image, (kx, ky), 7, (255,255,255), 1)
    return image

def print_comparison(coco_ann, yolo_data, ann_index):
    """Prints a formatted comparison for a single annotation."""
    print(f"\n--- Annotation #{ann_index + 1} ---")
    if coco_ann:
        print("  COCO (source):")
        print(f"    Bbox (x,y,w,h): {[f'{v:.2f}' for v in coco_ann['bbox']]}")
        kpts = coco_ann.get('keypoints', [])
        kpt_str = ", ".join([f"({kpts[i]:.1f},{kpts[i+1]:.1f},{kpts[i+2]})" for i in range(0, len(kpts), 3)])
        print(f"    Keypoints: {kpt_str}")
    if yolo_data:
        print("  YOLO (.txt):")
        headers = ['cls', 'x_c', 'y_c', 'w', 'h', 'n_x', 'n_y', 'n_v', 'c_x', 'c_y', 'c_v', 'le_x', 'le_y', 'le_v', 're_x', 're_y', 're_v']
        values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in yolo_data]
        print(f"    {", ".join(headers)}")
        print(f"    {", ".join(values)}")
    if not coco_ann and not yolo_data:
        print("  No annotations for this image (Negative Sample).")

def get_yolo_label_path(relative_img_path):
    """Constructs the expected path to the YOLO label file."""
    relative_img_path_str = str(Path(relative_img_path))
    if 'images' in relative_img_path_str:
        relative_label_path_str = relative_img_path_str.replace('images', 'labels', 1)
    else:
        p = Path(relative_img_path)
        relative_label_path_str = str(p.parent.parent / 'labels' / p.parent.name / p.name)
    return (DATA_ROOT / Path(relative_label_path_str)).with_suffix('.txt')

def main(args):
    """Main execution function."""
    if not ANNOTATIONS_PATH.exists():
        print(f"Error: Annotation file not found at {ANNOTATIONS_PATH}")
        return

    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data.get('annotations', [])
    image_map = {img['id']: img for img in images}
    ann_map = {}
    for ann in annotations:
        if ann['image_id'] not in ann_map:
            ann_map[ann['image_id']] = []
        ann_map[ann['image_id']].append(ann)

    if args.image_path:
        user_path = Path(args.image_path)
        # If the path doesn't exist as provided, assume it's relative to the DATA_ROOT
        if not user_path.exists():
            user_path = DATA_ROOT / user_path

        try:
            relative_path = user_path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
        except ValueError:
            print(f"Error: Provided image path {user_path} could not be resolved within the data directory {DATA_ROOT}")
            return

        target_id = None
        for img_id, img_info in image_map.items():
            if img_info['file_name'] == relative_path:
                target_id = img_id
                break
        
        if target_id is None:
            print(f"Error: Image path {args.image_path} not found in the annotation file.")
            return
        sample_ids = [target_id]
    else:
        if args.source:
            image_ids = [img['id'] for img in images if img.get('source_dataset', '').startswith(args.source)]
            if not image_ids:
                print(f"No images found for source: {args.source}")
                return
        else:
            image_ids = list(image_map.keys())
        sample_ids = random.sample(image_ids, min(args.num_samples, len(image_ids)))

    for img_id in sample_ids:
        img_info = image_map[img_id]
        coco_anns = ann_map.get(img_id, [])
        img_path = DATA_ROOT / img_info['file_name']
        if not img_path.exists():
            print(f"Warning: Image file not found, skipping: {img_path}")
            continue
        
        image = cv2.imread(str(img_path))
        vis_image = image.copy()
        img_height, img_width, _ = image.shape

        yolo_lines = []
        yolo_path = get_yolo_label_path(img_info['file_name'])
        if yolo_path.exists():
            with open(yolo_path, 'r') as f_yolo:
                yolo_lines = [line.strip() for line in f_yolo.readlines() if line.strip()]

        print("\n" + "="*80)
        print(f"IMAGE: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")
        print(f"SOURCE: {img_info.get('source_dataset', 'N/A')}")
        print(f"Found {len(coco_anns)} COCO annotations and {len(yolo_lines)} YOLO annotations.")
        print("="*80)

        if coco_anns:
            for i, coco_ann in enumerate(coco_anns):
                yolo_data = None
                if i < len(yolo_lines):
                    yolo_data = [float(x) for x in yolo_lines[i].split()]
                vis_image = draw_single_annotation(vis_image, coco_ann, yolo_data, img_width, img_height)
                print_comparison(coco_ann, yolo_data, i)
        else:
            print_comparison(None, None, 0)

        cv2.imshow(f"Image: {img_info['file_name']}", vis_image)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and validate the dog keypoints dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    parser.add_argument("--source", type=str, default=None, choices=['coco', 'stanford_dogs', 'oxford_pets'], help="Filter by dataset source.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a specific image file to visualize.")
    args = parser.parse_args()
    main(args)
