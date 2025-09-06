#!/usr/bin/env python3
"""
run_two_stage_inference.py

Runs the full two-stage pipeline for dog detection and keypoint estimation.
1.  Loads the best detector model and the best keypoint model.
2.  Runs the detector on sample images.
3.  For each detected dog, crops the image.
4.  Runs the keypoint estimator on the crop.
5.  If multiple keypoint instances are found, selects the one closest to the original detection.
6.  Visualizes both the bounding box and the final keypoints on the original image.
"""

import os
import glob
import shutil
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# --- Configuration ---
PADDING_FACTOR = 0.2  # Should match the padding used for training the keypoint model

def find_latest_model(run_name_prefix):
    """Find the latest model weights (best.pt) from a given run prefix."""
    model_dir = "models/phase1"
    list_of_dirs = glob.glob(os.path.join(model_dir, f"{run_name_prefix}*"))
    if not list_of_dirs:
        raise FileNotFoundError(f"No run directory found with prefix '{run_name_prefix}' in {model_dir}")
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    
    model_path = os.path.join(latest_dir, "weights", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No best.pt model found in {latest_dir}")
    return model_path

def main():
    print("=" * 60)
    print("Running Two-Stage Inference Pipeline")
    print("=" * 60)

    # --- Setup ---
    sample_images_dir = "outputs/phase1/detector_sample_images"
    output_dir = "outputs/phase1/final_inference_results"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if not os.path.exists(sample_images_dir) or not os.listdir(sample_images_dir):
        print(f"Error: Sample images directory is empty or does not exist.")
        print(f"Please populate '{sample_images_dir}' with images to process.")
        return

    try:
        print("Loading models...")
        detector_path = find_latest_model("detector_run")
        keypoint_path = find_latest_model("keypoint_run")
        
        detector_model = YOLO(detector_path)
        keypoint_model = YOLO(keypoint_path)
        print(f"  Detector: {detector_path}")
        print(f"  Keypoint Model: {keypoint_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- Inference ---
    image_files = glob.glob(os.path.join(sample_images_dir, "*.jpg"))
    print(f"\nFound {len(image_files)} images to process.")

    for img_path in image_files:
        print(f"Processing {Path(img_path).name}...")
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue

        detector_results = detector_model.predict(original_img, conf=0.25, verbose=False)
        num_detections = len(detector_results[0].boxes)

        if num_detections == 0:
            print("  -> No dogs detected.")
            cv2.imwrite(os.path.join(output_dir, Path(img_path).name), original_img)
            continue

        visualized_img = original_img.copy()

        for i, box in enumerate(detector_results[0].boxes):
            print(f"  -> Processing detection {i + 1} of {num_detections}...")
            x1_det, y1_det, x2_det, y2_det = [int(c) for c in box.xyxy[0]]
            w_det = x2_det - x1_det
            h_det = y2_det - y1_det

            cv2.rectangle(visualized_img, (x1_det, y1_det), (x2_det, y2_det), (255, 0, 0), 2)

            pad_w = int(w_det * PADDING_FACTOR)
            pad_h = int(h_det * PADDING_FACTOR)

            crop_x1 = max(0, x1_det - pad_w)
            crop_y1 = max(0, y1_det - pad_h)
            crop_x2 = min(original_img.shape[1], x2_det + pad_w)
            crop_y2 = min(original_img.shape[0], y2_det + pad_h)

            cropped_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped_img.size == 0:
                continue

            keypoint_results = keypoint_model.predict(cropped_img, verbose=False)
            if not keypoint_results or keypoint_results[0].keypoints is None or keypoint_results[0].keypoints.data.numel() == 0:
                continue

            # --- Disambiguation Logic ---
            best_instance_kpts = None
            if len(keypoint_results[0].keypoints.data) == 1:
                best_instance_kpts = keypoint_results[0].keypoints.data[0]
            else:
                print(f"    -> Keypoint model found {len(keypoint_results[0].keypoints.data)} instances. Selecting the best one.")
                detector_box_center_x = x1_det + w_det / 2
                detector_box_center_y = y1_det + h_det / 2
                min_distance = float('inf')

                for instance_kpts in keypoint_results[0].keypoints.data:
                    visible_kpts = [kpt for kpt in instance_kpts if kpt[2] > 0.5]
                    if not visible_kpts:
                        continue

                    avg_x_crop = sum(kpt[0] for kpt in visible_kpts) / len(visible_kpts)
                    avg_y_crop = sum(kpt[1] for kpt in visible_kpts) / len(visible_kpts)
                    avg_x_orig = avg_x_crop + crop_x1
                    avg_y_orig = avg_y_crop + crop_y1

                    distance = ((avg_x_orig - detector_box_center_x)**2 + (avg_y_orig - detector_box_center_y)**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_instance_kpts = instance_kpts
            
            if best_instance_kpts is None:
                continue

            # --- Drawing Logic ---
            for kpt in best_instance_kpts:
                kpt_x_crop, kpt_y_crop, kpt_conf = kpt
                if kpt_conf > 0.5:
                    kpt_x_orig = int(kpt_x_crop + crop_x1)
                    kpt_y_orig = int(kpt_y_crop + crop_y1)
                    cv2.circle(visualized_img, (kpt_x_orig, kpt_y_orig), 5, (0, 255, 0), -1)

        cv2.imwrite(os.path.join(output_dir, Path(img_path).name), visualized_img)
        print(f"  -> Saved visualization to {output_dir}")

    print("\nPipeline finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()