#!/usr/bin/env python3
"""
validate_and_infer.py - Validation and inference script for YOLOv8 pose model

Prompt 6: Run quick validation & save sample inference outputs

Loads the latest trained model, runs validation on the val split,
and performs inference on sample images with reporting.
"""

import os
import glob
import csv
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import shutil

def find_latest_model():
    """Find the latest model weights (best.pt) from phase1 runs"""
    model_dir = "models/phase1"
    pattern = os.path.join(model_dir, "*", "weights", "best.pt")

    model_paths = glob.glob(pattern)
    if not model_paths:
        raise FileNotFoundError("No best.pt model found in models/phase1/")

    model_paths.sort(key=os.path.getmtime, reverse=True)
    return model_paths[0]

def validate_model(model_path, data_yaml):
    """Run validation on the model and return metrics"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print("Running validation...")
    results = model.val(data=data_yaml, verbose=True)

    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'mAP50_keypoints': results.pose.map50,
        'mAP50-95_keypoints': results.pose.map,
        'precision_box': results.box.mp,
        'recall_box': results.box.mr,
        'precision_pose': results.pose.mp,
        'recall_pose': results.pose.mr
    }

    return model, metrics

def run_inference(model, sample_dir, output_dir):
    """Run inference on sample images and return results data"""
    print(f"\nRunning inference on: {sample_dir}")
    
    # Clear and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    results = model.predict(
        source=sample_dir,
        save=True,
        project=output_dir, # Save to a parent folder
        name='predictions', # Sub-folder name
        exist_ok=True,
        conf=0.25,
        verbose=False # We will print our own summary
    )

    inference_data = []
    for result in results:
        img_path = Path(result.path)
        boxes = result.boxes
        keypoints = result.keypoints

        if not boxes or len(boxes) == 0:
            print(f"{img_path}: No detections")
            inference_data.append({'image_path': img_path.name, 'detected_dog': False, 'num_boxes': 0, 'num_keypoints_detected': 0, 'avg_score': 0.0})
            continue

        print(f"{img_path}: {len(boxes)} detections")

        # Get keypoint data once
        kp_data = keypoints.data if hasattr(keypoints, 'data') else torch.empty(0)
        num_keypoints_detected_total = 0

        for i in range(len(boxes)):
            num_kpts_this_detection = 0
            if i < len(kp_data):
                kpts = kp_data[i]
                visible_kpts = kpts[kpts[:, 2] > 0] # Filter for visible keypoints
                num_kpts_this_detection = len(visible_kpts)
                num_keypoints_detected_total += num_kpts_this_detection
            
        # For CSV, aggregate info for the image
        inference_data.append({
            'image_path': img_path.name,
            'detected_dog': True,
            'num_boxes': len(boxes),
            'num_keypoints_detected': num_keypoints_detected_total,
            'avg_score': boxes.conf.mean().item()
        })

    return inference_data

def generate_csv_report(inference_data, output_path):
    """Generate CSV report from inference data"""
    print(f"\nGenerating CSV report: {output_path}")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'detected_dog', 'num_boxes', 'num_keypoints_detected', 'avg_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inference_data)

def main():
    """Main execution function"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sample_images_dir = "outputs/phase1/sample_images"
    inference_dir = "outputs/phase1/inference"
    csv_report_path = "outputs/phase1/inference_report.csv"
    data_yaml = "data/dogs_keypoints.yaml"

    if not os.path.exists(sample_images_dir):
        raise FileNotFoundError(f"Sample images directory not found: {sample_images_dir}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    model_path = find_latest_model()
    print(f"Using model: {model_path}")

    model, metrics = validate_model(model_path, data_yaml)

    print("\n=== VALIDATION METRICS ===")
    for key, value in metrics.items():
        print(f"  {key:<20}: {value:.4f}")

    inference_data = run_inference(model, sample_images_dir, inference_dir)
    generate_csv_report(inference_data, csv_report_path)

    print("\n=== INFERENCE SUMMARY ===")
    total_images = len(inference_data)
    images_with_dogs = sum(1 for item in inference_data if item['detected_dog'])
    print(f"Total sample images: {total_images}")
    print(f"Images with detected dogs: {images_with_dogs} ({images_with_dogs/total_images*100:.1f}%)")
    print(f"Visualized outputs saved to: {os.path.join(inference_dir, 'predictions')}")
    print(f"CSV report saved to: {csv_report_path}")

if __name__ == "__main__":
    main()
