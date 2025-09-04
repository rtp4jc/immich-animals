#!/usr/bin/env python3
"""
validate_detector.py - Validation and inference script for the YOLOv8 detector model.

Loads the latest trained detector model, runs validation on the val split,
and performs inference on sample images, saving the visualized results.
"""

import os
import glob
import csv
import torch
import argparse
from ultralytics import YOLO
from pathlib import Path
import shutil

def find_latest_detector_model():
    """Find the latest model weights (best.pt) from the detector_run directory."""
    # The project name is 'detector_run', so the path will contain it.
    model_dir = "models/phase1"
    # We find the latest directory inside models/phase1 that starts with detector_run
    list_of_dirs = glob.glob(os.path.join(model_dir, "detector_run*"))
    if not list_of_dirs:
        raise FileNotFoundError(f"No detector run found in {model_dir}")
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    
    model_path = os.path.join(latest_dir, "weights", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No best.pt model found in {latest_dir}")

    return model_path

def validate_model(model_path, data_yaml):
    """Run validation on the model and return metrics."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print("Running validation on the detector model...")
    results = model.val(data=data_yaml, verbose=True)

    # Metrics for an object detector (no pose)
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
    }

    return model, metrics

def run_inference(model, sample_dir, output_dir):
    """Run inference on sample images and return results data."""
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
        conf=0.25, # Confidence threshold for detection
        verbose=False
    )

    inference_data = []
    for result in results:
        img_path = Path(output_dir) / 'predictions' / Path(result.path).name
        boxes = result.boxes

        if not boxes or len(boxes) == 0:
            print(f"{img_path}: No detections")
            inference_data.append({'image_path': img_path.name, 'detected_dog': False, 'num_boxes': 0, 'avg_score': 0.0})
            continue

        print(f"{img_path}: {len(boxes)} detections")

        inference_data.append({
            'image_path': img_path.name,
            'detected_dog': True,
            'num_boxes': len(boxes),
            'avg_score': boxes.conf.mean().item()
        })

    return inference_data

def generate_csv_report(inference_data, output_path):
    """Generate CSV report from inference data."""
    print(f"\nGenerating CSV report: {output_path}")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'detected_dog', 'num_boxes', 'avg_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inference_data)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', default=False,
                      help='Run validation on the model')
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Define paths for the detector validation task
    sample_images_dir = "outputs/phase1/detector_sample_images"
    inference_dir = "outputs/phase1/detector_inference"
    csv_report_path = os.path.join(inference_dir, "detector_inference_report.csv")
    data_yaml = "data/detector/dogs_detection.yaml"

    # Ensure sample directory exists
    if not os.path.exists(sample_images_dir):
        os.makedirs(sample_images_dir)
        print(f"Created sample directory: {sample_images_dir}")
        print("Please add some unseen images to this directory to test the detector.")
        return
    
    if not os.listdir(sample_images_dir):
        print(f"The sample directory is empty: {sample_images_dir}")
        print("Please add some unseen images to this directory to test the detector.")
        return

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    try:
        model_path = find_latest_detector_model()
        print(f"Using model: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the detector model has been trained and a 'best.pt' file exists.")
        return

    if args.validate:
        model, metrics = validate_model(model_path, data_yaml)
        print("=== VALIDATION METRICS ===")
        for key, value in metrics.items():
            print(f"  {key:<10}: {value:.4f}")
    else:
        model = YOLO(model_path)

    inference_data = run_inference(model, sample_images_dir, inference_dir)
    generate_csv_report(inference_data, csv_report_path)

    print("\n=== INFERENCE SUMMARY ===")
    total_images = len(inference_data)
    images_with_dogs = sum(1 for item in inference_data if item['detected_dog'])
    print(f"Total sample images: {total_images}")
    if total_images > 0:
        print(f"Images with detected dogs: {images_with_dogs} ({images_with_dogs/total_images*100:.1f}%)")
    print(f"Visualized outputs saved to: {os.path.join(inference_dir, 'predictions')}")
    print(f"CSV report saved to: {csv_report_path}")

if __name__ == "__main__":
    main()
