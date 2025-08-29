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

def find_latest_model():
    """Find the latest model weights (best.pt) from phase1 runs"""
    model_dir = "models/phase1"
    pattern = os.path.join(model_dir, "*", "weights", "best.pt")

    model_paths = glob.glob(pattern)
    if not model_paths:
        raise FileNotFoundError("No best.pt model found in models/phase1/")

    # Sort by run number (pose_run14 > pose_run13 > etc.)
    model_paths.sort(key=lambda x: int(x.split("\\pose_run")[1].split("\\")[0]), reverse=True)
    return model_paths[0]

def validate_model(model_path, data_yaml):
    """Run validation on the model and return metrics"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print("Running validation...")
    results = model.val(data=data_yaml, verbose=True)

    # Extract key metrics from results
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
    print(f"Running inference on: {sample_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run prediction with saving enabled
    results = model.predict(
        source=sample_dir,
        save=True,
        save_dir=output_dir,
        conf=0.25,  # confidence threshold
        verbose=True
    )

    # Collect results data for CSV
    inference_data = []

    for result in results:
        img_path = result.path
        img_name = os.path.basename(img_path)

        # Get boxes and keypoints
        boxes = result.boxes
        keypoints = result.keypoints

        detected_dog = len(boxes) > 0 if boxes is not None else False
        num_boxes = len(boxes) if boxes is not None else 0

        # Count keypoints detected (visibility > 0)
        if keypoints is not None and hasattr(keypoints, 'data'):
            kp_data = keypoints.data
            if kp_data.numel() > 0:  # Check if tensor has elements
                # Keypoints shape is [num_predictions, num_keypoints, 3] where 3=(x,y,visibility)
                num_keypoints_detected = torch.sum(kp_data[..., 2] > 0).item()
            else:
                num_keypoints_detected = 0
        else:
            num_keypoints_detected = 0

        # Get average confidence score for boxes
        if boxes is not None and len(boxes) > 0:
            avg_score = boxes.conf.mean().item() if hasattr(boxes, 'conf') else 0.0
        else:
            avg_score = 0.0

        inference_data.append({
            'image_path': img_name,
            'detected_dog': detected_dog,
            'num_boxes': num_boxes,
            'num_keypoints_detected': num_keypoints_detected,
            'avg_score': avg_score
        })

    return inference_data

def generate_csv_report(inference_data, output_path):
    """Generate CSV report from inference data"""
    print(f"Generating CSV report: {output_path}")

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'detected_dog', 'num_boxes', 'num_keypoints_detected', 'avg_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inference_data)

def main():
    """Main execution function"""
    # Verify CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Configuration
    sample_images_dir = "outputs/phase1/sample_images"
    inference_dir = "outputs/phase1/inference"
    csv_report_path = "outputs/phase1/inference_report.csv"
    data_yaml = "data/dogs_keypoints.yaml"

    # Verify prerequisites
    if not os.path.exists(sample_images_dir):
        raise FileNotFoundError(f"Sample images directory not found: {sample_images_dir}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    # Find and load the latest model
    model_path = find_latest_model()
    print(f"Using model: {model_path}")

    # Run validation
    model, metrics = validate_model(model_path, data_yaml)

    print("\n=== VALIDATION METRICS ===")
    for key, value in metrics.items():
        print(".4f")

    # Run inference on sample images
    inference_data = run_inference(model, sample_images_dir, inference_dir)

    # Generate CSV report
    generate_csv_report(inference_data, csv_report_path)

    # Summary statistics
    total_images = len(inference_data)
    images_with_dogs = sum(1 for item in inference_data if item['detected_dog'])
    avg_keypoints = sum(item['num_keypoints_detected'] for item in inference_data) / total_images

    print("=== INFERENCE SUMMARY ===")
    print(f"Total sample images: {total_images}")
    print(f"Images with detected dogs: {images_with_dogs} ({images_with_dogs/total_images*100:.1f}%)")
    print(".1f")
    print(f"Visualized outputs saved to: {inference_dir}")
    print(f"CSV report saved to: {csv_report_path}")

    # Find potential issues
    no_detection = [item for item in inference_data if not item['detected_dog']]
    low_keypoints = [item for item in inference_data if item['detected_dog'] and item['num_keypoints_detected'] < 3]

    if no_detection:
        print("Top images with no detection:")
        for item in no_detection[:5]:  # Show first 5
            print(f"  {item['image_path']}")

    if low_keypoints:
        print("Top images with low keypoints (< 3):")
        for item in low_keypoints[:5]:  # Show first 5
            print(f"  {item['image_path']}: {item['num_keypoints_detected']} keypoints")

    return metrics, inference_data

if __name__ == "__main__":
    # This will be run via command line
    main()