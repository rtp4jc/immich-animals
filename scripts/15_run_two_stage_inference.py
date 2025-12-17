#!/usr/bin/env python3
"""
Two-Stage Inference Pipeline

Runs the full two-stage pipeline for dog detection and keypoint estimation.
1. Loads the best detector model and the best keypoint model
2. Runs the detector on sample images
3. For each detected dog, crops the image
4. Runs the keypoint estimator on the crop
5. If multiple keypoint instances are found, selects the one closest to the original detection
6. Visualizes both the bounding box and the final keypoints on the original image
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from animal_id.common.visualization import setup_output_dir

# Add project root to Python path


# Configuration
PADDING_FACTOR = 0.2  # Should match the padding used for training the keypoint model


def find_latest_model(run_name_prefix):
    """Find the latest model weights (best.pt) from a given run prefix."""
    model_dir = "models/phase1"
    list_of_dirs = glob.glob(os.path.join(model_dir, f"{run_name_prefix}*"))
    if not list_of_dirs:
        raise FileNotFoundError(
            f"No run directory found with prefix '{run_name_prefix}' in {model_dir}"
        )

    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    best_pt_path = os.path.join(latest_dir, "weights", "best.pt")

    if not os.path.exists(best_pt_path):
        raise FileNotFoundError(f"best.pt not found in {latest_dir}/weights/")

    return best_pt_path


def crop_with_padding(image, bbox, padding_factor=0.2):
    """Crop image around bounding box with padding."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Calculate padding
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    pad_w = int(bbox_w * padding_factor)
    pad_h = int(bbox_h * padding_factor)

    # Apply padding with bounds checking
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w, x2 + pad_w)
    crop_y2 = min(h, y2 + pad_h)

    return image[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1)


def find_closest_keypoint_instance(keypoint_results, original_bbox):
    """Find keypoint instance closest to original detection center."""
    if not keypoint_results or len(keypoint_results[0].boxes) == 0:
        return None

    orig_center_x = (original_bbox[0] + original_bbox[2]) / 2
    orig_center_y = (original_bbox[1] + original_bbox[3]) / 2

    best_idx = 0
    min_distance = float("inf")

    for i, box in enumerate(keypoint_results[0].boxes.xyxy):
        kp_center_x = (box[0] + box[2]) / 2
        kp_center_y = (box[1] + box[3]) / 2
        distance = (
            (kp_center_x - orig_center_x) ** 2 + (kp_center_y - orig_center_y) ** 2
        ) ** 0.5

        if distance < min_distance:
            min_distance = distance
            best_idx = i

    return best_idx


def visualize_results(image, detections, keypoint_results, crop_offsets, output_path):
    """Visualize detection and keypoint results."""
    vis_image = image.copy()

    for i, (det_box, kp_result, crop_offset) in enumerate(
        zip(detections, keypoint_results, crop_offsets)
    ):
        # Draw detection bounding box
        x1, y1, x2, y2 = map(int, det_box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_image,
            f"Dog {i + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # Draw keypoints if available
        if kp_result and len(kp_result[0].keypoints) > 0:
            keypoints = kp_result[0].keypoints.xy[0]  # First (closest) instance
            crop_x, crop_y = crop_offset

            # Keypoint names for visualization
            kp_names = ["nose", "chin", "left_ear", "right_ear"]
            colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

            for j, (kp, name, color) in enumerate(zip(keypoints, kp_names, colors)):
                if len(kp) >= 2:
                    # Convert crop coordinates back to original image coordinates
                    orig_x = int(kp[0] + crop_x)
                    orig_y = int(kp[1] + crop_y)
                    cv2.circle(vis_image, (orig_x, orig_y), 3, color, -1)
                    cv2.putText(
                        vis_image,
                        name,
                        (orig_x + 5, orig_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1,
                    )

    cv2.imwrite(str(output_path), vis_image)
    return vis_image


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run two-stage dog detection and keypoint inference"
    )
    parser.add_argument(
        "--detector-prefix",
        default="detector_yolo11n",
        help="Detector model run prefix",
    )
    parser.add_argument(
        "--keypoint-prefix",
        default="keypoint_yolo11n",
        help="Keypoint model run prefix",
    )
    parser.add_argument(
        "--input-dir",
        default="data/sample_images",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/15_two_stage_inference",
        help="Output directory for results",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Two-Stage Inference Pipeline")
    print("=" * 60)

    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)

    # Load models
    print("Loading models...")
    try:
        detector_path = find_latest_model(args.detector_prefix)
        keypoint_path = find_latest_model(args.keypoint_prefix)
        print(f"Detector: {detector_path}")
        print(f"Keypoint: {keypoint_path}")

        detector = YOLO(detector_path)
        keypoint_model = YOLO(keypoint_path)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Process images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        print(f"Processing: {img_file.name}")

        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Failed to load {img_file}")
            continue

        # Run detection
        det_results = detector(image, conf=args.confidence)

        if len(det_results[0].boxes) == 0:
            print(f"No dogs detected in {img_file.name}")
            continue

        detections = []
        keypoint_results = []
        crop_offsets = []

        # Process each detection
        for box in det_results[0].boxes.xyxy:
            bbox = box.cpu().numpy()
            detections.append(bbox)

            # Crop image around detection
            crop, crop_offset = crop_with_padding(image, bbox, PADDING_FACTOR)
            crop_offsets.append(crop_offset)

            # Run keypoint estimation on crop
            kp_results = keypoint_model(crop)

            # Find closest keypoint instance if multiple exist
            if len(kp_results[0].boxes) > 1:
                closest_idx = find_closest_keypoint_instance(kp_results, bbox)
                if closest_idx is not None:
                    # Keep only the closest instance
                    kp_results[0].boxes = kp_results[0].boxes[
                        closest_idx : closest_idx + 1
                    ]
                    kp_results[0].keypoints = kp_results[0].keypoints[
                        closest_idx : closest_idx + 1
                    ]

            keypoint_results.append(kp_results)

        # Visualize and save results
        output_path = output_dir / f"result_{img_file.name}"
        visualize_results(
            image, detections, keypoint_results, crop_offsets, output_path
        )
        print(f"Saved: {output_path}")

    print(f"\nProcessing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
