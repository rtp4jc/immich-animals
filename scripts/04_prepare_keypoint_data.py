#!/usr/bin/env python3
"""
Prepare Keypoint Dataset

Creates COCO format keypoint dataset and converts to YOLO format.
Combines dataset creation and format conversion without visualization.
Use script 02 to inspect the created datasets.
"""

import argparse

from animal_id.keypoint.dataset_converter import (
    CocoKeypointDatasetConverter,
    create_default_config,
)
from animal_id.keypoint.yolo_converter import (
    CocoToYoloKeypointConverter,
)

# Add project root to Python path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare keypoint dataset")
    parser.add_argument(
        "--output-dir",
        default="data/keypoints/coco",
        help="Output directory for COCO dataset",
    )
    parser.add_argument(
        "--yaml-path",
        default="data/keypoints/dogs_keypoints_only.yaml",
        help="Output path for YOLO config file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing Keypoint Dataset")
    print("=" * 60)

    # Step 1: Create COCO dataset
    print("\n1. Creating COCO format keypoint dataset...")
    config = create_default_config()
    config["output_coco_dir"] = args.output_dir

    converter = CocoKeypointDatasetConverter(config)
    converter.convert()

    # Step 2: Convert to YOLO format
    print("\n2. Converting COCO to YOLO format...")
    yolo_converter = CocoToYoloKeypointConverter(
        coco_annotations_dir=args.output_dir,
        labels_output_dir="data/keypoints/labels",
        data_root="data",
        yaml_output_path=args.yaml_path,
    )
    yolo_converter.convert()

    print("\n" + "=" * 60)
    print("Keypoint dataset preparation complete!")
    print(f"COCO dataset: {args.output_dir}")
    print(f"YOLO config: {args.yaml_path}")
    print("Use script 02 to inspect the datasets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
