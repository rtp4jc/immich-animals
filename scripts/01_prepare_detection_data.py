#!/usr/bin/env python3
"""
Prepare Detection Dataset

Creates COCO format detection dataset and converts to YOLO format.
Combines dataset creation and format conversion without visualization.
Use script 02 to inspect the created datasets.
"""

import argparse
from pathlib import Path

from dog_id.detection.dataset_converter import CocoDetectorDatasetConverter, create_default_config
from dog_id.detection.yolo_converter import CocoToYoloDetectionConverter, create_default_converter


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare detection dataset")
    parser.add_argument("--output-dir", default="data/detector/coco", 
                       help="Output directory for COCO dataset")
    parser.add_argument("--yaml-path", default="data/detector/dogs_detection.yaml",
                       help="Output path for YOLO config file")
    args = parser.parse_args()

    print("=" * 60)
    print("Preparing Detection Dataset")
    print("=" * 60)

    # Step 1: Create COCO dataset
    print("\n1. Creating COCO format detection dataset...")
    config = create_default_config()
    config['output_dir'] = args.output_dir
    
    converter = CocoDetectorDatasetConverter(config)
    converter.convert()

    # Step 2: Convert to YOLO format
    print("\n2. Converting COCO to YOLO format...")
    yolo_converter = CocoToYoloDetectionConverter(
        coco_annotations_dir=args.output_dir,
        labels_output_dir="data",
        data_root="data", 
        yaml_output_path=args.yaml_path
    )
    yolo_converter.convert()

    print("\n" + "=" * 60)
    print("Detection dataset preparation complete!")
    print(f"COCO dataset: {args.output_dir}")
    print(f"YOLO config: {args.yaml_path}")
    print("Use script 02 to inspect the datasets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
