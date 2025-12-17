#!/usr/bin/env python3
"""
Inspect Datasets

Generic dataset inspection tool that can handle COCO and YOLO formats.
Provides dataset statistics and sample visualizations.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from animal_id.common.constants import DATA_DIR
from animal_id.common.visualization import (
    print_dataset_statistics,
    setup_output_dir,
    visualize_coco_annotations,
    visualize_yolo_annotations,
)

# Add project root to Python path


def analyze_data_distributions():
    """Analyze data distributions across source datasets."""
    print("\n" + "=" * 60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Analyze identity datasets
    for split in ["train", "val"]:
        json_path = DATA_DIR / f"identity_{split}.json"
        if json_path.exists():
            print(f"\n{split.upper()} SET:")
            with open(json_path) as f:
                data = json.load(f)

            # Count by source dataset (extract from file path)
            source_counts = Counter()
            for item in data:
                path = item["file_path"]
                if "dogfacenet" in path:
                    source_counts["DogFaceNet"] += 1
                elif "stanford_dogs" in path:
                    source_counts["Stanford Dogs"] += 1
                elif "oxford_pets" in path:
                    source_counts["Oxford Pets"] += 1
                elif "additional_identities" in path:
                    source_counts["Additional Identities"] += 1
                else:
                    source_counts["Other"] += 1

            total = len(data)
            print(f"  Total samples: {total}")
            for source, count in source_counts.most_common():
                percentage = (count / total) * 100
                print(f"  {source}: {count} ({percentage:.1f}%)")

    # Analyze COCO detection datasets
    coco_dir = DATA_DIR / "detector" / "coco"
    if coco_dir.exists():
        print("\nDETECTION DATASETS:")
        for json_file in coco_dir.glob("annotations_*.json"):
            split_name = json_file.stem.replace("annotations_", "")
            with open(json_file) as f:
                coco_data = json.load(f)

            # Count images by source (extract from file path)
            source_counts = Counter()
            for img in coco_data["images"]:
                path = img["file_name"]
                if "coco" in path:
                    source_counts["COCO"] += 1
                elif "stanford_dogs" in path:
                    source_counts["Stanford Dogs"] += 1
                elif "oxford_pets" in path:
                    source_counts["Oxford Pets"] += 1
                else:
                    source_counts["Other"] += 1

            total = len(coco_data["images"])
            print(f"\n  {split_name.upper()} SET:")
            print(f"    Total images: {total}")
            for source, count in source_counts.most_common():
                percentage = (count / total) * 100
                print(f"    {source}: {count} ({percentage:.1f}%)")


def inspect_coco_dataset(dataset_path, output_dir, num_samples, display):
    """Inspect COCO format dataset."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return

    # Look for COCO annotation files
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        # Single JSON file provided
        coco_files = [dataset_path]
        data_root = dataset_path.parents[
            2
        ]  # Assume structure: data/detector/coco/annotations_*.json
    else:
        # Directory provided, look for annotation files
        coco_files = list(dataset_path.glob("annotations_*.json"))
        data_root = (
            dataset_path.parents[1]
            if "coco" in str(dataset_path)
            else dataset_path.parent
        )

    if not coco_files:
        print(f"No COCO annotation files found in: {dataset_path}")
        return

    print(f"Found {len(coco_files)} COCO annotation files")

    for coco_file in coco_files:
        print(f"\nInspecting: {coco_file.name}")

        # Print statistics
        print_dataset_statistics(coco_file)

        # Create visualizations
        split_name = coco_file.stem.replace("annotations_", "")
        output_subdir = output_dir / f"coco_{split_name}"

        visualize_coco_annotations(
            coco_json_path=coco_file,
            data_root=data_root,
            output_dir=output_subdir,
            num_samples=num_samples,
            display=display,
        )


def inspect_yolo_dataset(yaml_path, output_dir, num_samples, display):
    """Inspect YOLO format dataset."""
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"Error: YOLO config file not found: {yaml_path}")
        return

    print(f"Inspecting YOLO dataset: {yaml_path.name}")

    # Create visualizations
    output_subdir = output_dir / f"yolo_{yaml_path.stem}"

    visualize_yolo_annotations(
        data_yaml_path=yaml_path,
        output_dir=output_subdir,
        num_samples=num_samples,
        display=display,
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Inspect datasets (COCO or YOLO format)"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Path to dataset (COCO dir/file or YOLO yaml). If not provided, shows data distributions only.",
    )
    parser.add_argument(
        "--format",
        choices=["coco", "yolo"],
        help="Dataset format (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/02_dataset_inspection",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample images to visualize",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display plots in windows (default: save only)",
    )
    parser.add_argument(
        "--distributions-only",
        action="store_true",
        help="Only show data distributions, skip visualizations",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Inspection")
    print("=" * 60)

    # Always show data distributions
    analyze_data_distributions()

    # If only distributions requested or no dataset path provided, exit here
    if args.distributions_only or args.dataset_path is None:
        return

    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    dataset_path = Path(args.dataset_path)

    # Auto-detect format if not specified
    if args.format is None:
        if dataset_path.suffix == ".yaml" or dataset_path.suffix == ".yml":
            args.format = "yolo"
        elif dataset_path.suffix == ".json" or (
            dataset_path.is_dir() and list(dataset_path.glob("*.json"))
        ):
            args.format = "coco"
        else:
            print("Error: Could not auto-detect format. Please specify --format")
            return

    print(f"\nDataset: {dataset_path}")
    print(f"Format: {args.format.upper()}")
    print(f"Output: {output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Display: {args.display}")

    # Inspect based on format
    if args.format == "coco":
        inspect_coco_dataset(dataset_path, output_dir, args.num_samples, args.display)
    elif args.format == "yolo":
        inspect_yolo_dataset(dataset_path, output_dir, args.num_samples, args.display)

    print(f"\nInspection complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
