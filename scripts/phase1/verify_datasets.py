#!/usr/bin/env python3
"""
Dataset Verification and Sample Manifest Creator
Verifies presence of required datasets and creates sample manifest with images.
Usage: python verify_datasets.py [--data-root DATA_ROOT]
"""

import os
import json
import pathlib
import argparse
import csv
import random
import shutil
from typing import Dict, List, Tuple, Optional

# Dataset configuration
DATASETS = {
    'coco': {
        'image_dirs': ['train2017', 'val2017', 'test2017'],
        'ann_dir': 'annotations',
        'ann_files': ['instances_train2017.json', 'instances_val2017.json'],
        'label_hint': 'dog'
    },
    'stanford_dogs': {
        'image_dirs': ['images'],
        'ann_dir': 'annotation',
        'ann_files': [],
        'label_hint': 'dog'
    },
    'dogfacenet': {
        'image_dirs': ['DogFaceNet_224resized', 'DogFaceNet_alignment', 'DogFaceNet_large'],
        'ann_dir': None,
        'ann_files': ['classes_test.txt', 'classes_train.txt'],
        'label_hint': 'dog-head'
    },
    'oxford_pets': {
        'image_dirs': ['images'],
        'ann_dir': 'annotations',
        'ann_files': [],
        'label_hint': 'dog'  # Oxford-IIIT has dogs too
    }
}

def count_files_in_dir(dir_path: pathlib.Path, extensions: List[str]) -> int:
    """Count files with given extensions in directory."""
    if not dir_path.exists():
        return 0
    count = 0
    for ext in extensions:
        count += len(list(dir_path.glob(f'**/*{ext}')))
    return count

def sample_images(image_paths: List[pathlib.Path], max_samples: int = 10) -> List[pathlib.Path]:
    """Randomly sample up to max_samples image paths."""
    if len(image_paths) <= max_samples:
        return image_paths
    return random.sample(image_paths, max_samples)

def verify_dataset(dataset_name: str, base_path: pathlib.Path, output_csv: List[Dict], sample_dir: pathlib.Path) -> Tuple[int, int, bool]:
    """Verify a single dataset and collect samples."""
    config = DATASETS[dataset_name]
    dataset_path = base_path / dataset_name
    images_created = False

    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Path: {dataset_path}")

    if not dataset_path.exists():
        print(f"❌ Missing: {dataset_path}")
        return 0, 0, True

    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png']
    total_images = 0
    all_image_paths = []

    for img_dir in config['image_dirs']:
        img_path = dataset_path / img_dir
        print(f"  Image dir: {img_path}")
        img_count = count_files_in_dir(img_path, image_extensions)
        print(f"    Images: {img_count}")
        total_images += img_count
        if img_path.exists():
            all_image_paths.extend(list(img_path.glob(f'**/*{image_extensions[0]}')))

    # Count annotations
    total_annotations = 0
    if config['ann_dir']:
        ann_path = dataset_path / config['ann_dir']
        print(f"  Annotations dir: {ann_path}")
        if ann_path.exists():
            ann_count = count_files_in_dir(ann_path, ['.json'])
            print(f"    JSON files: {ann_count}")
            total_annotations = ann_count
        else:
            print(f"  ⚠️ Missing annotations dir")

    # Sample images
    label_hint = config['label_hint']
    if all_image_paths:
        sampled_paths = sample_images(all_image_paths, 10)
        for i, img_path in enumerate(sampled_paths):
            # Store relative path from base_path
            rel_path = img_path.relative_to(base_path)
            abs_path = img_path.resolve()
            output_csv.append({
                'dataset': dataset_name,
                'rel_image_path': str(rel_path),
                'abs_path': str(abs_path),
                'label_hint': label_hint
            })

            # Copy sample image
            sample_name = f"{dataset_name}_{i+1:03d}{img_path.suffix}"
            sample_path = sample_dir / sample_name
            try:
                shutil.copy2(img_path, sample_path)
                print(f"    ✅ Copied sample {i+1}: {sample_name}")
                images_created = True
            except Exception as e:
                print(f"    ❌ Failed to copy {img_path}: {e}")

    return total_images, total_annotations, images_created

def main():
    parser = argparse.ArgumentParser(description='Verify datasets and create sample manifest')
    parser.add_argument('--data-root', default='data', help='Path to data root directory')
    args = parser.parse_args()

    base_data = pathlib.Path(args.data_root)
    print(f"Data root: {base_data.resolve()}")

    manifest_path = base_data / 'manifest' / 'phase1_samples.csv'
    sample_dir = pathlib.Path('outputs/phase1/sample_images')

    # Ensure output directories exist
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_csv = []
    missing_datasets = []

    for dataset_name in DATASETS:
        images, annotations, has_images = verify_dataset(dataset_name, base_data, output_csv, sample_dir)
        if images == 0 and dataset_name in ['coco', 'stanford_dogs']:
            missing_datasets.append(dataset_name)

    # Write CSV
    if output_csv:
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'rel_image_path', 'abs_path', 'label_hint'])
            writer.writeheader()
            writer.writerows(output_csv)
        print(f"\n✅ Manifest created: {manifest_path.resolve()}")
        print(f"   Total samples: {len(output_csv)}")
    else:
        print("\n⚠️ No samples found - all datasets missing?")

    # Summary
    if missing_datasets:
        print(f"\n⚠️ Critical datasets missing: {', '.join(missing_datasets)}")
        print("Please download per prerequisites.")
    else:
        print("\n✅ All critical datasets present.")

if __name__ == '__main__':
    random.seed(42)  # Reproducible sampling
    main()