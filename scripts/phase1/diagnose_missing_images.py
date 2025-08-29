#!/usr/bin/env python3
"""
Diagnose missing images from COCO annotations for dog pose detection.

This script helps identify where missing images are located by:
1. Loading COCO annotations that failed to resolve in make_yolo_splits.py
2. Searching for missing images in alternate locations
3. Checking if images exist with different naming conventions
4. Providing detailed diagnostics and remediation suggestions

Requirements: Python 3.12+
"""

import os
import json
import glob
from pathlib import Path
from functools import lru_cache

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent  # E:\Code\GitHub\immich-dogs
DATA_ROOT = REPO_ROOT / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_keypoints" / "annotations_train.json"

# Global cache for image searching
_image_search_cache = {}

def load_coco_annotations():
    """Load COCO annotations and return relevant data."""
    print(f"Loading annotations from: {COCO_ANNOTATIONS}")

    if not COCO_ANNOTATIONS.exists():
        raise FileNotFoundError(f"Annotations file not found: {COCO_ANNOTATIONS}")

    with open(COCO_ANNOTATIONS, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    print(f"Found {len(images)} images in annotations")

    return images, coco_data

def resolve_image_path(file_name, base_path=DATA_ROOT):
    """Smart path resolution using multiple strategies (copied from make_yolo_splits.py)."""
    # Try multiple strategies to find images based on different naming conventions

    # Strategy 1: Direct path as given in annotations
    direct_path = base_path / file_name
    if direct_path.exists():
        return direct_path, "direct_match"

    # Strategy 2: Handle Stanford Dogs format (e.g., "n02097658-silky_terrier/n02097658_6678.jpg")
    if "/" in file_name:
        breed_dir, img_file = file_name.split("/", 1)
        if breed_dir.startswith("n02"):  # Stanford Dogs format
            stanford_path = base_path / "stanford_dogs" / "images" / breed_dir / img_file
            if stanford_path.exists():
                return stanford_path, "stanford_dogs"

    # Strategy 3: Handle Oxford Pets format (e.g., "oxford_pets/images/samoyed.47.jpg" -> "samoyed_47.jpg")
    if file_name.startswith("oxford_pets/images/") and "." in file_name.split("/")[-1]:
        oxford_img = file_name.split("/")[-1]  # e.g., "samoyed.47.jpg"
        if "." in oxford_img:
            breed_name, number = oxford_img.rsplit(".", 1)
            # Remove file extension from number
            if "." in number:
                number, ext = number.split(".", 1)

            oxford_path = base_path / "oxford_pets" / "images" / f"{breed_name}_{number}.jpg"
            if oxford_path.exists():
                return oxford_path, "oxford_pets_underscore"

            # Try variations for Oxford Pets
            for separator in ['_', '.']:
                for ext in ['.jpg', '.png', '.jpeg']:
                    try:
                        oxford_path = base_path / "oxford_pets" / "images" / f"{breed_name}{separator}{number}{ext}"
                        if oxford_path.exists():
                            return oxford_path, f"oxford_pets_{separator}_format"
                    except:
                        continue

    # Strategy 4: Handle DogFaceNet format if it's a simple filename with dots
    if "/" not in file_name and file_name.count(".") >= 2:  # e.g., "0.123.jpg"
        try:
            parts = file_name.split(".")
            if len(parts) >= 3:
                folder_idx = parts[0]
                img_idx = ".".join(parts[1:-1])  # Handle cases like "0.123.jpg"
                dogfacenet_path = base_path / "dogfacenet" / "DogFaceNet_224resized" / "after_4_bis" / folder_idx / f"{folder_idx}.{img_idx}.jpg"
                if dogfacenet_path.exists():
                    return dogfacenet_path, "dogfacenet"
        except:
            pass

    # If no strategy worked, return the direct path for error reporting
    return direct_path, "not_found"

def find_image_by_pattern(base_name, search_dirs):
    """Find image by searching for patterns in multiple directories."""
    global _image_search_cache

    # Create cache key from base_name and search_dirs (convert to hashable format)
    cache_key = (base_name,) + tuple(str(d) for d in search_dirs)

    # Check cache first
    if cache_key in _image_search_cache:
        return _image_search_cache[cache_key]

    # Compute result
    result = _find_image_by_pattern_impl(base_name, search_dirs)
    _image_search_cache[cache_key] = result

    # Limit cache size
    if len(_image_search_cache) > 1000:
        # Clear oldest entries (simple FIFO)
        oldest_keys = list(_image_search_cache.keys())[:500]
        for key in oldest_keys:
            _image_search_cache.pop(key, None)

    return result

def _find_image_by_pattern_impl(base_name, search_dirs):
    """Implementation of find_image_by_pattern without caching."""
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Try different file extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            # Exact match
            exact_path = search_dir / f"{base_name}{ext}"
            if exact_path.exists():
                return exact_path, f"found_in_{search_dir.name}_exact"

            # Try with underscore instead of dot
            if '.' in base_name:
                alt_name = base_name.replace('.', '_')
                alt_path = search_dir / f"{alt_name}{ext}"
                if alt_path.exists():
                    return alt_path, f"found_in_{search_dir.name}_underscore_alt"

    return None, "not_found"

def diagnose_missing_images():
    """Main diagnostic function."""
    print("=" * 70)
    print("DIAGNOSTIC: Finding missing images from COCO annotations")
    print("=" * 70)

    try:
        # Load annotations
        images, coco_data = load_coco_annotations()

        # Define search directories
        search_dirs = [
            DATA_ROOT / "oxford_pets" / "images",
            DATA_ROOT / "stanford_dogs" / "images",
            DATA_ROOT / "dogfacenet" / "DogFaceNet_224resized" / "after_4_bis",
        ]

        # Track statistics
        stats = {
            'total_images': len(images),
            'found_direct': 0,
            'found_stanford': 0,
            'found_oxford_variations': [],
            'found_dogfacenet': 0,
            'found_alternate_search': 0,
            'still_missing': 0,
            'missing_by_category': {},
            'sample_missing': []
        }

        print(f"\nAnalyzing {len(images)} images...")

        for i, image in enumerate(images):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(images)} images analyzed")

            file_name = image['file_name']
            resolved_path, strategy = resolve_image_path(file_name)

            # Track found images
            if strategy == "direct_match":
                stats['found_direct'] += 1
                continue
            elif strategy == "stanford_dogs":
                stats['found_stanford'] += 1
                continue
            elif strategy.startswith("oxford_pets"):
                stats['found_oxford_variations'].append(strategy)
                continue
            elif strategy == "dogfacenet":
                stats['found_dogfacenet'] += 1
                continue

            # For missing images, try additional search strategies
            found_alternative = None
            alt_strategy = "not_found"

            # Try searching with different patterns
            base_name = Path(file_name).stem  # Remove extension
            found_path, alt_strategy = find_image_by_pattern(base_name, search_dirs)

            if found_path:
                found_alternative = found_path
                stats['found_alternate_search'] += 1
            else:
                stats['still_missing'] += 1

                # Categorize missing images
                if file_name.startswith('oxford_pets/'):
                    category = 'oxford_pets'
                elif '/' in file_name and file_name.split('/')[0].startswith('n02'):
                    category = 'stanford_dogs'
                elif file_name.count('.') >= 2:
                    category = 'dogfacenet'
                else:
                    category = 'other'

                if category not in stats['missing_by_category']:
                    stats['missing_by_category'][category] = 0
                stats['missing_by_category'][category] += 1

                # Keep sample of missing images for reporting
                if len(stats['sample_missing']) < 10:
                    stats['sample_missing'].append({
                        'file_name': file_name,
                        'width': image.get('width', 'unknown'),
                        'height': image.get('height', 'unknown'),
                        'category': category
                    })

        # Print comprehensive report
        print("\n" + "=" * 70)
        print("DIAGNOSTIC REPORT")
        print("=" * 70)

        print("OVERALL STATISTICS:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Found (direct path): {stats['found_direct']}")
        print(f"  Found (Stanford Dogs): {stats['found_stanford']}")
        print(f"  Found (Oxford Pets variations): {len(stats['found_oxford_variations'])}")
        print(f"  Found (DogFaceNet): {stats['found_dogfacenet']}")
        print(f"  Found (alternate search): {stats['found_alternate_search']}")
        print(f"  Still missing: {stats['still_missing']}")

        found_total = stats['found_direct'] + stats['found_stanford'] + len(stats['found_oxford_variations']) + stats['found_dogfacenet'] + stats['found_alternate_search']
        coverage_pct = (found_total / stats['total_images']) * 100
        print(".1f")

        print("MISSING BY CATEGORY:")
        for category, count in stats['missing_by_category'].items():
            pct = (count / stats['still_missing']) * 100
            print(".1f")

        if stats['found_oxford_variations']:
            print("OXFORD PETS VARIATIONS FOUND:")
            from collections import Counter
            variation_counts = Counter(stats['found_oxford_variations'])
            for variation, count in variation_counts.most_common():
                print(f"  {variation}: {count}")

        if stats['sample_missing']:
            print("SAMPLE MISSING IMAGES:")
            for i, sample in enumerate(stats['sample_missing'], 1):
                print(f"  {i}. {sample['file_name']} ({sample['width']}x{sample['height']}) - Category: {sample['category']}")

        print("RECOMMENDATIONS:")
        if stats['missing_by_category'].get('oxford_pets', 0) > 0:
            print("  • Oxford Pets: Check if tar files need extraction or if naming conversion is needed")
        if stats['missing_by_category'].get('stanford_dogs', 0) > 0:
            print("  • Stanford Dogs: Verify all breed directories exist in data/stanford_dogs/images/")
        if stats['missing_by_category'].get('dogfacenet', 0) > 0:
            print("  • DogFaceNet: May need different path parsing logic")
        if stats['missing_by_category'].get('other', 0) > 0:
            print("  • Other: Check for additional dataset sources in data/ directory")

        if coverage_pct > 50:
            print("  • Dataset appears viable for training with ~60%+ coverage")
        else:
            print("  • Low coverage - investigate missing files before training")

        print("\n" + "=" * 70)
        print("Diagnostic complete!")
        print("=" * 70)

        return stats

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    stats = diagnose_missing_images()
    if stats:
        exit_code = 1 if stats['still_missing'] > 0 else 0
    else:
        exit_code = 1
    exit(exit_code)