
#!/usr/bin/env python3
"""
Create YOLOv8 dataset YAML and validation split for dog pose/keypoint detection.

This script:
1. Reads COCO keypoints annotations from data/coco_keypoints/annotations_train.json
2. Creates a validation split of up to 200 images
3. Copies val images to E:/data/combined_images/val/ (for Windows compatibility)
4. Creates data/dogs_keypoints.yaml with paths for Ultralytics YOLOv8

Requirements: Python 3.12+, PyTorch with CUDA available
"""

import os
import json
import random
import shutil
from pathlib import Path
import sys

# Verify PyTorch CUDA availability early
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - ensure PyTorch with CUDA is installed")
    print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}")
except ImportError:
    print("PyTorch not installed - install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent  # E:\Code\GitHub\immich-dogs
DATA_ROOT = REPO_ROOT / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_keypoints" / "annotations_train.json"
YOLO_YAML = DATA_ROOT / "dogs_keypoints.yaml"

# Combined images paths within repo (relative to REPO_ROOT)
COMBINED_IMAGES_TRAIN = DATA_ROOT / "combined_images" / "train"
COMBINED_IMAGES_VAL = DATA_ROOT / "combined_images" / "val"

MAX_VAL_IMAGES = 200

# Global cache for image searching (copied from diagnostic script)
_image_search_cache = {}

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

def create_validation_split(images):
    """Create a validation split of up to MAX_VAL_IMAGES images."""

    # Randomly shuffle and select validation images
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)

    val_images = shuffled_images[:min(MAX_VAL_IMAGES, len(images))]
    train_images = shuffled_images[MAX_VAL_IMAGES:]

    print(f"Validation split: {len(val_images)} images")
    print(f"Training split: {len(train_images)} images")

    return train_images, val_images

def resolve_image_path(file_name, base_path=DATA_ROOT):
    """Resolve relative paths from COCO annotations to absolute paths using multiple strategies."""
    # Define search directories for enhanced search
    search_dirs = [
        base_path / "oxford_pets" / "images",
        base_path / "stanford_dogs" / "images",
        base_path / "dogfacenet" / "DogFaceNet_224resized" / "after_4_bis",
    ]

    # Strategy 1: Direct path as given in annotations
    direct_path = base_path / file_name
    if direct_path.exists():
        return direct_path

    # Strategy 2: Handle Stanford Dogs format (e.g., "n02097658-silky_terrier/n02097658_6678.jpg")
    if "/" in file_name:
        breed_dir, img_file = file_name.split("/", 1)
        if breed_dir.startswith("n02"):  # Stanford Dogs format
            stanford_path = base_path / "stanford_dogs" / "images" / breed_dir / img_file
            if stanford_path.exists():
                return stanford_path

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
                return oxford_path

            # Try variations for Oxford Pets
            for separator in ['_', '.']:
                for ext in ['.jpg', '.png', '.jpeg']:
                    try:
                        oxford_path = base_path / "oxford_pets" / "images" / f"{breed_name}{separator}{number}{ext}"
                        if oxford_path.exists():
                            return oxford_path
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
                    return dogfacenet_path
        except:
            pass

    # Strategy 5: Enhanced search across all directories (similar to diagnostic script)
    base_name = Path(file_name).stem  # Remove extension from annotation filename
    found_path, search_method = find_image_by_pattern(base_name, search_dirs)
    if found_path:
        return found_path

    # If no strategy worked, return the direct path for error reporting
    return direct_path

def create_directory_structure():
    """Create necessary directories for validation images."""
    COMBINED_IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    COMBINED_IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {COMBINED_IMAGES_VAL}")
    print(f"Created directory: {COMBINED_IMAGES_TRAIN}")

def copy_validation_images(val_images):
    """Copy validation images to the combined images directory with improved error handling."""
    print(f"Copying {len(val_images)} validation images to {COMBINED_IMAGES_VAL}")

    copied_count = 0
    skipped_count = 0

    for image in val_images:
        src_path = resolve_image_path(image['file_name'])

        if not src_path.exists():
            print(f"Warning: Source image not found: {src_path}")
            skipped_count += 1
            continue

        # Keep original directory structure in val folder
        dst_path = COMBINED_IMAGES_VAL / image['file_name']
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {src_path} -> {dst_path}: {e}")
            skipped_count += 1

    print(f"Successfully copied: {copied_count} images")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} images (not found or copy failed)")

    return copied_count, skipped_count

def get_training_image_paths(train_images):
    """Get resolved paths for training images (keep in original locations)."""
    train_paths = []
    valid_train_paths = []
    missing_train_count = 0

    for image in train_images:
        full_path = resolve_image_path(image['file_name'])
        train_paths.append(str(full_path))

        if full_path.exists():
            valid_train_paths.append(str(full_path))
        else:
            missing_train_count += 1

    print(f"Training image paths: {len(train_paths)} total, {len(valid_train_paths)} valid files")
    if missing_train_count > 0:
        print(f"Warning: {missing_train_count} training images not found")

    return valid_train_paths, missing_train_count

def get_validation_image_paths():
    """Get paths to copied validation images."""
    val_paths = []

    if COMBINED_IMAGES_VAL.exists():
        for img_path in COMBINED_IMAGES_VAL.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                val_paths.append(str(img_path))

    print(f"Validation image paths found: {len(val_paths)}")
    return sorted(val_paths)

def create_yaml_config(train_paths, val_paths):
    """Create the YOLOv8 dataset YAML configuration file."""

    # Create train.txt file with all training image paths
    train_txt_path = DATA_ROOT / "combined_images" / "train.txt"
    print(f"Creating train.txt with {len(train_paths)} training image paths...")
    with open(train_txt_path, 'w') as f:
        for path in sorted(train_paths):
            f.write(f"{path}\n")

    yaml_content = f"""# YOLOv8 Keypoint Detection Dataset Configuration
# Created by scripts/phase1/make_yolo_splits.py

train: {train_txt_path}
val: {COMBINED_IMAGES_VAL}

# Number of classes
nc: 1

# Class names
names: ['dog']

# Keypoint shape: [num_keypoints, 3] where 3 is (x, y, visibility)
kpt_shape: [5, 3]

# Keypoint names
keypoints: ['left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear']
"""

    print(f"Creating YAML config at: {YOLO_YAML}")
    with open(YOLO_YAML, 'w') as f:
        f.write(yaml_content)

    print("YAML config created successfully")
    return yaml_content

def main():
    """Main execution function."""
    print("=" * 60)
    print("Creating YOLOv8 dataset split for dog pose detection")
    print("=" * 60)

    try:
        # Load and process annotations
        images, coco_data = load_coco_annotations()

        if len(images) == 0:
            print("ERROR: No images found in annotations")
            return False

        # Create validation split
        train_images, val_images = create_validation_split(images)

        # Create directory structure
        create_directory_structure()

        # Copy validation images
        copied_count, skipped_count = copy_validation_images(val_images)

        # Get training paths (original locations)
        valid_train_paths, missing_train_count = get_training_image_paths(train_images)

        # Get validation paths (copied locations)
        val_paths = get_validation_image_paths()

        # Create YAML configuration
        yaml_content = create_yaml_config(valid_train_paths, val_paths)

        print("\n" + "=" * 60)

        # Check if we have a working dataset
        if len(val_paths) == 0:
            print("ERROR: No validation images were successfully processed")
            print("Possible issues:")
            print("- Image files may not be extracted from tar archives")
            print("- Path resolution logic may need adjustment")
            print("- Original COCO annotations may have incorrect paths")
            return False

        if len(valid_train_paths) == 0:
            print("WARNING: No training images found - this will likely cause training issues")

        print(f"SUCCESS: Dataset configuration completed!")
        print(f"Training images: {len(valid_train_paths)} (original locations)")
        print(f"Validation images: {len(val_paths)} (copied to {COMBINED_IMAGES_VAL})")
        print(f"YAML config: {YOLO_YAML}")

        if skipped_count > 0 or missing_train_count > 0:
            print("\nWARNING: Some images were not found:")
            print(f"- Validation images skipped: {skipped_count}")
            print(f"- Training images missing: {missing_train_count}")
            print("This may indicate path resolution issues.")

        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
