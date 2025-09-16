"""
YOLO format converter for detection datasets.

Converts COCO format detection datasets to YOLO format for training with Ultralytics.
"""

import json
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple


class CocoToYoloDetectionConverter:
    """Convert COCO detection format to YOLO format."""
    
    def __init__(self, coco_annotations_dir: str, labels_output_dir: str, 
                 data_root: str, yaml_output_path: str):
        """Initialize converter with paths."""
        self.coco_annotations_dir = Path(coco_annotations_dir)
        self.labels_output_dir = Path(labels_output_dir)
        self.data_root = Path(data_root)
        self.yaml_output_path = Path(yaml_output_path)
    
    def convert_split(self, split_name: str) -> List[str]:
        """Processes a single split (e.g., 'train' or 'val')."""
        coco_json_path = self.coco_annotations_dir / f"annotations_{split_name}.json"

        if not coco_json_path.exists():
            print(f"Warning: Annotation file not found, skipping split '{split_name}': {coco_json_path}")
            return []

        print(f"Processing {split_name} split from {coco_json_path}...")
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        images_map = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        image_paths = []
        found_annotations_count = 0
        written_labels_count = 0
        
        for img_id, image_info in tqdm(images_map.items(), desc=f"Generating {split_name} labels"):
            img_height = image_info['height']
            img_width = image_info['width']
            relative_img_path = image_info['file_name']
            
            relative_img_path_str = str(Path(relative_img_path))
            # Replace 'images' with 'labels' in path
            relative_label_path_str = relative_img_path_str.replace('images', 'labels', 1)
            label_path = (self.labels_output_dir / Path(relative_label_path_str)).with_suffix('.txt')

            image_paths.append(str(self.data_root / relative_img_path))
            label_path.parent.mkdir(parents=True, exist_ok=True)

            with open(label_path, 'w') as f_label:
                if img_id in annotations_by_image:
                    found_annotations_count += 1
                    annotations = annotations_by_image[img_id]
                    if len(annotations) > 0:
                        written_labels_count += 1
                    for ann in annotations:
                        bbox = ann['bbox']
                        x, y, w, h = bbox

                        # Clamp bbox to image boundaries
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(img_width, x + w)
                        y2 = min(img_height, y + h)

                        # Log if clamping was necessary
                        if x1 != x or y1 != y or x2 != (x + w) or y2 != (y + h):
                            print(f"[WARN] Clamped bbox for {relative_img_path}. "
                                  f"Original: {[x, y, w, h]}, Clamped: {[x1, y1, x2 - x1, y2 - y1]}")

                        final_w = x2 - x1
                        final_h = y2 - y1

                        if final_w <= 0 or final_h <= 0: 
                            continue

                        # Convert to YOLO format (normalized center coordinates)
                        x_center_norm = (x1 + final_w / 2) / img_width
                        y_center_norm = (y1 + final_h / 2) / img_height
                        width_norm = final_w / img_width
                        height_norm = final_h / img_height

                        f_label.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        print(f"Split {split_name}: Found annotations for {found_annotations_count} images, "
              f"wrote labels for {written_labels_count} images")
        return image_paths

    def create_yaml_config(self, train_image_paths: List[str], val_image_paths: List[str]) -> None:
        """Creates the final YOLOv8 dataset YAML configuration file for detection."""
        # Create train/val text files
        train_txt_path = self.data_root / "detector/train.txt"
        val_txt_path = self.data_root / "detector/val.txt"

        with open(train_txt_path, 'w') as f:
            for path in sorted(train_image_paths):
                f.write(f"{Path(path).as_posix()}\n")
        print(f"Created {train_txt_path.name} with {len(train_image_paths)} image paths.")

        with open(val_txt_path, 'w') as f:
            for path in sorted(val_image_paths):
                f.write(f"{Path(path).as_posix()}\n")
        print(f"Created {val_txt_path.name} with {len(val_image_paths)} image paths.")

        # Create YAML for detection
        yaml_content = {
            'path': Path(self.data_root.resolve()).as_posix(),
            'train': Path(train_txt_path.resolve()).as_posix(),
            'val': Path(val_txt_path.resolve()).as_posix(),
            'nc': 1,
            'names': ['dog'],
        }

        with open(self.yaml_output_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        print(f"Successfully created YAML config for detection at: {self.yaml_output_path}")

    def convert(self) -> None:
        """Main conversion function."""
        print("=" * 60)
        print("Converting COCO Detector Dataset to YOLOv8 Detection Format")
        print("=" * 60)

        train_paths = self.convert_split('train')
        val_paths = self.convert_split('val')

        if not train_paths and not val_paths:
            print(f"Error: No data was processed. Check that your COCO JSON files exist in {self.coco_annotations_dir}")
            return

        self.create_yaml_config(train_paths, val_paths)

        print("\nConversion complete!")
        print("You are now ready to train the YOLOv8 detector model.")
        print("=" * 60)


def create_default_converter() -> CocoToYoloDetectionConverter:
    """Create converter with default paths."""
    return CocoToYoloDetectionConverter(
        coco_annotations_dir="data/detector/coco",
        labels_output_dir="data",
        data_root="data",
        yaml_output_path="data/detector/dogs_detection.yaml"
    )
