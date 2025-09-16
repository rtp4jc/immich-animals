"""
YOLO format converter for keypoint datasets.

Converts COCO format keypoint datasets to YOLO pose format for training with Ultralytics.
"""

import json
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any


class CocoToYoloKeypointConverter:
    """Convert COCO keypoint format to YOLO pose format."""
    
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
        
        for img_id, image_info in tqdm(images_map.items(), desc=f"Generating {split_name} labels"):
            img_height = image_info['height']
            img_width = image_info['width']
            relative_img_path = image_info['file_name']
            
            # All images are in one folder, so label path is simple
            label_path = self.labels_output_dir / Path(relative_img_path).with_suffix('.txt').name
            
            image_paths.append(str(self.data_root / relative_img_path))
            label_path.parent.mkdir(parents=True, exist_ok=True)

            with open(label_path, 'w') as f_label:
                if img_id in annotations_by_image:
                    annotations = annotations_by_image[img_id]
                    for ann in annotations:
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        
                        # Convert bbox to normalized center coordinates
                        x_center_norm = (x + w / 2) / img_width
                        y_center_norm = (y + h / 2) / img_height
                        width_norm = w / img_width
                        height_norm = h / img_height

                        # Process keypoints
                        keypoints = ann.get('keypoints', [])
                        kpts_str = ""
                        if keypoints and ann.get('num_keypoints', 0) > 0:
                            for i in range(0, len(keypoints), 3):
                                kpt_x = keypoints[i] / img_width
                                kpt_y = keypoints[i+1] / img_height
                                kpt_v = keypoints[i+2]
                                # YOLO format expects v=0 (not present), v=1 (present but not visible), v=2 (visible)
                                if kpt_v > 0: 
                                    kpt_v = 2 
                                kpts_str += f" {kpt_x:.6f} {kpt_y:.6f} {kpt_v}"
                        else:
                            # If no keypoints, write zeros for 4 keypoints * 3 values each
                            kpts_str = " 0" * (4 * 3) 
                        
                        f_label.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}{kpts_str}\n")

        print(f"Split {split_name}: Processed {len(image_paths)} images")
        return image_paths

    def create_yaml_config(self, train_image_paths: List[str], val_image_paths: List[str]) -> None:
        """Creates the final YOLOv8 dataset YAML configuration file for keypoint detection."""
        train_txt_path = self.data_root / "keypoints/train.txt"
        val_txt_path = self.data_root / "keypoints/val.txt"

        with open(train_txt_path, 'w') as f:
            for path in sorted(train_image_paths):
                f.write(f"{Path(path).as_posix()}\n")
        print(f"Created {train_txt_path.name} with {len(train_image_paths)} image paths.")

        with open(val_txt_path, 'w') as f:
            for path in sorted(val_image_paths):
                f.write(f"{Path(path).as_posix()}\n")
        print(f"Created {val_txt_path.name} with {len(val_image_paths)} image paths.")

        # Create YAML for keypoints, specifying kpt_shape and flip_idx
        yaml_content = {
            'path': Path(self.data_root.resolve()).as_posix(),
            'train': Path(train_txt_path.resolve()).as_posix(),
            'val': Path(val_txt_path.resolve()).as_posix(),
            'nc': 1,
            'names': ['dog'],
            'kpt_shape': [4, 3],  # 4 keypoints, 3 dims (x, y, visibility)
            # Keypoints: ['nose', 'chin', 'left_ear_base', 'right_ear_base']
            # Indices:      0,      1,           2,               3
            'flip_idx': [0, 1, 3, 2],  # Swap left and right ear base
        }

        with open(self.yaml_output_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        print(f"Successfully created YAML config for keypoints at: {self.yaml_output_path}")

    def convert(self) -> None:
        """Main conversion function."""
        print("=" * 60)
        print("Converting Cropped COCO Keypoint Dataset to YOLOv8 Pose Format")
        print("=" * 60)

        if self.labels_output_dir.exists():
            shutil.rmtree(self.labels_output_dir)
        self.labels_output_dir.mkdir(parents=True)

        train_paths = self.convert_split('train')
        val_paths = self.convert_split('val')

        if not train_paths and not val_paths:
            print(f"Error: No data was processed. Check that your COCO JSON files exist in {self.coco_annotations_dir}")
            return

        self.create_yaml_config(train_paths, val_paths)

        print("\nConversion complete!")
        print("You are now ready to train the YOLOv8 keypoint model.")
        print("=" * 60)


def create_default_converter() -> CocoToYoloKeypointConverter:
    """Create converter with default paths."""
    return CocoToYoloKeypointConverter(
        coco_annotations_dir="data/keypoints/coco",
        labels_output_dir="data/keypoints/labels",
        data_root="data",
        yaml_output_path="data/keypoints/dogs_keypoints_only.yaml"
    )
