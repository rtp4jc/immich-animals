"""
Dataset converter for creating COCO format keypoint datasets.

Processes StanfordExtra dataset with keypoint annotations, crops dog regions,
and creates a unified COCO format dataset for keypoint training.
"""

import os
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Dict, List, Tuple, Any, Optional


class CocoKeypointDatasetConverter:
    """Convert StanfordExtra keypoint dataset to COCO format."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize converter with configuration."""
        self.config = config
        self.source_json = Path(config['source_json'])
        self.cropped_image_dir = Path(config['cropped_image_dir'])
        self.output_coco_dir = Path(config['output_coco_dir'])
        self.data_root = Path(config['data_root'])
        
        # Parameters
        self.padding_factor = config.get('padding_factor', 0.2)
        self.train_val_split = config.get('train_val_split', 0.9)
        
        # Keypoint mapping from 24-point to 4-point schema
        self.keypoint_map = {
            'nose': 16,
            'chin': 17,
            'left_ear_base': 14,
            'right_ear_base': 15,
        }
        self.keypoint_names = ['nose', 'chin', 'left_ear_base', 'right_ear_base']

    def _setup_output_dirs(self) -> None:
        """Create output directories, clearing existing ones."""
        for dir_path in [self.cropped_image_dir, self.output_coco_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)

    def _map_and_transform_keypoints(self, joints: List[List[float]], 
                                   x_offset: int, y_offset: int) -> List[float]:
        """
        Selects our 4 target keypoints from the 24 available, and transforms
        their coordinates from original image space to cropped space.
        """
        output_keypoints = []
        for name in self.keypoint_names:
            idx = self.keypoint_map[name]
            
            if len(joints) > idx and joints[idx][2] > 0:
                kpt = joints[idx]
                x, y, v = kpt[0], kpt[1], kpt[2]
                new_x = x - x_offset
                new_y = y - y_offset
                output_keypoints.extend([new_x, new_y, v])
            else:
                output_keypoints.extend([0, 0, 0])
                
        return output_keypoints

    def _validate_keypoints(self, keypoints: List[float], width: int, height: int) -> List[float]:
        """Validate transformed keypoints to ensure they are within the cropped image bounds."""
        final_kpts = []
        for i in range(0, len(keypoints), 3):
            kx, ky, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v > 0:
                if not (0 <= kx < width and 0 <= ky < height):
                    v = 0  # Mark as not visible if outside crop
                    kx = 0  # Set to 0,0 as per convention
                    ky = 0
            final_kpts.extend([kx, ky, v])
        return final_kpts

    def _process_image(self, entry: Dict[str, Any], image_id_counter: int, 
                      annotation_id_counter: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Process a single image entry and return image and annotation dictionaries."""
        if entry.get('is_multiple_dogs', False):
            return None, None

        original_img_path = self.data_root / "stanford_dogs" / "images" / entry['img_path']
        if not original_img_path.exists():
            return None, None

        img = cv2.imread(str(original_img_path))
        if img is None:
            return None, None

        img_h, img_w, _ = img.shape
        x, y, w, h = [int(v) for v in entry['img_bbox']]

        # Clamp bbox to image boundaries
        x = max(0, x)
        y = max(0, y)
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y
        
        if w <= 0 or h <= 0:
            return None, None

        # Add padding
        pad_w = int(w * self.padding_factor)
        pad_h = int(h * self.padding_factor)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return None, None

        # Save cropped image
        new_filename = f"{Path(entry['img_path']).stem}_{annotation_id_counter}.jpg"
        cropped_img_path = self.cropped_image_dir / new_filename
        cv2.imwrite(str(cropped_img_path), cropped_img)

        new_height, new_width, _ = cropped_img.shape
        image_entry = {
            'id': image_id_counter,
            'width': new_width,
            'height': new_height,
            'file_name': str(cropped_img_path.relative_to(self.data_root)),
            'source_dataset': 'stanford_extra_cropped'
        }

        # Transform and validate keypoints
        transformed_kpts = self._map_and_transform_keypoints(entry['joints'], x1, y1)
        final_kpts = self._validate_keypoints(transformed_kpts, new_width, new_height)

        # Adjust bbox coordinates to cropped image space
        new_bbox_x = x - x1
        new_bbox_y = y - y1
        new_bbox = [new_bbox_x, new_bbox_y, w, h]

        annotation_entry = {
            'id': annotation_id_counter,
            'image_id': image_id_counter,
            'category_id': 1,
            'bbox': new_bbox,
            'keypoints': final_kpts,
            'num_keypoints': sum(1 for i in range(2, len(final_kpts), 3) if final_kpts[i] > 0),
            'area': w * h,
            'iscrowd': 0
        }

        return image_entry, annotation_entry

    def convert(self) -> None:
        """Main conversion function."""
        print("=" * 60)
        print("Creating Cropped Keypoint Dataset in COCO Format")
        print("=" * 60)

        if not self.source_json.exists():
            print(f"Error: Source keypoint JSON not found: {self.source_json}")
            return

        self._setup_output_dirs()

        with open(self.source_json, 'r') as f:
            source_data = json.load(f)

        coco_output = {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1, 
                'name': 'dog',
                'keypoints': self.keypoint_names,
                'skeleton': []
            }]
        }

        image_id_counter = 0
        annotation_id_counter = 0

        for entry in tqdm(source_data, desc="Processing and cropping images"):
            image_entry, annotation_entry = self._process_image(
                entry, image_id_counter, annotation_id_counter)
            
            if image_entry and annotation_entry:
                coco_output['images'].append(image_entry)
                coco_output['annotations'].append(annotation_entry)
                image_id_counter += 1
                annotation_id_counter += 1

        # Split data into train and validation sets
        random.seed(42)
        random.shuffle(coco_output['images'])
        split_idx = int(len(coco_output['images']) * self.train_val_split)

        train_images = coco_output['images'][:split_idx]
        val_images = coco_output['images'][split_idx:]

        train_image_ids = {img['id'] for img in train_images}
        val_image_ids = {img['id'] for img in val_images}

        train_annotations = [ann for ann in coco_output['annotations'] if ann['image_id'] in train_image_ids]
        val_annotations = [ann for ann in coco_output['annotations'] if ann['image_id'] in val_image_ids]

        train_coco = {'images': train_images, 'annotations': train_annotations, 'categories': coco_output['categories']}
        val_coco = {'images': val_images, 'annotations': val_annotations, 'categories': coco_output['categories']}

        # Save COCO format files
        with open(self.output_coco_dir / 'annotations_train.json', 'w') as f:
            json.dump(train_coco, f, indent=2)
        with open(self.output_coco_dir / 'annotations_val.json', 'w') as f:
            json.dump(val_coco, f, indent=2)

        print(f"\nSuccessfully created {len(train_images)} training samples and {len(val_images)} validation samples.")
        print(f"Cropped images saved to: {self.cropped_image_dir}")
        print(f"COCO annotations saved to: {self.output_coco_dir}")
        print("=" * 60)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for keypoint dataset conversion."""
    return {
        'source_json': 'data/stanford_dogs/stanford_extra_keypoints.json',
        'cropped_image_dir': 'data/keypoints/images',
        'output_coco_dir': 'data/keypoints/coco',
        'data_root': 'data',
        'padding_factor': 0.2,
        'train_val_split': 0.9
    }
