#!/usr/bin/env python3
"""
Convert available keypoints/masks to COCO-style keypoints annotations for dog detection.

This script creates COCO-format annotations with bounding boxes and 5-point keypoints:
[left_eye_x, left_eye_y, v, right_eye_x, right_eye_y, v, nose_x, nose_y, v,
 left_ear_x, left_ear_y, v, right_ear_x, right_ear_y, v]

Where v = 0 (not visible), 1 (visible/occluded), 2 (fully visible)

Maps StanfordExtra keypoints (24-point) to our compact 5-point schema.
Synthesizes keypoints from masks when explicit landmarks unavailable.
"""
import os
import json
import random
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import argparse


class CocoKeypointsConverter:
    """Convert various dog datasets to COCO keypoints format."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.debug_dir = self.output_dir / 'debug_annotations'
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # COCO format structure
        self.coco_template = {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1,
                'name': 'dog',
                'keypoints': ['left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear'],
                'skeleton': []  # Empty for pose estimation
            }]
        }

        self.image_id = 0
        self.annotation_id = 0

    def _map_stanford_keypoints(self, joints):
        """
        Map StanfordExtra 24-point joints to our 5-point keypoint schema.

        Stanford joints appear to follow this pattern:
        - 0-5: Head/snout area
        - 6-8: Right side landmarks
        - 9-13: Left side landmarks
        - 14-15: Eyes
        - 16-23: Additional face/body points

        Returns: [x,y,v, x,y,v, ...] for 5 keypoints (10 values)
        """
        keypoints = []

        # Extract available joints, preferring visible ones (v > 0)
        candidate_eyes = []
        candidate_ears = []

        # Eyes: typically joints 14, 15 (face level points)
        if len(joints) > 15 and joints[15][2] > 0:  # Left eye (eye level)
            keypoints.extend(joints[15][:2] + [joints[15][2]])
        else:
            keypoints.extend([0, 0, 0])  # Not visible

        if len(joints) > 14 and joints[14][2] > 0:  # Right eye
            keypoints.extend(joints[14][:2] + [joints[14][2]])
        else:
            keypoints.extend([0, 0, 0])

        # Nose: try joint 17, or centroid of face joints
        nose_candidates = []
        if len(joints) > 17 and joints[17][2] > 0:
            nose_candidates.append(joints[17][:2])
        if len(joints) > 16 and joints[16][2] > 0:
            nose_candidates.append(joints[16][:2])
        if len(joints) > 12 and joints[12][2] > 0:
            nose_candidates.append(joints[12][:2])

        if nose_candidates:
            nose_x = sum(p[0] for p in nose_candidates) / len(nose_candidates)
            nose_y = sum(p[1] for p in nose_candidates) / len(nose_candidates)
            keypoints.extend([nose_x, nose_y, 1])  # Visible since we detected it
        else:
            keypoints.extend([0, 0, 0])

        # Left ear: try joints near head/torso area (joints 6, 9-13)
        left_candidates = []
        for i in [6, 13, 9, 10, 11, 12]:  # Prioritize outer points
            if len(joints) > i and joints[i][2] > 0:
                left_candidates.append(joints[i][:2])

        if left_candidates:
            left_ear_x = sum(p[0] for p in left_candidates) / len(left_candidates)
            left_ear_y = sum(p[1] for p in left_candidates) / len(left_candidates)
            keypoints.extend([left_ear_x, left_ear_y, 1])  # Approximate
        else:
            keypoints.extend([0, 0, 0])

        # Right ear: joints near head (joints 8, 0-5)
        right_candidates = []
        for i in [8, 0, 1, 2, 3, 4, 5]:
            if len(joints) > i and joints[i][2] > 0:
                right_candidates.append(joints[i][:2])

        if right_candidates:
            right_ear_x = sum(p[0] for p in right_candidates) / len(right_candidates)
            right_ear_y = sum(p[1] for p in right_candidates) / len(right_candidates)
            keypoints.extend([right_ear_x, right_ear_y, 1])  # Approximate
        else:
            keypoints.extend([0, 0, 0])

        return keypoints

    def _load_stanford_extra(self, json_path):
        """Load StanfordExtra keypoints dataset."""
        if not os.path.exists(json_path):
            return []

        with open(json_path, 'r') as f:
            data = json.load(f)

        images = []
        annotations = []

        for entry in data:
            if entry.get('is_multiple_dogs', False):
                continue  # Skip images with multiple dogs for now

            # Create COCO image entry
            image_entry = {
                'id': self.image_id,
                'file_name': entry['img_path'],
                'width': entry['img_width'],
                'height': entry['img_height']
            }
            images.append(image_entry)

            # Create COCO annotation
            bbox = entry['img_bbox']  # [x, y, w, h]
            keypoints = self._map_stanford_keypoints(entry.get('joints', []))

            annotation = {
                'id': self.annotation_id,
                'image_id': self.image_id,
                'category_id': 1,  # dog
                'bbox': bbox,
                'keypoints': keypoints,
                'num_keypoints': sum(1 for i in range(2, 10, 3) if keypoints[i] > 0),
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            }

            annotations.append(annotation)
            self.image_id += 1
            self.annotation_id += 1

        return images, annotations

    def _load_coco_bbox_only(self, json_path):
        """Load COCO dataset for bbox-only annotations (set keypoints to zeros)."""
        if not os.path.exists(json_path):
            return [], []

        with open(json_path, 'r') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = []

        # Update image IDs to be unique
        for img in images:
            img['id'] = self.image_id
            self.image_id += 1

        for ann in data.get('annotations', []):
            ann['id'] = self.annotation_id
            ann['keypoints'] = [0] * 15  # 5 keypoints * 3 values each
            ann['num_keypoints'] = 0
            ann['category_id'] = 1  # dog
            annotations.append(ann)
            self.annotation_id += 1

        return images, annotations

    def _synthesize_from_masks(self, mask_dir):
        """Synthesize keypoints from segmentation masks (Oxford Pets style)."""
        if not os.path.exists(mask_dir):
            return [], []

        images = []
        annotations = []

        # Get all PNG masks
        mask_files = Path(mask_dir).glob("*.png")
        for mask_path in mask_files:
            # Assuming mask filename gives original image path
            img_filename = mask_path.stem.replace('_', '.') + '.jpg'
            img_path = f"oxford_pets/images/{img_filename}"

            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            height, width = mask.shape

            # Find head centroid (weak supervision)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Use largest contour (dog body)
            main_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(main_contour)

            if moments['m00'] == 0:
                continue

            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            # Synthesize keypoints based on head centroid
            # Nose: slightly in front of centroid
            nose_x, nose_y = center_x, center_y - 30
            # Eyes: offset horizontally from centroid
            left_eye_x, left_eye_y = center_x - 20, center_y - 20
            right_eye_x, right_eye_y = center_x + 20, center_y - 20
            # Ears: higher and outer
            left_ear_x, left_ear_y = center_x - 35, center_y - 40
            right_ear_x, right_ear_y = center_x + 35, center_y - 40

            # Create image entry
            image_entry = {
                'id': self.image_id,
                'file_name': img_path,
                'width': width,
                'height': height
            }
            images.append(image_entry)

            # Create bbox from contour
            x, y, w, h = cv2.boundingRect(main_contour)
            bbox = [x, y, w, h]

            keypoints = [
                left_eye_x, left_eye_y, 1,    # left eye
                right_eye_x, right_eye_y, 1,  # right eye
                nose_x, nose_y, 1,           # nose
                left_ear_x, left_ear_y, 1,    # left ear
                right_ear_x, right_ear_y, 1   # right ear
            ]

            annotation = {
                'id': self.annotation_id,
                'image_id': self.image_id,
                'category_id': 1,  # dog
                'bbox': bbox,
                'keypoints': keypoints,
                'num_keypoints': 5,
                'area': w * h,
                'iscrowd': 0
            }

            annotations.append(annotation)
            self.image_id += 1
            self.annotation_id += 1

        return images, annotations

    def convert(self):
        """Main conversion function."""
        all_images = []
        all_annotations = []

        # Load StanfordExtra keypoints
        if os.path.exists(self.config.get('stanford_json', '')):
            print("Loading StanfordExtra keypoints...")
            images, annotations = self._load_stanford_extra(self.config['stanford_json'])
            all_images.extend(images)
            all_annotations.extend(annotations)
            print(f"Loaded {len(images)} StanfordExtra images")

        # Load COCO bbox-only (set keypoints to zeros)
        if os.path.exists(self.config.get('coco_json', '')):
            print("Loading COCO bbox-only...")
            images, annotations = self._load_coco_bbox_only(self.config['coco_json'])
            all_images.extend(images)
            all_annotations.extend(annotations)
            print(f"Loaded {len(images)} COCO bbox-only images")

        # Synthesize from Oxford masks
        if os.path.exists(self.config.get('oxford_mask_dir', '')):
            print("Synthesizing from Oxford masks...")
            images, annotations = self._synthesize_from_masks(self.config['oxford_mask_dir'])
            all_images.extend(images)
            all_annotations.extend(annotations)
            print(f"Loaded {len(images)} Oxford mask-synthesized images")

        # Split train/val (90/10)
        random.seed(42)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.9)

        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        # Filter annotations for train/val
        train_annotations = [ann for ann in all_annotations if ann['image_id'] in {img['id'] for img in train_images}]
        val_annotations = [ann for ann in all_annotations if ann['image_id'] in {img['id'] for img in val_images}]

        # Create output files
        train_coco = self.coco_template.copy()
        train_coco['images'] = train_images
        train_coco['annotations'] = train_annotations

        val_coco = self.coco_template.copy()
        val_coco['images'] = val_images
        val_coco['annotations'] = val_annotations

        # Save
        train_path = self.output_dir / 'annotations_train.json'
        val_path = self.output_dir / 'annotations_val.json'

        with open(train_path, 'w') as f:
            json.dump(train_coco, f, indent=2)
        with open(val_path, 'w') as f:
            json.dump(val_coco, f, indent=2)

        # Generate debug outputs for first 20 images
        self._generate_debug_outputs(train_images[:20], train_annotations, "train")
        self._generate_debug_outputs(val_images[:min(20, len(val_images))], val_annotations, "val")

        print("Conversion complete!")
        print(f"Train set: {len(train_images)} images, {len(train_annotations)} annotations")
        print(f"Val set: {len(val_images)} images, {len(val_annotations)} annotations")
        print(f"Files saved to: {train_path}, {val_path}")

        # Return stats
        total_with_keypoints = len([ann for ann in all_annotations if ann['num_keypoints'] > 0])
        total_synthesized = len([ann for ann in all_annotations if ann['num_keypoints'] == 5])
        total_bbox_only = len([ann for ann in all_annotations if ann['num_keypoints'] == 0])

        return {
            'total_images': len(all_images),
            'total_annotations': len(all_annotations),
            'with_full_keypoints': total_with_keypoints - total_synthesized,
            'with_synthesized_keypoints': total_synthesized,
            'with_bbox_only': total_bbox_only
        }

    def _generate_debug_outputs(self, images, annotations, split_type):
        """Generate debug JSON for first 20 images."""
        debug_data = []

        for img in images[:20]:
            img_anns = [ann for ann in annotations if ann['image_id'] == img['id']]
            debug_entry = {
                'image_meta': img,
                'annotations': img_anns
            }
            debug_data.append(debug_entry)

        debug_path = self.debug_dir / f'debug_{split_type}_first_20.json'
        with open(debug_path, 'w') as f:
            json.dump(debug_data, f, indent=2)


def main():
    """Main entry point."""
    config = {
        'stanford_json': 'data/stanford_dogs/stanford_extra_keypoints.json',
        'coco_json': 'data/coco/trainval.json',  # Add if available
        'oxford_mask_dir': 'data/oxford_pets/annotations/trimaps',
        'dogfacenet_crop_dir': 'data/dogfacenet/DogFaceNet_224resized',
        'output_dir': 'data/coco_keypoints'
    }

    converter = CocoKeypointsConverter(config)
    stats = converter.convert()

    print("Conversion Statistics:")
    print(f"Total images converted: {stats['total_images']}")
    print(f"Images with full keypoints: {stats['with_full_keypoints']}")
    print(f"Images with synthesized keypoints: {stats['with_synthesized_keypoints']}")
    print(f"Images with bbox-only: {stats['with_bbox_only']}")


if __name__ == '__main__':
    main()