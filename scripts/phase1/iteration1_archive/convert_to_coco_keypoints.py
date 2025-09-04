#!/usr/bin/env python3
"""
Convert available keypoints/masks to COCO-style keypoints annotations for dog detection.

This script creates COCO-format annotations with bounding boxes and 4-point keypoints.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil

class CocoKeypointsConverter:
    """Convert various dog datasets to COCO keypoints format."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])

        # Clear existing directory to prevent stale data
        if self.output_dir.exists():
            print(f"Clearing existing COCO annotations directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        self.debug_dir = self.output_dir / 'debug_annotations'
        self.debug_dir.mkdir(parents=True)

        # COCO format structure
        self.coco_template = {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1,
                'name': 'dog',
                'keypoints': ['nose', 'chin', 'left_ear_base', 'right_ear_base'],
                'skeleton': []
            }]
        }

        self.image_id = 0
        self.annotation_id = 0

    def _map_stanford_keypoints(self, joints):
        """
        Map StanfordExtra 24-point joints to our 4-point keypoint schema.
        """
        keypoint_map = {
            'nose': 16,
            'chin': 17,
            'left_ear_base': 14,
            'right_ear_base': 15,
        }

        keypoints = []
        for name in ['nose', 'chin', 'left_ear_base', 'right_ear_base']:
            idx = keypoint_map[name]
            if len(joints) > idx and joints[idx][2] > 0:
                keypoints.extend(joints[idx])
            else:
                keypoints.extend([0, 0, 0])
        
        return keypoints

    def _load_stanford_extra(self, json_path):
        """Load StanfordExtra keypoints dataset."""
        if not os.path.exists(json_path):
            return [], []

        with open(json_path, 'r') as f:
            data = json.load(f)

        images = []
        annotations = []

        for entry in data:
            if entry.get('is_multiple_dogs', False):
                continue

            image_entry = {
                'id': self.image_id,
                'file_name': f"stanford_dogs/images/{entry['img_path']}",
                'width': entry['img_width'],
                'height': entry['img_height'],
                'source_dataset': 'stanford_dogs'
            }
            images.append(image_entry)

            bbox = entry['img_bbox']
            keypoints = self._map_stanford_keypoints(entry.get('joints', []))

            annotation = {
                'id': self.annotation_id,
                'image_id': self.image_id,
                'category_id': 1,
                'bbox': bbox,
                'keypoints': keypoints,
                'num_keypoints': sum(1 for i in range(2, 12, 3) if keypoints[i] > 0),
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            }

            annotations.append(annotation)
            self.image_id += 1
            self.annotation_id += 1

        return images, annotations

    def _load_coco_bbox_only(self, json_path, num_negatives=5000):
        """Load COCO dataset, including dog images and a sample of negative images."""
        if not os.path.exists(json_path):
            return [], []

        with open(json_path, 'r') as f:
            data = json.load(f)

        original_images = data.get('images', [])
        original_annotations = data.get('annotations', [])
        
        images = []
        annotations = []
        
        dog_category_id = 18

        dog_image_ids = {ann['image_id'] for ann in original_annotations if ann['category_id'] == dog_category_id}

        dog_images = [img for img in original_images if img['id'] in dog_image_ids]
        negative_images = [img for img in original_images if img['id'] not in dog_image_ids]

        random.shuffle(negative_images)
        sampled_negatives = negative_images[:num_negatives]

        image_id_map = {}
        for img in dog_images:
            old_id = img['id']
            new_id = self.image_id
            image_id_map[old_id] = new_id
            
            img['id'] = new_id
            img['file_name'] = f"coco/train2017/{img['file_name']}"
            img['source_dataset'] = 'coco'
            images.append(img)
            self.image_id += 1

        dog_annotations = [ann for ann in original_annotations if ann['category_id'] == dog_category_id]
        for ann in dog_annotations:
            old_image_id = ann['image_id']
            if old_image_id in image_id_map:
                ann['id'] = self.annotation_id
                ann['image_id'] = image_id_map[old_image_id]
                ann['keypoints'] = [0] * 12
                ann['num_keypoints'] = 0
                ann['category_id'] = 1
                annotations.append(ann)
                self.annotation_id += 1

        for img in sampled_negatives:
            img['id'] = self.image_id
            img['file_name'] = f"coco/train2017/{img['file_name']}"
            img['source_dataset'] = 'coco_negative'
            images.append(img)
            self.image_id += 1

        return images, annotations

    def _load_oxford_pets_negatives(self, xml_dir):
        """
        Loads non-dog images from the Oxford-IIIT Pets dataset as negative samples.

        It parses the XML annotations to identify the species. If the species
        is not 'dog', it is added to the dataset as a negative sample.
        Dog images from this dataset are ignored to avoid using their face-only boxes.
        """
        if not os.path.exists(xml_dir):
            return [], []

        images = []
        xml_files = list(Path(xml_dir).glob("*.xml"))

        for xml_path in tqdm(xml_files, desc="Processing Oxford annotations for negative samples"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Get species name from <object><name>
                species_name = root.find('object/name').text
                
                # If the object is not a dog, treat it as a negative sample.
                if species_name != 'dog':
                    img_filename = root.find('filename').text
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)

                    image_entry = {
                        'id': self.image_id,
                        'file_name': f"oxford_pets/images/{img_filename}",
                        'width': width,
                        'height': height,
                        'source_dataset': 'oxford_pets_negative'
                    }
                    images.append(image_entry)
                    self.image_id += 1
            except ET.ParseError:
                print(f"Warning: Could not parse XML file, skipping: {xml_path}")
                continue
        
        return images, [] # No annotations are created

    def convert(self):
        """Main conversion function."""
        all_images = []
        all_annotations = []

        if os.path.exists(self.config.get('stanford_json', '')):
            print("Loading StanfordExtra keypoints...")
            images, annotations = self._load_stanford_extra(self.config['stanford_json'])
            all_images.extend(images)
            all_annotations.extend(annotations)
            print(f"Loaded {len(images)} StanfordExtra images")

        if os.path.exists(self.config.get('coco_json', '')):
            print("Loading COCO bbox-only and negative samples...")
            images, annotations = self._load_coco_bbox_only(self.config['coco_json'])
            all_images.extend(images)
            all_annotations.extend(annotations)
            print(f"Loaded {len(images)} COCO images")

        if os.path.exists(self.config.get('oxford_xml_dir', '')):
            print("Loading Oxford Pets negative samples (cats)...")
            images, annotations = self._load_oxford_pets_negatives(self.config['oxford_xml_dir'])
            all_images.extend(images)
            all_annotations.extend(annotations)
            print(f"Loaded {len(images)} Oxford Pets negative samples.")

        random.seed(42)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.9)

        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        train_image_ids = {img['id'] for img in train_images}
        val_image_ids = {img['id'] for img in val_images}

        train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_image_ids]
        val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_image_ids]

        train_coco = self.coco_template.copy()
        train_coco['images'] = train_images
        train_coco['annotations'] = train_annotations

        val_coco = self.coco_template.copy()
        val_coco['images'] = val_images
        val_coco['annotations'] = val_annotations

        train_path = self.output_dir / 'annotations_train.json'
        val_path = self.output_dir / 'annotations_val.json'

        with open(train_path, 'w') as f:
            json.dump(train_coco, f, indent=2)
        with open(val_path, 'w') as f:
            json.dump(val_coco, f, indent=2)

        print(f"Saved train annotations to {train_path}")
        print(f"Saved val annotations to {val_path}")


def main():
    """Main entry point."""
    config = {
        'stanford_json': 'data/stanford_dogs/stanford_extra_keypoints.json',
        'coco_json': 'data/coco/annotations/instances_train2017.json',
        'oxford_xml_dir': 'data/oxford_pets/annotations/xmls',
        'output_dir': 'data/coco_keypoints'
    }

    converter = CocoKeypointsConverter(config)
    converter.convert()


if __name__ == '__main__':
    main()
