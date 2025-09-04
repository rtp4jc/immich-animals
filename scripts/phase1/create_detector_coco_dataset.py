#!/usr/bin/env python3
"""
Creates a COCO-style dataset for object detection (bounding boxes only).

This script processes various source datasets (Stanford, COCO, Oxford) and
creates a unified dataset in COCO format, without any keypoint information,
storing the output in the `data/detector/coco` directory.
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil

class CocoDetectorDatasetConverter:
    """Convert various dog datasets to COCO format for detection."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.image_id = 0
        self.annotation_id = 0
        self.image_filenames = set() # To track added images and prevent duplicates

        if self.output_dir.exists():
            print(f"Clearing existing COCO detector directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        self.coco_template = {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1,
                'name': 'dog',
                'supercategory': 'animal',
            }]
        }

    def _add_image_and_annotations(self, image_entry, annotation_entries):
        """Adds a new image and its annotations, avoiding duplicates."""
        if image_entry['file_name'] in self.image_filenames:
            return False # Skip if image already processed
        
        self.image_filenames.add(image_entry['file_name'])
        
        original_image_id = image_entry['id']
        new_image_id = self.image_id
        image_entry['id'] = new_image_id

        self.coco_template['images'].append(image_entry)

        for ann in annotation_entries:
            ann['id'] = self.annotation_id
            ann['image_id'] = new_image_id
            self.coco_template['annotations'].append(ann)
            self.annotation_id += 1
        
        self.image_id += 1
        return True

    def _load_stanford_extra(self, json_path):
        """Load StanfordExtra dataset for bounding boxes, prioritizing these versions."""
        if not os.path.exists(json_path):
            return 0

        with open(json_path, 'r') as f:
            data = json.load(f)

        count = 0
        for entry in data:
            if entry.get('is_multiple_dogs', False):
                continue

            relative_path = f"stanford_dogs/images/{entry['img_path']}"
            image_entry = {
                'id': -1, # Temporary ID
                'file_name': relative_path,
                'width': entry['img_width'],
                'height': entry['img_height'],
                'source_dataset': 'stanford_dogs'
            }

            bbox = entry['img_bbox']
            annotation = {
                'id': -1,
                'image_id': -1,
                'category_id': 1,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            }
            if self._add_image_and_annotations(image_entry, [annotation]):
                count += 1
        return count

    def _load_stanford_base_bboxes(self, base_dir):
        """Loads bboxes from base Stanford Dogs XMLs, skipping duplicates and fixing errors."""
        annotation_dir = Path(base_dir) / 'annotation'
        if not annotation_dir.exists():
            return 0

        count = 0
        xml_files = [p for p in annotation_dir.glob('*/*') if p.is_file()]

        for xml_path in tqdm(xml_files, desc="Processing Stanford Base BBoxes"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                filename_stem = root.find('filename').text
                if '%s' in filename_stem:
                    continue

                breed_dir = xml_path.parent.name
                relative_path = f"stanford_dogs/images/{breed_dir}/{filename_stem}.jpg"

                if relative_path in self.image_filenames:
                    continue

                size_node = root.find('size')
                width = int(size_node.find('width').text)
                height = int(size_node.find('height').text)

                image_entry = {
                    'id': -1,
                    'file_name': relative_path,
                    'width': width,
                    'height': height,
                    'source_dataset': 'stanford_dogs_base'
                }

                annotations = []
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)

                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue

                    annotations.append({
                        'id': -1, 'image_id': -1, 'category_id': 1,
                        'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0
                    })
                
                if annotations and self._add_image_and_annotations(image_entry, annotations):
                    count += 1
            except (ET.ParseError, FileNotFoundError, AttributeError):
                continue
        return count

    def _load_coco_bbox_only(self, json_path, num_negatives=15000):
        """Load COCO dataset, including dog images and an increased number of negative samples."""
        if not os.path.exists(json_path):
            return 0

        with open(json_path, 'r') as f:
            data = json.load(f)

        original_images = {img['id']: img for img in data.get('images', [])}
        original_annotations = data.get('annotations', [])
        dog_category_id = 18
        dog_image_ids = {ann['image_id'] for ann in original_annotations if ann['category_id'] == dog_category_id}

        count = 0
        dog_annotations = [ann for ann in original_annotations if ann['category_id'] == dog_category_id]
        
        annotations_by_image = {}
        for ann in dog_annotations:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            ann.pop('keypoints', None)
            ann.pop('num_keypoints', None)
            ann['category_id'] = 1
            annotations_by_image[img_id].append(ann)

        for img_id, anns in annotations_by_image.items():
            img = original_images[img_id]
            image_entry = {
                'id': -1, 'file_name': f"coco/train2017/{img['file_name']}",
                'width': img['width'], 'height': img['height'], 'source_dataset': 'coco'
            }
            if self._add_image_and_annotations(image_entry, anns):
                count += 1

        negative_images = [img for img_id, img in original_images.items() if img_id not in dog_image_ids]
        random.shuffle(negative_images)
        sampled_negatives = negative_images[:num_negatives]
        for img in sampled_negatives:
            image_entry = {
                'id': -1, 'file_name': f"coco/train2017/{img['file_name']}",
                'width': img['width'], 'height': img['height'], 'source_dataset': 'coco_negative'
            }
            if self._add_image_and_annotations(image_entry, []):
                count += 1
        return count

    def _load_oxford_pets_negatives(self, xml_dir):
        """Loads non-dog images from the Oxford-IIIT Pets dataset as negative samples."""
        if not os.path.exists(xml_dir):
            return 0
        count = 0
        xml_files = list(Path(xml_dir).glob("*.xml"))
        for xml_path in tqdm(xml_files, desc="Processing Oxford annotations for negative samples"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if root.find('object/name').text != 'dog':
                    img_filename = root.find('filename').text
                    size = root.find('size')
                    image_entry = {
                        'id': -1, 'file_name': f"oxford_pets/images/{img_filename}",
                        'width': int(size.find('width').text), 'height': int(size.find('height').text),
                        'source_dataset': 'oxford_pets_negative'
                    }
                    if self._add_image_and_annotations(image_entry, []):
                        count += 1
            except (ET.ParseError, AttributeError):
                continue
        return count

    def convert(self):
        """Main conversion function."""
        print("Loading StanfordExtra for bounding boxes...")
        count_extra = self._load_stanford_extra(self.config['stanford_json'])
        print(f"Loaded {count_extra} unique StanfordExtra images")

        print("Loading base Stanford Dogs bboxes (skipping duplicates)...")
        count_base = self._load_stanford_base_bboxes(self.config['stanford_base_dir'])
        print(f"Loaded {count_base} new images from Stanford Dogs base")

        print("Loading COCO for bounding boxes and negative samples...")
        count_coco = self._load_coco_bbox_only(self.config['coco_json'])
        print(f"Loaded {count_coco} COCO images (positives and negatives)")

        print("Loading Oxford Pets negative samples (non-dogs)...")
        count_oxford = self._load_oxford_pets_negatives(self.config['oxford_xml_dir'])
        print(f"Loaded {count_oxford} Oxford Pets negative samples.")

        all_images = self.coco_template['images']
        random.seed(42)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.9)

        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        train_image_ids = {img['id'] for img in train_images}
        val_image_ids = {img['id'] for img in val_images}

        all_annotations = self.coco_template['annotations']
        train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_image_ids]
        val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_image_ids]

        train_coco = {'images': train_images, 'annotations': train_annotations, 'categories': self.coco_template['categories']}
        val_coco = {'images': val_images, 'annotations': val_annotations, 'categories': self.coco_template['categories']}

        train_path = self.output_dir / 'annotations_train.json'
        val_path = self.output_dir / 'annotations_val.json'

        with open(train_path, 'w') as f: json.dump(train_coco, f, indent=2)
        with open(val_path, 'w') as f: json.dump(val_coco, f, indent=2)

        print(f"Saved train annotations to {train_path}")
        print(f"Saved val annotations to {val_path}")

def main():
    """Main entry point."""
    config = {
        'stanford_json': 'data/stanford_dogs/stanford_extra_keypoints.json',
        'stanford_base_dir': 'data/stanford_dogs',
        'coco_json': 'data/coco/annotations/instances_train2017.json',
        'oxford_xml_dir': 'data/oxford_pets/annotations/xmls',
        'output_dir': 'data/detector/coco'
    }
    converter = CocoDetectorDatasetConverter(config)
    converter.convert()

if __name__ == '__main__':
    main()
