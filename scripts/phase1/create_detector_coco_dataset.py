#!/usr/bin/env python3
"""
Creates a COCO-style dataset for object detection (bounding boxes only).

This script processes various source datasets (Stanford, COCO, Oxford) and
creates a unified dataset in COCO format, with a robust train/validation split,
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

        if self.output_dir.exists():
            print(f"Clearing existing COCO detector directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def _load_stanford_extra(self, json_path):
        """Loads images and annotations from StanfordExtra JSON."""
        if not os.path.exists(json_path):
            return [], []
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        images, annotations = [], []
        processed_files = set()
        for i, entry in enumerate(data):
            if entry.get('is_multiple_dogs', False):
                continue
            
            relative_path = f"stanford_dogs/images/{entry['img_path']}"
            if relative_path in processed_files:
                continue
            processed_files.add(relative_path)

            temp_id = i
            image_entry = {
                'id': temp_id, 'file_name': relative_path,
                'width': entry['img_width'], 'height': entry['img_height']
            }
            bbox = entry['img_bbox']
            annotation = {
                'id': -1, 'image_id': temp_id, 'category_id': 1, 'bbox': bbox,
                'area': bbox[2] * bbox[3], 'iscrowd': 0
            }
            images.append(image_entry)
            annotations.append(annotation)
        return images, annotations

    def _load_stanford_base_bboxes(self, base_dir, existing_files):
        """Loads images and annotations from Stanford Dogs base XMLs."""
        annotation_dir = Path(base_dir) / 'annotation'
        if not annotation_dir.exists():
            return [], []
        
        images, annotations = [], []
        xml_files = [p for p in annotation_dir.glob('*/*') if p.is_file()]
        for i, xml_path in enumerate(tqdm(xml_files, desc="Processing Stanford Base")):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename_stem = root.find('filename').text
                if '%s' in filename_stem: continue

                relative_path = f"stanford_dogs/images/{xml_path.parent.name}/{filename_stem}.jpg"
                if relative_path in existing_files: continue
                existing_files.add(relative_path)

                size_node = root.find('size')
                width = int(size_node.find('width').text)
                height = int(size_node.find('height').text)
                
                temp_id = i
                image_entry = {'id': temp_id, 'file_name': relative_path, 'width': width, 'height': height}
                
                current_annotations = []
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin, ymin, xmax, ymax = [int(bndbox.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]
                    bbox = [max(0, xmin), max(0, ymin), min(width, xmax) - max(0, xmin), min(height, ymax) - max(0, ymin)]
                    if bbox[2] <= 0 or bbox[3] <= 0: continue
                    current_annotations.append({'id': -1, 'image_id': temp_id, 'category_id': 1, 'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0})
                
                if current_annotations:
                    images.append(image_entry)
                    annotations.extend(current_annotations)
            except (ET.ParseError, FileNotFoundError, AttributeError):
                continue
        return images, annotations

    def _load_coco_bbox_only(self, json_path, split_name, num_negatives):
        """Loads positive and negative samples from a COCO JSON split."""
        if not os.path.exists(json_path):
            return [], [], []

        with open(json_path, 'r') as f:
            data = json.load(f)

        all_images_map = {img['id']: img for img in data.get('images', [])}
        dog_cat_id = next((cat['id'] for cat in data.get('categories', []) if cat['name'] == 'dog'), -1)
        
        dog_image_ids = {ann['image_id'] for ann in data.get('annotations', []) if ann.get('category_id') == dog_cat_id}
        
        pos_images, pos_annotations = [], []
        annotations_by_image = {img_id: [] for img_id in dog_image_ids}
        for ann in data.get('annotations', []):
            if ann.get('category_id') == dog_cat_id:
                annotations_by_image[ann['image_id']].append(ann)

        for i, (img_id, anns) in enumerate(annotations_by_image.items()):
            if img_id not in all_images_map: continue
            img = all_images_map[img_id]
            temp_id = i
            pos_images.append({'id': temp_id, 'file_name': f"coco/images/{split_name}/{img['file_name']}", 'width': img['width'], 'height': img['height']})
            for ann in anns:
                bbox = ann['bbox']
                pos_annotations.append({'id': -1, 'image_id': temp_id, 'category_id': 1, 'bbox': bbox, 'area': ann['area'], 'iscrowd': ann['iscrowd']})

        neg_images_all = [img for img_id, img in all_images_map.items() if img_id not in dog_image_ids]
        random.shuffle(neg_images_all)
        neg_images_sampled = neg_images_all[:num_negatives]
        neg_images = [{'id': -1, 'file_name': f"coco/images/{split_name}/{img['file_name']}", 'width': img['width'], 'height': img['height']} for img in neg_images_sampled]
        
        return pos_images, pos_annotations, neg_images

    def _load_oxford_pets_negatives(self, xml_dir):
        """Loads non-dog images from Oxford-IIIT Pets."""
        if not os.path.exists(xml_dir):
            return [], []
        
        images = []
        for xml_path in tqdm(list(Path(xml_dir).glob("*.xml")), desc="Processing Oxford Negatives"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if root.find('object/name').text != 'dog':
                    img_filename = root.find('filename').text
                    size = root.find('size')
                    images.append({'id': -1, 'file_name': f"oxford_pets/images/{img_filename}", 'width': int(size.find('width').text), 'height': int(size.find('height').text)})
            except (ET.ParseError, AttributeError):
                continue
        return images, []

    def _assign_unified_ids(self, images, annotations):
        """Assigns new, sequential, and unique IDs to a finalized set of images and annotations."""
        image_id_map = {}
        # Use a real list for final images to preserve order
        final_images = []
        for i, img in enumerate(images):
            original_temp_id = img['id']
            new_id = i
            # Map the original ID to the new sequential ID
            image_id_map[original_temp_id] = new_id
            img['id'] = new_id
            final_images.append(img)

        final_annotations = []
        for ann in annotations:
            original_temp_image_id = ann['image_id']
            if original_temp_image_id in image_id_map:
                ann['image_id'] = image_id_map[original_temp_image_id]
                final_annotations.append(ann)
        
        # Re-ID annotations themselves to be unique
        for i, ann in enumerate(final_annotations):
            ann['id'] = i
            
        return final_images, final_annotations

    def convert(self):
        """Main conversion function."""
        train_images, train_annotations = [], []
        val_images, val_annotations = [], []

        # --- 1. Load and split Stanford & Oxford data ---
        datasets_to_split = []
        
        print("Loading StanfordExtra...")
        se_images, se_annotations = self._load_stanford_extra(self.config['stanford_json'])
        datasets_to_split.append(("StanfordExtra", se_images, se_annotations))
        
        processed_se_files = {img['file_name'] for img in se_images}
        print("Loading Stanford Base...")
        sb_images, sb_annotations = self._load_stanford_base_bboxes(self.config['stanford_base_dir'], processed_se_files)
        datasets_to_split.append(("Stanford Base", sb_images, sb_annotations))

        print("Loading Oxford Pets Negatives...")
        op_images, op_annotations = self._load_oxford_pets_negatives(self.config['oxford_xml_dir'])
        datasets_to_split.append(("Oxford Pets (Negatives)", op_images, op_annotations))

        print("\nSplitting non-COCO datasets...")
        for name, images, annotations in datasets_to_split:
            if not images: continue
            
            # Link annotations to images via a temporary ID for splitting
            for i, img in enumerate(images): img['id'] = i
            img_map = {img['file_name']: img['id'] for img in images}
            for ann in annotations: ann['image_id'] = img_map.get(ann.get('file_name'), -1)

            random.seed(42)
            random.shuffle(images)
            split_idx = int(len(images) * 0.9)
            
            d_train_imgs = images[:split_idx]
            d_val_imgs = images[split_idx:]
            
            train_ids = {img['id'] for img in d_train_imgs}
            val_ids = {img['id'] for img in d_val_imgs}

            d_train_anns = [ann for ann in annotations if ann['image_id'] in train_ids]
            d_val_anns = [ann for ann in annotations if ann['image_id'] in val_ids]
            
            train_images.extend(d_train_imgs)
            train_annotations.extend(d_train_anns)
            val_images.extend(d_val_imgs)
            val_annotations.extend(d_val_anns)
            print(f"Split {name}: {len(d_train_imgs)} train, {len(d_val_imgs)} val")

        # --- 2. Add COCO datasets to their respective splits ---
        print("\nLoading COCO datasets...")
        print("Loading COCO Train...")
        coco_train_pos_imgs, coco_train_pos_anns, coco_train_neg_imgs = self._load_coco_bbox_only(self.config['coco_train_json'], 'train2017', 15000)
        train_images.extend(coco_train_pos_imgs)
        train_images.extend(coco_train_neg_imgs)
        train_annotations.extend(coco_train_pos_anns)
        print(f"Added {len(coco_train_pos_imgs) + len(coco_train_neg_imgs)} images from COCO Train")

        print("Loading COCO Val...")
        coco_val_pos_imgs, coco_val_pos_anns, coco_val_neg_imgs = self._load_coco_bbox_only(self.config['coco_val_json'], 'val2017', 2000)
        val_images.extend(coco_val_pos_imgs)
        val_images.extend(coco_val_neg_imgs)
        val_annotations.extend(coco_val_pos_anns)
        print(f"Added {len(coco_val_pos_imgs) + len(coco_val_neg_imgs)} images from COCO Val")

        # --- 3. Assign unified IDs and save ---
        print("\nFinalizing datasets...")
        final_train_images, final_train_annotations = self._assign_unified_ids(train_images, train_annotations)
        final_val_images, final_val_annotations = self._assign_unified_ids(val_images, val_annotations)

        categories = [{'id': 1, 'name': 'dog', 'supercategory': 'animal'}]
        train_coco = {'images': final_train_images, 'annotations': final_train_annotations, 'categories': categories}
        val_coco = {'images': final_val_images, 'annotations': final_val_annotations, 'categories': categories}

        train_path = self.output_dir / 'annotations_train.json'
        val_path = self.output_dir / 'annotations_val.json'
        with open(train_path, 'w') as f: json.dump(train_coco, f, indent=2)
        with open(val_path, 'w') as f: json.dump(val_coco, f, indent=2)
        
        print(f"\nFinal training set: {len(final_train_images)} images, {len(final_train_annotations)} annotations.")
        print(f"Final validation set: {len(final_val_images)} images, {len(final_val_annotations)} annotations.")
        print(f"Saved train annotations to {train_path}")
        print(f"Saved val annotations to {val_path}")

def main():
    """Main entry point."""
    config = {
        'stanford_json': 'data/stanford_dogs/stanford_extra_keypoints.json',
        'stanford_base_dir': 'data/stanford_dogs',
        'coco_train_json': 'data/coco/annotations/instances_train2017.json',
        'coco_val_json': 'data/coco/annotations/instances_val2017.json',
        'oxford_xml_dir': 'data/oxford_pets/annotations/xmls',
        'output_dir': 'data/detector/coco'
    }
    converter = CocoDetectorDatasetConverter(config)
    converter.convert()

if __name__ == '__main__':
    main()