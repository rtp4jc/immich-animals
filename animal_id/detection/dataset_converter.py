"""
Dataset converter for creating COCO format detection datasets.

Processes various source datasets (Stanford, COCO, Oxford) and creates a unified
dataset in COCO format for dog detection training.
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil
from typing import Dict, List, Tuple, Set, Any, Optional


class CocoDetectorDatasetConverter:
    """Convert various dog datasets to COCO format for detection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize converter with configuration."""
        self.config = config
        self.output_dir = Path(config['output_dir'])

        if self.output_dir.exists():
            print(f"Clearing existing COCO detector directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def _load_stanford_base_bboxes(self, base_dir: str, existing_files: Set[str], 
                                 image_id_counter: int) -> Tuple[List[Dict], List[Dict], int]:
        """Loads images and annotations from Stanford Dogs base XMLs."""
        annotation_dir = Path(base_dir) / 'annotation'
        if not annotation_dir.exists():
            return [], [], image_id_counter
        
        images, annotations = [], []
        xml_files = [p for p in annotation_dir.glob('*/*') if p.is_file()]
        
        for xml_path in tqdm(xml_files, desc="Processing Stanford Base"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename_stem = root.find('filename').text
                if '%s' in filename_stem: 
                    continue

                relative_path = f"stanford_dogs/images/{xml_path.parent.name}/{filename_stem}.jpg"
                if relative_path in existing_files: 
                    continue
                existing_files.add(relative_path)

                size_node = root.find('size')
                width = int(size_node.find('width').text)
                height = int(size_node.find('height').text)
                
                temp_id = image_id_counter
                image_id_counter += 1
                image_entry = {
                    'id': temp_id, 
                    'file_name': relative_path, 
                    'width': width, 
                    'height': height
                }
                
                current_annotations = []
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    x1 = max(0.0, xmin)
                    y1 = max(0.0, ymin)
                    x2 = min(float(width), xmax)
                    y2 = min(float(height), ymax)

                    bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
                    if bbox[2] <= 0 or bbox[3] <= 0: 
                        continue
                        
                    current_annotations.append({
                        'id': -1, 
                        'image_id': temp_id, 
                        'category_id': 1, 
                        'bbox': bbox, 
                        'area': bbox[2] * bbox[3], 
                        'iscrowd': 0
                    })
                
                if current_annotations:
                    images.append(image_entry)
                    annotations.extend(current_annotations)
                    
            except (ET.ParseError, FileNotFoundError, AttributeError):
                continue
                
        return images, annotations, image_id_counter

    def _load_coco_bbox_only(self, json_path: str, split_name: str, num_negatives: int, 
                           image_id_counter: int) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
        """Loads positive and negative samples from a COCO JSON split."""
        if not os.path.exists(json_path):
            return [], [], [], image_id_counter

        with open(json_path, 'r') as f:
            data = json.load(f)

        all_images_map = {img['id']: img for img in data.get('images', [])}
        dog_cat_id = next((cat['id'] for cat in data.get('categories', []) if cat['name'] == 'dog'), -1)
        
        dog_annotations = [ann for ann in data.get('annotations', []) if ann.get('category_id') == dog_cat_id]
        dog_image_ids = {ann['image_id'] for ann in dog_annotations}

        pos_images, pos_annotations = [], []
        
        annotations_by_image = {img_id: [] for img_id in dog_image_ids}
        for ann in dog_annotations:
            annotations_by_image[ann['image_id']].append(ann)

        for img_id, anns in annotations_by_image.items():
            if img_id not in all_images_map: 
                continue
            img = all_images_map[img_id]
            width, height = img['width'], img['height']
            temp_id = image_id_counter
            image_id_counter += 1
            pos_images.append({
                'id': temp_id, 
                'file_name': f"coco/images/{split_name}/{img['file_name']}", 
                'width': width, 
                'height': height
            })
            
            for ann in anns:
                x, y, w, h = [float(c) for c in ann['bbox']]
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = min(float(width), x + w)
                y2 = min(float(height), y + h)
                bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
                if bbox[2] <= 0 or bbox[3] <= 0: 
                    continue
                pos_annotations.append({
                    'id': -1, 
                    'image_id': temp_id, 
                    'category_id': 1, 
                    'bbox': bbox, 
                    'area': bbox[2] * bbox[3], 
                    'iscrowd': ann['iscrowd']
                })

        neg_images_all = [img for img_id, img in all_images_map.items() if img_id not in dog_image_ids]
        random.shuffle(neg_images_all)
        neg_images_sampled = neg_images_all[:num_negatives]
        for img in neg_images_sampled:
            img['id'] = image_id_counter
            image_id_counter += 1
        neg_images = [{
            'id': img['id'], 
            'file_name': f"coco/images/{split_name}/{img['file_name']}", 
            'width': img['width'], 
            'height': img['height']
        } for img in neg_images_sampled]
        
        return pos_images, pos_annotations, neg_images, image_id_counter

    def _load_oxford_pets_negatives(self, xml_dir: str, image_id_counter: int) -> Tuple[List[Dict], List[Dict], int]:
        """Loads non-dog images from Oxford-IIIT Pets."""
        if not os.path.exists(xml_dir):
            return [], [], image_id_counter
        
        images = []
        for xml_path in tqdm(list(Path(xml_dir).glob("*.xml")), desc="Processing Oxford Negatives"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if root.find('object/name').text != 'dog':
                    img_filename = root.find('filename').text
                    size = root.find('size')
                    temp_id = image_id_counter
                    image_id_counter += 1
                    images.append({
                        'id': temp_id, 
                        'file_name': f"oxford_pets/images/{img_filename}", 
                        'width': int(size.find('width').text), 
                        'height': int(size.find('height').text)
                    })
            except (ET.ParseError, AttributeError):
                continue
        return images, [], image_id_counter

    def _finalize_annotations(self, annotations: List[Dict], annotation_id_counter: int) -> Tuple[List[Dict], int]:
        """Assign final annotation IDs."""
        for ann in annotations:
            ann['id'] = annotation_id_counter
            annotation_id_counter += 1
        return annotations, annotation_id_counter

    def convert(self) -> None:
        """Main conversion function."""
        train_images, train_annotations = [], []
        val_images, val_annotations = [], []
        image_id_counter = 0
        annotation_id_counter = 0

        # --- 1. Load and split Stanford & Oxford data ---
        datasets_to_split = []
        
        print("Loading Stanford Base...")
        sb_images, sb_annotations, image_id_counter = self._load_stanford_base_bboxes(
            self.config['stanford_base_dir'], set(), image_id_counter)
        datasets_to_split.append(("Stanford Base", sb_images, sb_annotations))

        print("Loading Oxford Pets Negatives...")
        op_images, op_annotations, image_id_counter = self._load_oxford_pets_negatives(
            self.config['oxford_xml_dir'], image_id_counter)
        datasets_to_split.append(("Oxford Pets (Negatives)", op_images, op_annotations))

        print("\nSplitting non-COCO datasets...")
        for name, images, annotations in datasets_to_split:
            if not images: 
                continue
            
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
        coco_train_pos_imgs, coco_train_pos_anns, coco_train_neg_imgs, image_id_counter = self._load_coco_bbox_only(
            self.config['coco_train_json'], 'train2017', 15000, image_id_counter)
        train_images.extend(coco_train_pos_imgs)
        train_images.extend(coco_train_neg_imgs)
        train_annotations.extend(coco_train_pos_anns)
        print(f"Added {len(coco_train_pos_imgs) + len(coco_train_neg_imgs)} images from COCO Train")

        print("Loading COCO Val...")
        coco_val_pos_imgs, coco_val_pos_anns, coco_val_neg_imgs, image_id_counter = self._load_coco_bbox_only(
            self.config['coco_val_json'], 'val2017', 2000, image_id_counter)
        val_images.extend(coco_val_pos_imgs)
        val_images.extend(coco_val_neg_imgs)
        val_annotations.extend(coco_val_pos_anns)
        print(f"Added {len(coco_val_pos_imgs) + len(coco_val_pos_anns)} images from COCO Val")

        # --- 3. Finalize annotation IDs and save ---
        print("\nFinalizing datasets...")
        train_annotations, annotation_id_counter = self._finalize_annotations(train_annotations, annotation_id_counter)
        val_annotations, annotation_id_counter = self._finalize_annotations(val_annotations, annotation_id_counter)

        categories = [{'id': 1, 'name': 'dog', 'supercategory': 'animal'}]
        train_coco = {'images': train_images, 'annotations': train_annotations, 'categories': categories}
        val_coco = {'images': val_images, 'annotations': val_annotations, 'categories': categories}

        train_path = self.output_dir / 'annotations_train.json'
        val_path = self.output_dir / 'annotations_val.json'
        with open(train_path, 'w') as f: 
            json.dump(train_coco, f, indent=2)
        with open(val_path, 'w') as f: 
            json.dump(val_coco, f, indent=2)
        
        print(f"\nFinal training set: {len(train_images)} images, {len(train_annotations)} annotations.")
        print(f"Final validation set: {len(val_images)} images, {len(val_annotations)} annotations.")
        print(f"Saved train annotations to {train_path}")
        print(f"Saved val annotations to {val_path}")


def create_default_config() -> Dict[str, str]:
    """Create default configuration for dataset conversion."""
    return {
        'stanford_base_dir': 'data/stanford_dogs',
        'coco_train_json': 'data/coco/annotations/instances_train2017.json',
        'coco_val_json': 'data/coco/annotations/instances_val2017.json',
        'oxford_xml_dir': 'data/oxford_pets/annotations/xmls',
        'output_dir': 'data/detector/coco'
    }
