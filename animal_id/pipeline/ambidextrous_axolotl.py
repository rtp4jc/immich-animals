"""
AmbidextrousAxolotl: Advanced animal identification pipeline system.

Combines detection, keypoint, and embedding models for robust animal identification.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from .models import DetectionModel, KeypointModel, EmbeddingModel, AnimalClass
from ..benchmark.evaluator import AnimalIdentificationSystem


class AmbidextrousAxolotl(AnimalIdentificationSystem):
    """Advanced animal identification pipeline with injectable models."""
    
    def __init__(self, 
                 detector: DetectionModel,
                 embedding_model: EmbeddingModel,
                 keypoint_model: Optional[KeypointModel] = None,
                 target_class: AnimalClass = AnimalClass.DOG,
                 detection_threshold: float = 0.5,
                 keypoint_threshold: float = 0.3,
                 use_keypoints: bool = False):
        """
        Initialize AmbidextrousAxolotl pipeline.
        
        Args:
            detector: Detection model instance
            embedding_model: Embedding model instance
            keypoint_model: Keypoint model instance (optional)
            target_class: Animal class to identify
            detection_threshold: Minimum confidence for detections
            keypoint_threshold: Minimum confidence for keypoints
            use_keypoints: Whether to use keypoint refinement. Currently defaulting to 
                false because benchmarks show the keypoints reduce key metrics like top-5 
                accuracy with current data quality
        """
        self.detector = detector
        self.keypoint_model = keypoint_model
        self.embedding_model = embedding_model
        self.target_class = target_class
        self.detection_threshold = detection_threshold
        self.keypoint_threshold = keypoint_threshold
        self.use_keypoints = use_keypoints
        
        self.gallery_embeddings = None
        self.gallery_paths = None
    
    def build_gallery(self, image_paths: List[str]) -> None:
        """Pre-compute embeddings for all gallery images."""
        embeddings = []
        valid_paths = []
        
        for image_path in image_paths:
            embedding = self.generate_embedding(image_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_paths.append(image_path)
        
        if embeddings:
            self.gallery_embeddings = np.array(embeddings)
            self.gallery_paths = valid_paths
        else:
            self.gallery_embeddings = None
            self.gallery_paths = None
    
    def predict(self, image_path: str) -> Tuple[bool, List[Tuple[str, float]]]:
        """Predict if image contains target animal and return similar images."""
        embedding = self.generate_embedding(image_path)
        
        if embedding is None:
            return False, []
        
        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            return True, []
        
        # Compute similarities
        similarities = cosine_similarity([embedding], self.gallery_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Return top matches (excluding self)
        similar_images = []
        for idx in sorted_indices:
            gallery_path = self.gallery_paths[idx]
            if gallery_path != image_path:
                similar_images.append((gallery_path, float(similarities[idx])))
        
        return True, similar_images
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate embedding for image using full pipeline.
        
        This is the source of truth for the complete evaluation logic.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector or None if no target animal detected
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
        
        # Stage 1: Detection
        detections = self.detector.predict(image)
        if not detections:
            return None
        
        # Filter by target class and confidence
        valid_detections = [
            d for d in detections 
            if d.get('class', self.target_class) == self.target_class 
            and d['confidence'] >= self.detection_threshold
        ]
        
        if not valid_detections:
            return None
        
        # Use highest confidence detection
        best_detection = max(valid_detections, key=lambda d: d['confidence'])
        
        # Crop with padding
        x1, y1, x2, y2 = best_detection['bbox']
        h, w = image.shape[:2]
        padding = int((x2 - x1) * 0.1)
        crop_x1, crop_y1 = max(0, x1 - padding), max(0, y1 - padding)
        crop_x2, crop_y2 = min(w, x2 + padding), min(h, y2 + padding)
        detector_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if detector_crop.size == 0:
            return None
        
        # Stage 2: Keypoint refinement (optional)
        final_crop = detector_crop
        if self.use_keypoints and self.keypoint_model:
            keypoint_detections = self.keypoint_model.predict(detector_crop)
            
            if keypoint_detections:
                best_keypoint = max(keypoint_detections, key=lambda k: k['confidence'])
                keypoints = np.array(best_keypoint['keypoints'])
                
                if len(keypoints) > 0 and np.all(keypoints[:, 2] > self.keypoint_threshold):
                    # Refine crop using keypoint bounding box
                    kx1, ky1 = np.min(keypoints[:, :2], axis=0).astype(int)
                    kx2, ky2 = np.max(keypoints[:, :2], axis=0).astype(int)
                    kp_padding = int((kx2 - kx1) * 0.2)
                    
                    ch, cw = detector_crop.shape[:2]
                    final_crop_x1 = max(0, kx1 - kp_padding)
                    final_crop_y1 = max(0, ky1 - kp_padding)
                    final_crop_x2 = min(cw, kx2 + kp_padding)
                    final_crop_y2 = min(ch, ky2 + kp_padding)
                    
                    final_crop = detector_crop[final_crop_y1:final_crop_y2, final_crop_x1:final_crop_x2]
        
        if final_crop.size == 0:
            return None
        
        # Stage 3: Embedding generation
        return self.embedding_model.predict(final_crop)
    
    def classify_image(self, image_path: str) -> AnimalClass:
        """
        Classify image and return detected animal class.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Detected animal class
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return AnimalClass.NO_ANIMAL
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return AnimalClass.NO_ANIMAL
        
        detections = self.detector.predict(image)
        if not detections:
            return AnimalClass.NO_ANIMAL
        
        # Return class of highest confidence detection
        best_detection = max(detections, key=lambda d: d['confidence'])
        if best_detection['confidence'] >= self.detection_threshold:
            return best_detection.get('class', AnimalClass.OTHER)
        
        return AnimalClass.NO_ANIMAL
