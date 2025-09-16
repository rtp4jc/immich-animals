"""
Benchmark system for evaluating animal detection and similarity matching accuracy.

Evaluates a system that takes images with identity labels and predicts:
1. Whether there is a target animal present
2. An ordered list of most similar animal images

Provides overall scoring based on detection accuracy and identity matching.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Protocol
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from ..pipeline.models import AnimalClass


@dataclass
class EvaluationResult:
    """Results from evaluating a single image."""
    image_path: str
    has_animal_gt: bool
    has_animal_pred: bool
    identity_gt: Optional[str]
    similar_images: List[Tuple[str, float]]  # (image_path, similarity_score)
    detection_correct: bool
    identity_rank: Optional[int]  # Rank of correct identity in results (1-based)


@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics."""
    detection_accuracy: float
    detection_precision: float
    detection_recall: float
    mean_reciprocal_rank: float
    top_k_accuracy: Dict[int, float]
    tar_at_far: Dict[float, Tuple[float, float]]  # FAR -> (TAR, threshold)
    total_images: int
    animal_images: int
    non_animal_images: int
    
    def __str__(self) -> str:
        """Format metrics for display."""
        lines = [
            f"Detection Accuracy:     {self.detection_accuracy:.3f} ({self.detection_accuracy*100:.1f}%)",
            f"Detection Precision:    {self.detection_precision:.3f}",
            f"Detection Recall:       {self.detection_recall:.3f}",
            f"Mean Reciprocal Rank:   {self.mean_reciprocal_rank:.3f}",
            f"Top-1 Accuracy:         {self.top_k_accuracy[1]:.3f} ({self.top_k_accuracy[1]*100:.1f}%)",
            f"Top-3 Accuracy:         {self.top_k_accuracy[3]:.3f} ({self.top_k_accuracy[3]*100:.1f}%)",
            f"Top-5 Accuracy:         {self.top_k_accuracy[5]:.3f} ({self.top_k_accuracy[5]*100:.1f}%)",
        ]
        
        # Add TAR @ FAR metrics
        if self.tar_at_far:
            lines.append("TAR @ FAR:")
            for far, (tar, threshold) in sorted(self.tar_at_far.items()):
                lines.append(f"  FAR={far*100:5.1f}%: TAR={tar:.3f} ({tar*100:.1f}%) @ threshold={threshold:.4f}")
        
        lines.append(f"Dataset: {self.animal_images} animals, {self.non_animal_images} non-animals ({self.total_images} total)")
        return "\n".join(lines)


class AnimalIdentificationSystem(Protocol):
    """Protocol for animal identification systems to be benchmarked."""
    
    def build_gallery(self, image_paths: List[str]) -> None:
        """
        Pre-compute embeddings for all gallery images.
        
        Args:
            image_paths: List of all image paths in the evaluation set
        """
        ...
    
    def predict(self, image_path: str) -> Tuple[bool, List[Tuple[str, float]]]:
        """
        Predict if image contains target animal and return similar images.
        
        Args:
            image_path: Path to query image
            
        Returns:
            has_animal: Whether target animal was detected
            similar_images: List of (image_path, similarity_score) ordered by similarity
        """
        ...


# Backward compatibility alias
DogIdentificationSystem = AnimalIdentificationSystem


class BenchmarkEvaluator:
    """Evaluates dog identification systems against ground truth data."""
    
    def __init__(self, ground_truth_path: str, data_root: str):
        """
        Initialize evaluator with ground truth data.
        
        Args:
            ground_truth_path: Path to JSON file with ground truth labels
            data_root: Root directory containing images
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.data_root = Path(data_root)
        self.ground_truth = self._load_ground_truth()
        self.results: List[EvaluationResult] = []
        
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground truth data from JSON file."""
        with open(self.ground_truth_path, 'r') as f:
            return json.load(f)
    
    def evaluate(self, system: DogIdentificationSystem) -> BenchmarkMetrics:
        """
        Evaluate system against ground truth data.
        
        Args:
            system: Dog identification system to evaluate
            
        Returns:
            BenchmarkMetrics with evaluation results
        """
        # Build gallery with all image paths
        all_image_paths = [str(self.data_root / item['image_path']) for item in self.ground_truth]
        system.build_gallery(all_image_paths)
        
        self.results = []
        
        # Group ground truth by identity for similarity evaluation
        identity_groups = defaultdict(list)
        for item in self.ground_truth:
            if item.get('identity_label'):
                identity_groups[item['identity_label']].append(item['image_path'])
        
        for item in tqdm(self.ground_truth, desc="Evaluating images"):
            image_path = str(self.data_root / item['image_path'])
            has_animal_gt = item.get('identity_label') is not None
            identity_gt = item.get('identity_label')
            
            # Get system predictions
            has_animal_pred, similar_images = system.predict(image_path)
            
            # Evaluate detection accuracy
            detection_correct = has_animal_gt == has_animal_pred
            
            # Evaluate identity ranking (only for images with animals)
            identity_rank = None
            if has_animal_gt and has_animal_pred and identity_gt:
                same_identity_images = set(identity_groups[identity_gt])
                for rank, (sim_path, _) in enumerate(similar_images, 1):
                    sim_rel_path = str(Path(sim_path).relative_to(self.data_root))
                    if sim_rel_path in same_identity_images and sim_rel_path != item['image_path']:
                        identity_rank = rank
                        break
            
            result = EvaluationResult(
                image_path=item['image_path'],
                has_animal_gt=has_animal_gt,
                has_animal_pred=has_animal_pred,
                identity_gt=identity_gt,
                similar_images=similar_images,
                detection_correct=detection_correct,
                identity_rank=identity_rank
            )
            self.results.append(result)
        
        return self._compute_metrics()
    
    def _compute_metrics(self) -> BenchmarkMetrics:
        """Compute benchmark metrics from evaluation results."""
        total_images = len(self.results)
        animal_images = sum(1 for r in self.results if r.has_animal_gt)
        non_animal_images = total_images - animal_images
        
        # Detection metrics
        detection_correct = sum(1 for r in self.results if r.detection_correct)
        detection_accuracy = detection_correct / total_images if total_images > 0 else 0.0
        
        true_positives = sum(1 for r in self.results if r.has_animal_gt and r.has_animal_pred)
        false_positives = sum(1 for r in self.results if not r.has_animal_gt and r.has_animal_pred)
        false_negatives = sum(1 for r in self.results if r.has_animal_gt and not r.has_animal_pred)
        
        detection_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        detection_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # Identity ranking metrics
        valid_ranks = [r.identity_rank for r in self.results if r.identity_rank is not None]
        mean_reciprocal_rank = np.mean([1.0 / rank for rank in valid_ranks]) if valid_ranks else 0.0
        
        # Top-k accuracy
        top_k_accuracy = {}
        for k in [1, 3, 5, 10]:
            correct_at_k = sum(1 for rank in valid_ranks if rank <= k)
            top_k_accuracy[k] = correct_at_k / len(valid_ranks) if valid_ranks else 0.0
        
        # Compute TAR @ FAR metrics
        tar_at_far = self._compute_tar_at_far()
        
        return BenchmarkMetrics(
            detection_accuracy=detection_accuracy,
            detection_precision=detection_precision,
            detection_recall=detection_recall,
            mean_reciprocal_rank=mean_reciprocal_rank,
            top_k_accuracy=top_k_accuracy,
            tar_at_far=tar_at_far,
            total_images=total_images,
            animal_images=animal_images,
            non_animal_images=non_animal_images
        )
    
    def get_results(self) -> List[EvaluationResult]:
        """Get detailed evaluation results."""
        return self.results
    
    def _compute_tar_at_far(self) -> Dict[float, Tuple[float, float]]:
        """Compute TAR @ FAR metrics for identity verification using binary search."""
        same_identity_scores = []
        different_identity_scores = []
        
        # Get all detected images with identities
        detected_results = [r for r in self.results if r.has_animal_gt and r.has_animal_pred and r.identity_gt]
        
        if len(detected_results) < 2:
            return {}
        
        # Create identity mapping using absolute paths for consistent matching
        path_to_identity = {}
        for r in detected_results:
            abs_path = str(Path(r.image_path).resolve())
            path_to_identity[abs_path] = r.identity_gt
        
        # Collect similarity pairs
        for query_result in detected_results:
            query_identity = query_result.identity_gt
            
            for similar_path, similarity in query_result.similar_images:
                abs_similar_path = str(Path(similar_path).resolve())
                similar_identity = path_to_identity.get(abs_similar_path)
                
                if similar_identity is not None:
                    if similar_identity == query_identity:
                        same_identity_scores.append(similarity)
                    else:
                        different_identity_scores.append(similarity)
        
        if not same_identity_scores or not different_identity_scores:
            return {}
        
        # Convert to numpy arrays and sort for binary search
        same_scores = np.array(same_identity_scores)
        diff_scores = np.sort(np.array(different_identity_scores))
        
        # Compute TAR @ FAR for standard FAR values
        far_values = [0.001, 0.01, 0.1]
        tar_at_far = {}
        
        for target_far in far_values:
            threshold = self._binary_search_threshold(diff_scores, target_far)
            tar = np.mean(same_scores >= threshold)
            tar_at_far[target_far] = (tar, threshold)
        
        return tar_at_far
    
    def _binary_search_threshold(self, sorted_diff_scores: np.ndarray, target_far: float) -> float:
        """Binary search for threshold that achieves target FAR."""
        left, right = 0.0, 1.0
        tolerance = 1e-6
        
        for _ in range(50):  # Max iterations
            threshold = (left + right) / 2
            
            # Compute FAR at this threshold using binary search on sorted scores
            num_above = len(sorted_diff_scores) - np.searchsorted(sorted_diff_scores, threshold, side='left')
            far = num_above / len(sorted_diff_scores)
            
            if abs(far - target_far) < tolerance:
                break
                
            if far > target_far:
                left = threshold
            else:
                right = threshold
        
        return threshold

    def save_results(self, output_path: str) -> None:
        """Save evaluation results to JSON file."""
        results_data = {
            'ground_truth_path': str(self.ground_truth_path),
            'data_root': str(self.data_root),
            'results': [
                {
                    'image_path': r.image_path,
                    'has_animal_gt': r.has_animal_gt,
                    'has_animal_pred': r.has_animal_pred,
                    'identity_gt': r.identity_gt,
                    'similar_images': r.similar_images,
                    'detection_correct': r.detection_correct,
                    'identity_rank': r.identity_rank
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
