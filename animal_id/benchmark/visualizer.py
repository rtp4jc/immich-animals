"""
Visualization interface for benchmark results.

Provides interactive visualization of query images and their most similar matches.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2

from ..common.visualization import setup_output_dir, save_or_show_plot
from .evaluator import BenchmarkEvaluator, EvaluationResult


class BenchmarkVisualizer:
    """Visualizes benchmark evaluation results."""
    
    def __init__(self, evaluator: BenchmarkEvaluator):
        self.evaluator = evaluator
        self.results = evaluator.get_results()
        self.data_root = evaluator.data_root
        
    def visualize_queries(self, 
                         query_images: List[str],
                         output_dir: Union[str, Path] = "outputs/benchmark_visualization",
                         top_k: int = 5,
                         display: bool = False) -> None:
        """Visualize query images and their most similar matches."""
        output_path = setup_output_dir(output_dir)
        
        # Find results for query images
        query_results = []
        for query_path in query_images:
            for result in self.results:
                if result.image_path == query_path:
                    query_results.append(result)
                    break
        
        if not query_results:
            print("No query images found in evaluation results")
            return
            
        # Create visualization grid
        n_queries = len(query_results)
        n_cols = top_k + 1
        
        fig, axes = plt.subplots(n_queries, n_cols, figsize=(3 * n_cols, 3 * n_queries))
        if n_queries == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(query_results):
            # Query image
            query_img_path = self.data_root / result.image_path
            query_img = self._load_image(query_img_path)
            
            axes[i, 0].imshow(query_img)
            axes[i, 0].set_title(f"Query: {result.identity_gt or 'No Dog'}")
            axes[i, 0].axis('off')
            
            # Detection status border
            border_color = 'green' if result.detection_correct else 'red'
            for spine in axes[i, 0].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            
            # Top matches
            matches = result.similar_images[:top_k]
            for j, (match_path, similarity) in enumerate(matches):
                match_img = self._load_image(match_path)
                axes[i, j + 1].imshow(match_img)
                
                match_identity = self._get_identity_for_path(match_path)
                is_correct = match_identity == result.identity_gt if result.identity_gt else False
                
                title_color = 'green' if is_correct else 'red'
                axes[i, j + 1].set_title(f"{match_identity or 'No Dog'}\n{similarity:.3f}", 
                                       color=title_color)
                axes[i, j + 1].axis('off')
        
        plt.tight_layout()
        save_or_show_plot(output_path / "query_results.png", display)
        
    def plot_metrics_summary(self,
                           output_dir: Union[str, Path] = "outputs/benchmark_visualization",
                           display: bool = False) -> None:
        """Plot summary of benchmark metrics."""
        output_path = setup_output_dir(output_dir)
        metrics = self.evaluator._compute_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Detection metrics
        det_metrics = [metrics.detection_accuracy, metrics.detection_precision, metrics.detection_recall]
        det_labels = ['Accuracy', 'Precision', 'Recall']
        axes[0, 0].bar(det_labels, det_metrics)
        axes[0, 0].set_title('Detection Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # Top-K accuracy
        k_values = list(metrics.top_k_accuracy.keys())
        k_accuracies = list(metrics.top_k_accuracy.values())
        axes[0, 1].plot(k_values, k_accuracies, 'o-')
        axes[0, 1].set_title('Top-K Identity Accuracy')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Dataset composition
        dataset_counts = [metrics.animal_images, metrics.non_animal_images]
        dataset_labels = ['Animal Images', 'Non-Animal Images']
        axes[1, 0].pie(dataset_counts, labels=dataset_labels, autopct='%1.1f%%')
        axes[1, 0].set_title('Dataset Composition')
        
        # MRR display
        axes[1, 1].bar(['Mean Reciprocal Rank'], [metrics.mean_reciprocal_rank])
        axes[1, 1].set_title('Identity Ranking Quality')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        save_or_show_plot(output_path / "metrics_summary.png", display)
        
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and convert image to RGB."""
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _get_identity_for_path(self, image_path: str) -> Optional[str]:
        """Get identity label for an image path."""
        try:
            rel_path = str(Path(image_path).relative_to(self.data_root))
        except ValueError:
            rel_path = image_path
            
        for item in self.evaluator.ground_truth:
            if item['image_path'] == rel_path:
                return item.get('identity_label')
        return None
