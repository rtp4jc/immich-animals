#!/usr/bin/env python
"""
Benchmark animal identification pipeline using the Immich API as a black box.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import requests
from typing import List, Tuple
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from animal_id.benchmark.evaluator import BenchmarkEvaluator, AnimalIdentificationSystem
from animal_id.benchmark.visualizer import BenchmarkVisualizer
from animal_id.common.constants import DATA_DIR
from animal_id.common.identity_loader import IdentityLoader


class ImmichAnimalSystem(AnimalIdentificationSystem):
    """Immich API-based animal identification system for benchmarking."""

    def __init__(
        self, host: str = "localhost", port: int = 3003, model_config: dict = None
    ):
        self.host = host
        self.port = port
        self.model_config = model_config or {
            "dog-identification": {
                "detection": {"modelName": "dog_detector"},
                "keypoint": {"modelName": "dog_keypoint"},
                "recognition": {"modelName": "dog_embedder"},
            }
        }
        self.gallery_embeddings = None
        self.gallery_paths = None

    def build_gallery(self, image_paths: List[str]) -> None:
        """Pre-compute embeddings for all gallery images."""
        embeddings = []
        valid_paths = []

        for image_path in tqdm(image_paths, desc="Building gallery"):
            embedding = self._get_embedding(image_path)
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
        query_embedding = self._get_embedding(image_path)

        if query_embedding is None:
            return False, []

        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            return True, []

        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity([query_embedding], self.gallery_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        # Return top matches (excluding self)
        similar_images = []
        for idx in sorted_indices:
            gallery_path = self.gallery_paths[idx]
            if gallery_path != image_path:
                similar_images.append((gallery_path, float(similarities[idx])))

        return True, similar_images

    def _get_embedding(self, image_path: str) -> np.ndarray:
        """Get embedding for a single image via Immich API."""
        url = f"http://{self.host}:{self.port}/predict"

        try:
            with open(image_path, "rb") as f:
                files = {"image": (Path(image_path).name, f, "image/jpeg")}
                data = {"entries": json.dumps(self.model_config)}

                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()

                results = response.json()
                embeddings = results.get("dog-identification", [])
                return np.array(embeddings[0]) if embeddings else None

        except Exception as e:
            print(f"[ERROR] Request failed for {image_path}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark animal identification pipeline via Immich API"
    )
    parser.add_argument("--host", default="localhost", help="Immich ML host")
    parser.add_argument("--port", type=int, default=3003, help="Immich ML port")
    parser.add_argument(
        "--num-images", type=int, default=50, help="Number of images to evaluate"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of query images for visualization",
    )
    parser.add_argument(
        "--skip-keypoints", action="store_true", help="Skip keypoint stage in pipeline"
    )
    parser.add_argument(
        "--include-additional",
        action="store_true",
        help="Include additional identities from data/additional_identities",
    )
    args = parser.parse_args()

    # Load validation data using IdentityLoader
    loader = IdentityLoader()
    ground_truth = loader.load_validation_data(
        num_images=args.num_images, include_additional=args.include_additional
    )

    print(f"Evaluating {len(ground_truth)} images via Immich API...")

    # Save temporary ground truth file
    temp_gt_path = PROJECT_ROOT / "outputs/temp_ground_truth.json"
    temp_gt_path.parent.mkdir(exist_ok=True)
    with open(temp_gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Create model configuration
    if args.skip_keypoints:
        model_config = {
            "dog-identification": {
                "detection": {"modelName": "dog_detector"},
                "recognition": {"modelName": "dog_embedder_direct"},
            }
        }
    else:
        model_config = {
            "dog-identification": {
                "detection": {"modelName": "dog_detector"},
                "keypoint": {"modelName": "dog_keypoint"},
                "recognition": {"modelName": "dog_embedder"},
            }
        }

    # Add keypoint stage if not skipping
    # if not args.skip_keypoints:
    #     model_config["dog-identification"]["keypoint"] = {"modelName": "dog_keypoint"}

    # Create system
    immich_system = ImmichAnimalSystem(args.host, args.port, model_config)

    # Evaluate system
    evaluator = BenchmarkEvaluator(str(temp_gt_path), str(PROJECT_ROOT))

    pipeline_type = "WITHOUT keypoints" if args.skip_keypoints else "WITH keypoints"
    print(f"Evaluating Immich pipeline {pipeline_type}...")
    metrics = evaluator.evaluate(immich_system)

    # Print results
    print(f"\n{'='*60}")
    print(f"IMMICH ANIMAL IDENTIFICATION BENCHMARK RESULTS ({pipeline_type})")
    print(f"{'='*60}")
    print(metrics)

    # Create visualizations
    if args.num_queries > 0:
        print(f"\nCreating visualizations with {args.num_queries} query images...")
        visualizer = BenchmarkVisualizer(evaluator)

        # Select query images
        valid_queries = [
            item["image_path"] for item in ground_truth if item["identity_label"]
        ]
        if valid_queries:
            import random

            random.seed(42)
            num_queries = min(args.num_queries, len(valid_queries))
            query_images = random.sample(valid_queries, num_queries)

            output_dir = (
                "outputs/immich_with_keypoints"
                if not args.skip_keypoints
                else "outputs/immich_without_keypoints"
            )
            visualizer.visualize_queries(query_images, output_dir, display=False)
            visualizer.plot_metrics_summary(output_dir, display=False)

            print(f"Visualizations saved to {output_dir}/")

    # Clean up
    temp_gt_path.unlink()


if __name__ == "__main__":
    main()
