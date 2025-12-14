import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from animal_id.benchmark.evaluator import BenchmarkEvaluator
from animal_id.benchmark.visualizer import BenchmarkVisualizer
from animal_id.tracking.wandb_logger import WandBLogger
from animal_id.pipeline.ambidextrous_axolotl import AmbidextrousAxolotl
from animal_id.pipeline.models import AnimalClass
from animal_id.common.constants import (
    DATA_DIR,
    ONNX_DETECTOR_PATH,
    ONNX_KEYPOINT_PATH,
    ONNX_EMBEDDING_PATH,
)
from animal_id.common.identity_loader import IdentityLoader


class ONNXDetector:
    """ONNX detection model wrapper."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def predict(self, image: np.ndarray):
        """Detect animals in image."""
        detector_input, original_shape = self._preprocess(image)
        detections = self.session.run(
            None, {self.session.get_inputs()[0].name: detector_input}
        )[0][0]

        results = []
        h, w = original_shape
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            if conf < 0.1:
                continue

            # Scale to original image size
            x1 = int(x1 * w / self.input_size[1])
            x2 = int(x2 * w / self.input_size[1])
            y1 = int(y1 * h / self.input_size[0])
            y2 = int(y2 * h / self.input_size[0])

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": AnimalClass.DOG,
                }
            )

        return results

    def _preprocess(self, image: np.ndarray):
        """Preprocess image for detection."""
        original_shape = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_fp = resized.astype(np.float32) / 255.0
        img_fp = np.transpose(img_fp, (2, 0, 1))
        return np.expand_dims(img_fp, axis=0), original_shape


class ONNXKeypoint:
    """ONNX keypoint model wrapper."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def predict(self, image: np.ndarray):
        """Detect keypoints in image."""
        keypoint_input, crop_shape = self._preprocess(image)
        detections = self.session.run(
            None, {self.session.get_inputs()[0].name: keypoint_input}
        )[0][0]

        results = []
        ch, cw = crop_shape
        for det in detections:
            if len(det) < 7:
                continue
            conf = det[4]
            keypoints = det[6:].reshape((4, 3))

            # Scale keypoints to crop size
            keypoints[:, 0] = keypoints[:, 0] * cw / self.input_size[1]
            keypoints[:, 1] = keypoints[:, 1] * ch / self.input_size[0]

            results.append({"keypoints": keypoints.tolist(), "confidence": float(conf)})

        return results

    def _preprocess(self, image: np.ndarray):
        """Preprocess image for keypoint detection."""
        crop_shape = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_fp = resized.astype(np.float32) / 255.0
        img_fp = np.transpose(img_fp, (2, 0, 1))
        return np.expand_dims(img_fp, axis=0), crop_shape


class ONNXEmbedding:
    """ONNX embedding model wrapper."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for image."""
        embedding_input = self._preprocess(image)
        embedding = self.session.run(
            None, {self.session.get_inputs()[0].name: embedding_input}
        )[0][0]
        return embedding

    def _preprocess(self, image: np.ndarray):
        """Preprocess image for embedding."""
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        img_fp = resized.astype(np.float32) / 255.0
        img_fp = np.transpose(img_fp, (2, 0, 1))
        return np.expand_dims(img_fp, axis=0)


def main(args):
    """Main function to run benchmark evaluation."""
    val_json_path = DATA_DIR / "identity_val.json"
    if not val_json_path.exists():
        print(f"[ERROR] Validation JSON not found: {val_json_path}")
        sys.exit(1)

    print("Initializing AmbidextrousAxolotl pipeline...")

    # Initialize models
    detector = ONNXDetector(str(ONNX_DETECTOR_PATH))
    keypoint_model = ONNXKeypoint(str(ONNX_KEYPOINT_PATH))
    embedding_model = ONNXEmbedding(str(ONNX_EMBEDDING_PATH))

    # Create pipeline systems
    axolotl_with_keypoints = AmbidextrousAxolotl(
        detector=detector,
        keypoint_model=keypoint_model,
        embedding_model=embedding_model,
        target_class=AnimalClass.DOG,
        use_keypoints=True,
    )

    axolotl_without_keypoints = AmbidextrousAxolotl(
        detector=detector,
        keypoint_model=keypoint_model,
        embedding_model=embedding_model,
        target_class=AnimalClass.DOG,
        use_keypoints=False,
    )

    # Load validation data using IdentityLoader
    loader = IdentityLoader()
    ground_truth = loader.load_validation_data(
        num_images=args.num_images, include_additional=args.include_additional
    )
    
    # Create identity map for logging (path -> identity)
    identity_map = {
        item['image_path']: item['identity_label'] 
        for item in ground_truth 
        if item.get('identity_label')
    }

    dataset_size = "full dataset" if args.num_images is None else f"{args.num_images} images"
    print(
        f"Found {len(ground_truth)} validation images. Processing {dataset_size} ({len(ground_truth)} images)."
    )

    # Save temporary ground truth file
    temp_gt_path = PROJECT_ROOT / "outputs/temp_ground_truth.json"
    temp_gt_path.parent.mkdir(exist_ok=True)
    with open(temp_gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Evaluate both systems
    evaluator = BenchmarkEvaluator(str(temp_gt_path), str(PROJECT_ROOT))
    
    # Common config for logging
    common_config = {
        "num_images": args.num_images,
        "include_additional": args.include_additional,
        "dataset_size": len(ground_truth)
    }
    
    # Prepare tags
    user_tags = [args.tag] if args.tag else []

    # 1. WITH Keypoints
    print("\nEvaluating AmbidextrousAxolotl WITH keypoints...")
    wandb_kp = WandBLogger(
        project_name="animal-id-pipeline",
        group="pipeline-with-keypoints",
        config={**common_config, "use_keypoints": True},
        tags=["pipeline", "keypoints"] + user_tags,
        enabled=not args.no_wandb
    )
    wandb_kp.start()
    
    metrics_with_kp = evaluator.evaluate(axolotl_with_keypoints)
    
    wandb_kp.log_metrics(metrics_with_kp)
    wandb_kp.log_failures(evaluator.get_results(), data_root=PROJECT_ROOT, identity_map=identity_map)
    wandb_kp.finish()

    # 2. WITHOUT Keypoints
    print("Evaluating AmbidextrousAxolotl WITHOUT keypoints...")
    wandb_no_kp = WandBLogger(
        project_name="animal-id-pipeline",
        group="pipeline-no-keypoints",
        config={**common_config, "use_keypoints": False},
        tags=["pipeline", "baseline"] + user_tags,
        enabled=not args.no_wandb
    )
    wandb_no_kp.start()
    
    metrics_without_kp = evaluator.evaluate(axolotl_without_keypoints)
    
    wandb_no_kp.log_metrics(metrics_without_kp)
    wandb_no_kp.log_failures(evaluator.get_results(), data_root=PROJECT_ROOT, identity_map=identity_map)
    wandb_no_kp.finish()

    # Print results
    print(f"\n{'='*60}")
    print("AMBIDEXTROUS AXOLOTL BENCHMARK RESULTS")
    print(f"{'='*60}")
    print("\nWITH KEYPOINTS:")
    print(metrics_with_kp)
    print("\nWITHOUT KEYPOINTS:")
    print(metrics_without_kp)

    # Report missed detections for keypoint version
    missed_with_kp = sum(
        1 for r in evaluator.get_results() if r.has_animal_gt and not r.has_animal_pred
    )
    if missed_with_kp > 0:
        print(f"\n--- {missed_with_kp} images with no animals detected (with keypoints) ---")
        missed_examples = [
            r
            for r in evaluator.get_results()
            if r.has_animal_gt and not r.has_animal_pred
        ][:10]
        for result in missed_examples:
            print(f"  - {result.image_path} (ID {result.identity_gt})")
        if missed_with_kp > 10:
            print(f"  ... and {missed_with_kp - 10} more")

    # Create visualizations
    if args.num_queries > 0:
        print(f"\nCreating visualizations with {args.num_queries} query images...")
        visualizer = BenchmarkVisualizer(evaluator)

        # Select query images (evenly spaced like original)
        valid_queries = [
            item["image_path"] for item in ground_truth if item["identity_label"]
        ]
        if valid_queries:
            num_queries = min(args.num_queries, len(valid_queries))
            query_indices = np.linspace(
                0, len(valid_queries) - 1, num_queries, dtype=int
            )
            query_images = [valid_queries[i] for i in query_indices]

            visualizer.visualize_queries(
                query_images, "outputs/ambidextrous_axolotl", display=False
            )
            visualizer.plot_metrics_summary(
                "outputs/ambidextrous_axolotl", display=False
            )

            print(f"Visualizations saved to outputs/ambidextrous_axolotl/")

    # Clean up
    temp_gt_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark AmbidextrousAxolotl pipeline with and without keypoints."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to process from the validation set. If not specified, uses full dataset.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of query images to show in visualization.",
    )
    parser.add_argument(
        "--include-additional",
        action="store_true",
        help="Include additional identities from data/additional_identities",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for the WandB run",
    )
    args = parser.parse_args()
    main(args)
