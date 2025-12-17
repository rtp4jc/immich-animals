"""
WandB Logger for animal identification benchmarking.

Encapsulates Weights & Biases logging logic for the pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb

from ..benchmark.evaluator import BenchmarkMetrics, EvaluationResult

logger = logging.getLogger(__name__)


class WandBLogger:
    """Handles logging of benchmark results to Weights & Biases."""

    def __init__(
        self,
        project_name: str = "animal-id-benchmark",
        run_name: Optional[str] = None,
        group: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        """
        Initialize WandB logger.

        Args:
            project_name: Name of the WandB project
            run_name: Optional name for this specific run
            group: Optional group name for the run
            config: Dictionary of configuration parameters to log
            tags: List of tags to organize runs (e.g., ["baseline", "experiment"])
            enabled: Whether logging is enabled (useful for dry runs)
        """
        self.project_name = project_name
        self.run_name = run_name
        self.group = group
        self.config = config or {}
        self.tags = tags or []
        self.enabled = enabled
        self.run = None

    def start(self):
        """Start a new WandB run."""
        if not self.enabled:
            return

        try:
            self.run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                group=self.group,
                config=self.config,
                tags=self.tags,
                reinit=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize WandB: {e}. Logging disabled for this run."
            )
            self.enabled = False

    def log_metrics(self, metrics: BenchmarkMetrics, prefix: str = ""):
        """
        Log benchmark metrics to WandB.

        Args:
            metrics: BenchmarkMetrics object containing results
            prefix: Optional prefix for metric names (e.g., "val_")
        """
        if not self.enabled or self.run is None:
            return

        log_data = {
            f"{prefix}detection_accuracy": metrics.detection_accuracy,
            f"{prefix}detection_precision": metrics.detection_precision,
            f"{prefix}detection_recall": metrics.detection_recall,
            f"{prefix}mean_reciprocal_rank": metrics.mean_reciprocal_rank,
            f"{prefix}top_1_accuracy": metrics.top_k_accuracy.get(1, 0.0),
            f"{prefix}top_3_accuracy": metrics.top_k_accuracy.get(3, 0.0),
            f"{prefix}top_5_accuracy": metrics.top_k_accuracy.get(5, 0.0),
            f"{prefix}total_images": metrics.total_images,
        }

        # Log TAR @ FAR as a custom chart or table
        if metrics.tar_at_far:
            # Also log scalar values for quick comparison
            for far, (tar, _) in metrics.tar_at_far.items():
                # Format FAR nicely (e.g. 0.001 -> "0_001")
                far_str = str(far).replace(".", "_")
                log_data[f"{prefix}tar_at_far_{far_str}"] = tar

        wandb.log(log_data)

    def log_failures(
        self,
        results: List[EvaluationResult],
        max_failures: int = 20,
        data_root: Optional[Path] = None,
        identity_map: Optional[Dict[str, str]] = None,
    ):
        """
        Log failure cases (missed detections or wrong identities) as images.

        Args:
            results: List of evaluation results
            max_failures: Maximum number of failure images to log per category
            data_root: Root directory for relative image paths
            identity_map: Mapping from image path (str) to identity label (str)
        """
        if not self.enabled or self.run is None:
            return

        # 1. Missed Detections (False Negatives)
        missed_detections = [
            r for r in results if r.has_animal_gt and not r.has_animal_pred
        ]
        self._log_image_list(
            missed_detections[:max_failures],
            "missed_detections",
            "Missed Detection",
            data_root,
            identity_map,
        )

        # 2. False Positives (Detected something where there was nothing)
        false_positives = [
            r for r in results if not r.has_animal_gt and r.has_animal_pred
        ]
        self._log_image_list(
            false_positives[:max_failures],
            "false_positives",
            "False Positive",
            data_root,
            identity_map,
        )

        # 3. Incorrect Identifications (Detection correct, but identity wrong)
        # We consider it "wrong" if the correct identity wasn't in the top 1 result
        wrong_identities = [
            r
            for r in results
            if r.has_animal_gt
            and r.has_animal_pred
            and r.identity_gt
            and r.identity_rank != 1
        ]
        self._log_image_list(
            wrong_identities[:max_failures],
            "wrong_identities",
            "Wrong Identity (Rank != 1)",
            data_root,
            identity_map,
        )

    def _log_image_list(
        self,
        results: List[EvaluationResult],
        key: str,
        caption_prefix: str,
        data_root: Optional[Path],
        identity_map: Optional[Dict[str, str]] = None,
    ):
        """Helper to log a list of images to WandB."""
        if not results:
            return

        images = []
        for r in results:
            try:
                # Resolve image path
                img_path = Path(r.image_path)
                if not img_path.exists() and data_root:
                    img_path = data_root / r.image_path

                if img_path.exists():
                    # Build caption
                    caption = f"{caption_prefix}\nGT: {r.identity_gt}"

                    if r.identity_rank:
                        rank_str = (
                            f"{r.identity_rank}" if r.identity_rank <= 5 else ">5"
                        )
                        caption += f" | Rank: {rank_str}"
                    elif r.has_animal_pred and r.identity_gt:
                        caption += " | Rank: >5"  # Implied if rank is None but prediction existed

                    # Add top prediction info
                    if r.similar_images:
                        top_path, score = r.similar_images[0]
                        pred_id = "?"
                        if identity_map:
                            # Try to find identity for top match
                            # The paths in similar_images might be absolute or relative differently than keys
                            # We'll try a few variations if simple lookup fails
                            if top_path in identity_map:
                                pred_id = identity_map[top_path]
                            else:
                                # Try relative to data root if passed
                                try:
                                    rel = (
                                        str(Path(top_path).relative_to(data_root))
                                        if data_root
                                        else top_path
                                    )
                                    pred_id = identity_map.get(rel, "?")
                                except ValueError:
                                    pred_id = "?"

                        caption += f"\nPred: {pred_id} ({score:.3f})"

                    img = wandb.Image(str(img_path), caption=caption)
                    images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image for logging {r.image_path}: {e}")

        if images:
            wandb.log({key: images})

    def finish(self):
        """Finish the WandB run."""
        if self.enabled and self.run:
            self.run.finish()
