#!/usr/bin/env python
"""Training and export pipeline for the Animal ID models.

Subcommands cover each stage on its own (``detection-data``, ``detection``,
``embedding-data``, ``embedding``, ``export-detector``, ``export-embedding``,
``benchmark``); ``all`` — the default — runs detection, then embedding, then the
benchmark.

    python scripts/train_master.py                    # everything
    python scripts/train_master.py embedding          # just the embedding model
    python scripts/train_master.py benchmark --tag v2
"""

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from animal_id.benchmark.evaluator import BenchmarkEvaluator
from animal_id.benchmark.metrics import evaluate_embedding_model
from animal_id.common.constants import (
    DATA_DIR,
    DETECTOR_PROJECT_DIR,
    DETECTOR_RUN_NAME,
    MODELS_DIR,
    ONNX_DETECTOR_PATH,
    ONNX_EMBEDDING_PATH,
    ONNX_KEYPOINT_PATH,
)
from animal_id.common.datasets import IdentityDataset
from animal_id.common.identity_loader import IdentityLoader
from animal_id.common.logging_config import setup_logging
from animal_id.common.seed import set_seed, worker_init_fn
from animal_id.common.utils import find_latest_run, find_latest_timestamped_run
from animal_id.detection.dataset_converter import (
    CocoDetectorDatasetConverter,
    create_default_config,
)
from animal_id.detection.trainer import DetectionTrainer
from animal_id.detection.yolo_converter import (
    CocoToYoloDetectionConverter,
)
from animal_id.embedding.config import (
    DATA_CONFIG,
    DEFAULT_BACKBONE,
    TRAINING_CONFIG,
)
from animal_id.embedding.dataset_converter import EmbeddingDatasetConverter
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.models import AnimalEmbeddingModel
from animal_id.embedding.trainer import EmbeddingTrainer
from animal_id.pipeline.animal_pipeline import AnimalPipeline
from animal_id.pipeline.models import AnimalClass
from animal_id.pipeline.onnx_models import ONNXDetector, ONNXEmbedding, ONNXKeypoint
from animal_id.tracking.wandb_logger import WandBLogger

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42


setup_logging()
logger = logging.getLogger(__name__)


def run_full_pipeline_benchmark(
    num_images=None, include_additional=False, tag=None, no_wandb=False
):
    """Runs the full AnimalPipeline benchmark."""
    logger.info("STARTING FULL PIPELINE BENCHMARK")

    # Headline benchmark numbers are reported on the held-out TEST split (disjoint
    # identities from train/val). Val is reserved for model selection / early-stopping.
    test_json_path = DATA_DIR / "identity_test.json"
    if not test_json_path.exists():
        logger.error(f"Test JSON not found: {test_json_path}")
        return False

    logger.info("Initializing AnimalPipeline...")

    # Initialize models
    # Note: We use the just-exported ONNX models
    if not ONNX_DETECTOR_PATH.exists():
        logger.error(f"Detector ONNX model not found at {ONNX_DETECTOR_PATH}")
        return False

    # We use existing keypoint model if available, or skip keypoint part if not
    has_keypoints = ONNX_KEYPOINT_PATH.exists()
    if not has_keypoints:
        logger.warning(
            f"Keypoint ONNX model not found at {ONNX_KEYPOINT_PATH}. Skipping keypoint-enabled benchmark."
        )

    if not ONNX_EMBEDDING_PATH.exists():
        logger.error(f"Embedding ONNX model not found at {ONNX_EMBEDDING_PATH}")
        return False

    detector = ONNXDetector(str(ONNX_DETECTOR_PATH))
    embedding_model = ONNXEmbedding(str(ONNX_EMBEDDING_PATH))

    # Placeholder for keypoint model - only loaded if file exists
    keypoint_model = None
    if has_keypoints:
        keypoint_model = ONNXKeypoint(str(ONNX_KEYPOINT_PATH))

    # Load held-out test data for the reported (headline) benchmark
    loader = IdentityLoader(json_filename="identity_test.json")
    ground_truth = loader.load_validation_data(
        num_images=num_images, include_additional=include_additional
    )

    identity_map = {
        item["image_path"]: item["identity_label"]
        for item in ground_truth
        if item.get("identity_label")
    }

    dataset_size = "full dataset" if num_images is None else f"{num_images} images"
    logger.info(f"Found {len(ground_truth)} test images. Processing {dataset_size}.")

    # Save temporary ground truth file for Evaluator
    temp_gt_path = PROJECT_ROOT / "outputs/temp_ground_truth.json"
    temp_gt_path.parent.mkdir(exist_ok=True)
    with open(temp_gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    evaluator = BenchmarkEvaluator(str(temp_gt_path), str(PROJECT_ROOT))

    common_config = {
        "num_images": num_images,
        "include_additional": include_additional,
        "dataset_size": len(ground_truth),
        "pipeline": "AnimalPipeline",
    }
    user_tags = [tag] if tag else []

    # --- Run 1: WITHOUT Keypoints (Primary Goal) ---
    logger.info("\nEvaluating AnimalPipeline WITHOUT keypoints...")

    axolotl_without_keypoints = AnimalPipeline(
        detector=detector,
        embedding_model=embedding_model,
        keypoint_model=keypoint_model,
        target_class=AnimalClass.DOG,
        use_keypoints=False,
    )

    wandb_no_kp = WandBLogger(
        project_name="animal-id-pipeline",
        group="pipeline-no-keypoints",
        config={**common_config, "use_keypoints": False},
        tags=["pipeline", "baseline", "master-script"] + user_tags,
        enabled=not no_wandb,
    )
    wandb_no_kp.start()

    metrics_without_kp = evaluator.evaluate(axolotl_without_keypoints)

    wandb_no_kp.log_metrics(metrics_without_kp)
    wandb_no_kp.log_failures(
        evaluator.get_results(), data_root=PROJECT_ROOT, identity_map=identity_map
    )
    wandb_no_kp.finish()

    logger.info("\nResults (No Keypoints):")
    logger.info(str(metrics_without_kp))

    # --- Run 2: WITH Keypoints (Optional) ---
    if has_keypoints and keypoint_model:
        logger.info("\nEvaluating AnimalPipeline WITH keypoints...")

        axolotl_with_keypoints = AnimalPipeline(
            detector=detector,
            embedding_model=embedding_model,
            keypoint_model=keypoint_model,
            target_class=AnimalClass.DOG,
            use_keypoints=True,
        )

        wandb_kp = WandBLogger(
            project_name="animal-id-pipeline",
            group="pipeline-with-keypoints",
            config={**common_config, "use_keypoints": True},
            tags=["pipeline", "keypoints", "master-script"] + user_tags,
            enabled=not no_wandb,
        )
        wandb_kp.start()

        metrics_with_kp = evaluator.evaluate(axolotl_with_keypoints)

        wandb_kp.log_metrics(metrics_with_kp)
        wandb_kp.log_failures(
            evaluator.get_results(), data_root=PROJECT_ROOT, identity_map=identity_map
        )
        wandb_kp.finish()

        logger.info("\nResults (With Keypoints):")
        logger.info(str(metrics_with_kp))

    # Cleanup
    if temp_gt_path.exists():
        temp_gt_path.unlink()

    return True


def run_detection_data_prep(
    output_dir="data/detector/coco", yaml_path="data/detector/dogs_detection.yaml"
):
    """Runs the data preparation and conversion for the detection model."""
    logger.info("STARTING DETECTION DATA PREPARATION")

    # Create COCO dataset
    config = create_default_config()
    config["output_dir"] = output_dir
    converter = CocoDetectorDatasetConverter(config)
    converter.convert()

    # Convert to YOLO format
    yolo_converter = CocoToYoloDetectionConverter(
        coco_annotations_dir=output_dir,
        labels_output_dir="data",
        data_root="data",
        yaml_output_path=yaml_path,
    )
    yolo_converter.convert()


def run_detection_pipeline(
    output_dir="data/detector/coco", yaml_path="data/detector/dogs_detection.yaml"
):
    """Runs the full detection pipeline."""
    logger.info("STARTING DETECTION PIPELINE")

    # 1. Prepare Data
    logger.info("Step 1: Preparing Detection Dataset (COCO -> YOLO)")
    run_detection_data_prep(output_dir, yaml_path)

    # 2. Train Model
    logger.info("\nStep 2: Training YOLOv11 Detector")
    model_name = "yolo11n.pt"
    epochs = 100
    batch_size = 16
    imgsz = 640

    trainer = DetectionTrainer(model_name)
    trainer.update_config(data=yaml_path, epochs=epochs, batch=batch_size, imgsz=imgsz)
    results = trainer.train()
    logger.info(f"Detection training complete. Results saved to: {results.save_dir}")

    # 3. Export to ONNX
    best_model_path = results.save_dir / "weights/best.pt"
    run_detector_export(best_model_path)


def run_detector_export(model_path: Path):
    """Exports a trained detector model to ONNX format."""
    logger.info("\nStep 3: Exporting Detector to ONNX")

    if not model_path.exists():
        # Raise an exception instead of returning False
        raise FileNotFoundError(f"Best model not found at {model_path}")

    from ultralytics import YOLO

    model = YOLO(model_path)
    exported_path_str = model.export(format="onnx", opset=12, nms=True)
    exported_path = Path(exported_path_str)

    ONNX_DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.replace(exported_path, ONNX_DETECTOR_PATH)
    logger.info(f"Detector ONNX exported to: {ONNX_DETECTOR_PATH}")


def run_embedding_data_prep():
    """Runs the data preparation step for the embedding model."""
    logger.info("STARTING EMBEDDING DATA PREPARATION")

    dataset_converter = EmbeddingDatasetConverter(
        source_path=DATA_CONFIG.dogfacenet_path,
        output_train_json=DATA_CONFIG.train_json_path,
        output_val_json=DATA_CONFIG.val_json_path,
        output_test_json=DATA_CONFIG.test_json_path,
    )
    dataset_converter.convert()


def run_embedding_pipeline():
    """Runs the full embedding pipeline."""
    logger.info("STARTING EMBEDDING PIPELINE")

    # 1. Prepare Data
    logger.info("Step 1: Preparing Embedding Dataset")
    run_embedding_data_prep()

    # 2. Train Model
    logger.info("\nStep 2: Training Embedding Model")

    # Setup Run Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_name = DEFAULT_BACKBONE.value
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_{backbone_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training run directory: {run_dir}")

    # Save Run Config
    config_to_save = {
        "backbone": backbone_name,
        "training_config": asdict(TRAINING_CONFIG),
        "data_config": asdict(DATA_CONFIG),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    g = set_seed(SEED)

    # Load Datasets
    train_dataset = IdentityDataset(
        json_path=PROJECT_ROOT / DATA_CONFIG.train_json_path,
        img_size=DATA_CONFIG.img_size,
        is_training=True,
    )
    val_dataset = IdentityDataset(
        json_path=PROJECT_ROOT / DATA_CONFIG.val_json_path,
        img_size=DATA_CONFIG.img_size,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG.batch_size,
        shuffle=True,
        num_workers=TRAINING_CONFIG.hardware_workers,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DATA_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG.hardware_workers,
    )

    # Create Model
    model = AnimalEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=train_dataset.num_classes,
        embedding_dim=TRAINING_CONFIG.embedding_dim,
    ).to(device)

    # Create Trainer
    trainer = EmbeddingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
    )

    # Execute Training
    best_model_path = trainer.train(
        warmup_epochs=TRAINING_CONFIG.warmup_epochs,
        full_epochs=TRAINING_CONFIG.full_train_epochs,
        head_lr=TRAINING_CONFIG.head_lr,
        backbone_lr=TRAINING_CONFIG.backbone_lr,
        full_lr=TRAINING_CONFIG.full_train_lr,
        patience=TRAINING_CONFIG.early_stopping_patience,
    )
    logger.info(f"Embedding training complete. Best model: {best_model_path}")

    # 3. Evaluate Best Model and Export
    run_embedding_export(best_model_path, val_loader, device, train_dataset.num_classes)


def run_embedding_export(model_path, val_loader, device, num_classes):
    """Evaluates the best model and exports it to ONNX."""

    # Re-instantiate model for evaluation and export
    model = AnimalEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=num_classes,
        embedding_dim=TRAINING_CONFIG.embedding_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate Best Model
    logger.info("\nStep 3: Evaluating Best Model")
    val_metrics = evaluate_embedding_model(model, val_loader, device)

    logger.info("\nValidation Metrics:")
    for k, v in val_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Export to ONNX
    logger.info("\nStep 4: Exporting Embedding Model to ONNX")

    # Re-instantiate model on CPU for export to ensure consistency
    export_device = torch.device("cpu")
    export_model = AnimalEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=num_classes,
        embedding_dim=TRAINING_CONFIG.embedding_dim,
    )
    export_model.load_state_dict(torch.load(model_path, map_location=export_device))
    export_model.to(export_device)
    export_model.eval()

    export_embedding_onnx(
        export_model, ONNX_EMBEDDING_PATH, img_size=DATA_CONFIG.img_size
    )
    logger.info(f"Embedding ONNX exported to: {ONNX_EMBEDDING_PATH}")


def run_detector_export_latest():
    """Exports the most recent trained detector run to ONNX."""
    latest_run_dir = find_latest_run(DETECTOR_PROJECT_DIR, DETECTOR_RUN_NAME)
    if not latest_run_dir:
        raise FileNotFoundError(
            f"No training runs found for '{DETECTOR_RUN_NAME}' in '{DETECTOR_PROJECT_DIR}'."
        )

    model_checkpoint = latest_run_dir / "weights/best.pt"
    logger.info(f"Found latest model checkpoint: {model_checkpoint}")
    run_detector_export(model_checkpoint)


def run_embedding_export_latest():
    """Exports the most recent trained embedding run to ONNX."""
    latest_run = find_latest_timestamped_run()
    model_path = latest_run / "best_model.pt" if latest_run else None

    if model_path is None or not model_path.exists():
        # Fall back to the pre-`runs/` checkpoint location.
        model_path = MODELS_DIR / "dog_embedding_best.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found. Checked runs/*/best_model.pt and {model_path}"
            )

    logger.info(f"Found latest model checkpoint: {model_path}")

    # The export function evaluates the model first, so it needs a val loader, and
    # num_classes from the *training* split to rebuild the head.
    val_loader = DataLoader(
        IdentityDataset(
            json_path=DATA_CONFIG.val_json_path,
            img_size=DATA_CONFIG.img_size,
            is_training=False,
        ),
        batch_size=DATA_CONFIG.batch_size,
        shuffle=False,
        num_workers=2,
    )
    train_dataset = IdentityDataset(
        json_path=DATA_CONFIG.train_json_path,
        img_size=DATA_CONFIG.img_size,
        is_training=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_embedding_export(model_path, val_loader, device, train_dataset.num_classes)


def run_all(args):
    """Detection pipeline, then embedding pipeline, then the benchmark."""
    if args.skip_detection:
        logger.info("Manually skipping detection pipeline.")
    elif args.skip_trained and ONNX_DETECTOR_PATH.exists():
        logger.info(
            f"Skipping detection pipeline as {ONNX_DETECTOR_PATH} exists (--skip-trained)."
        )
    else:
        run_detection_pipeline()

    if args.skip_embedding:
        logger.info("Manually skipping embedding pipeline.")
    elif args.skip_trained and ONNX_EMBEDDING_PATH.exists():
        logger.info(
            f"Skipping embedding pipeline as {ONNX_EMBEDDING_PATH} exists (--skip-trained)."
        )
    else:
        run_embedding_pipeline()

    if not args.skip_benchmark:
        if ONNX_DETECTOR_PATH.exists() and ONNX_EMBEDDING_PATH.exists():
            run_full_pipeline_benchmark(no_wandb=args.no_wandb, tag=args.tag)
        else:
            logger.warning("Skipping benchmark because required ONNX models not found.")

    logger.info("All pipeline steps completed successfully.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Training and export pipeline for the Animal ID models."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser(
        "detection-data", help="Prepare the detection dataset (COCO -> YOLO)"
    )
    sub.add_parser("detection", help="Prepare, train and export the detector")
    sub.add_parser("embedding-data", help="Prepare the embedding dataset")
    sub.add_parser("embedding", help="Prepare, train and export the embedding model")
    sub.add_parser("export-detector", help="Export the latest detector run to ONNX")
    sub.add_parser("export-embedding", help="Export the latest embedding run to ONNX")

    benchmark = sub.add_parser("benchmark", help="Benchmark the full AnimalPipeline")
    benchmark.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Images to process from the test set. Default: the whole split.",
    )
    benchmark.add_argument(
        "--include-additional",
        action="store_true",
        help="Include identities from data/additional_identities",
    )

    run_all_parser = sub.add_parser(
        "all", help="Detection, then embedding, then benchmark (the default)"
    )
    run_all_parser.add_argument(
        "--skip-detection", action="store_true", help="Skip the detection pipeline"
    )
    run_all_parser.add_argument(
        "--skip-embedding", action="store_true", help="Skip the embedding pipeline"
    )
    run_all_parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip the pipeline benchmark"
    )
    run_all_parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip training a model whose ONNX file already exists",
    )

    # WandB options apply to both commands that can run the benchmark.
    for p in (benchmark, run_all_parser):
        p.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
        p.add_argument("--tag", default=None, help="Tag for the WandB run")

    return parser


COMMANDS = {
    "detection-data": lambda args: run_detection_data_prep(),
    "detection": lambda args: run_detection_pipeline(),
    "embedding-data": lambda args: run_embedding_data_prep(),
    "embedding": lambda args: run_embedding_pipeline(),
    "export-detector": lambda args: run_detector_export_latest(),
    "export-embedding": lambda args: run_embedding_export_latest(),
    "benchmark": lambda args: run_full_pipeline_benchmark(
        num_images=args.num_images,
        include_additional=args.include_additional,
        tag=args.tag,
        no_wandb=args.no_wandb,
    ),
    "all": run_all,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # No subcommand means "all", so the historical flag-only invocation
    # (`train_master.py --skip-detection`) still works.
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "all")
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
