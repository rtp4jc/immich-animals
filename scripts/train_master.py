#!/usr/bin/env python
"""
Master Training Script

This script orchestrates the entire training pipeline for the animal_id project.
It can train both the detector and the embedding model from scratch, ensuring
a consistent and reproducible workflow.

Workflow:
1. Detection Pipeline:
   - Prepare detection data (COCO -> YOLO format)
   - Train YOLOv11 detector
   - Export best detector to ONNX

2. Embedding Pipeline:
   - Prepare embedding data (DogFaceNet -> JSON)
   - Train Embedding model (ResNet backbone with ArcFace/CosFace)
   - Export best embedding model to ONNX

Usage:
    python scripts/train_master.py
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Imports for Detection ---
from animal_id.detection.dataset_converter import (
    CocoDetectorDatasetConverter,
    create_default_config,
)
from animal_id.detection.yolo_converter import (
    CocoToYoloDetectionConverter,
)
from animal_id.detection.trainer import DetectionTrainer
from animal_id.common.constants import (
    DETECTOR_PROJECT_DIR,
    DETECTOR_RUN_NAME,
    ONNX_DETECTOR_PATH,
)

# --- Imports for Embedding ---
import torch
from torch.utils.data import DataLoader
from animal_id.embedding.dataset_converter import EmbeddingDatasetConverter
from animal_id.embedding.models import DogEmbeddingModel
from animal_id.embedding.trainer import EmbeddingTrainer
from animal_id.embedding.config import (
    DEFAULT_BACKBONE,
    TRAINING_CONFIG,
    DATA_CONFIG,
)
from animal_id.common.datasets import DogIdentityDataset
from animal_id.common.constants import (
    ONNX_EMBEDDING_PATH,
    ONNX_KEYPOINT_PATH,
    DATA_DIR,
)
from animal_id.benchmark.metrics import evaluate_embedding_model

# --- Imports for Full Pipeline Benchmark ---
from animal_id.pipeline.ambidextrous_axolotl import AmbidextrousAxolotl
from animal_id.pipeline.onnx_models import ONNXDetector, ONNXKeypoint, ONNXEmbedding
from animal_id.pipeline.models import AnimalClass
from animal_id.benchmark.evaluator import BenchmarkEvaluator
from animal_id.tracking.wandb_logger import WandBLogger
from animal_id.common.identity_loader import IdentityLoader


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_full_pipeline_benchmark(num_images=None, include_additional=False, tag=None, no_wandb=False):
    """Runs the full AmbidextrousAxolotl pipeline benchmark."""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING FULL PIPELINE BENCHMARK")
    logger.info("=" * 60)

    val_json_path = DATA_DIR / "identity_val.json"
    if not val_json_path.exists():
        logger.error(f"Validation JSON not found: {val_json_path}")
        return False

    logger.info("Initializing AmbidextrousAxolotl pipeline...")

    # Initialize models
    # Note: We use the just-exported ONNX models
    if not ONNX_DETECTOR_PATH.exists():
         logger.error(f"Detector ONNX model not found at {ONNX_DETECTOR_PATH}")
         return False
    
    # We use existing keypoint model if available, or skip keypoint part if not
    has_keypoints = ONNX_KEYPOINT_PATH.exists()
    if not has_keypoints:
        logger.warning(f"Keypoint ONNX model not found at {ONNX_KEYPOINT_PATH}. Skipping keypoint-enabled benchmark.")

    if not ONNX_EMBEDDING_PATH.exists():
        logger.error(f"Embedding ONNX model not found at {ONNX_EMBEDDING_PATH}")
        return False

    detector = ONNXDetector(str(ONNX_DETECTOR_PATH))
    embedding_model = ONNXEmbedding(str(ONNX_EMBEDDING_PATH))
    
    # Placeholder for keypoint model - only loaded if file exists
    keypoint_model = None
    if has_keypoints:
        keypoint_model = ONNXKeypoint(str(ONNX_KEYPOINT_PATH))

    # Load validation data
    loader = IdentityLoader()
    ground_truth = loader.load_validation_data(
        num_images=num_images, include_additional=include_additional
    )
    
    identity_map = {
        item['image_path']: item['identity_label'] 
        for item in ground_truth 
        if item.get('identity_label')
    }

    dataset_size = "full dataset" if num_images is None else f"{num_images} images"
    logger.info(f"Found {len(ground_truth)} validation images. Processing {dataset_size}.")

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
        "pipeline": "AmbidextrousAxolotl"
    }
    user_tags = [tag] if tag else []

    # --- Run 1: WITHOUT Keypoints (Primary Goal) ---
    logger.info("\nEvaluating AmbidextrousAxolotl WITHOUT keypoints...")

    axolotl_without_keypoints = AmbidextrousAxolotl(
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
        enabled=not no_wandb
    )
    wandb_no_kp.start()
    
    metrics_without_kp = evaluator.evaluate(axolotl_without_keypoints)
    
    wandb_no_kp.log_metrics(metrics_without_kp)
    wandb_no_kp.log_failures(evaluator.get_results(), data_root=PROJECT_ROOT, identity_map=identity_map)
    wandb_no_kp.finish()

    logger.info("\nResults (No Keypoints):")
    logger.info(str(metrics_without_kp))

    # --- Run 2: WITH Keypoints (Optional) ---
    if has_keypoints and keypoint_model:
        logger.info("\nEvaluating AmbidextrousAxolotl WITH keypoints...")
        
        axolotl_with_keypoints = AmbidextrousAxolotl(
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
            enabled=not no_wandb
        )
        wandb_kp.start()
        
        metrics_with_kp = evaluator.evaluate(axolotl_with_keypoints)
        
        wandb_kp.log_metrics(metrics_with_kp)
        wandb_kp.log_failures(evaluator.get_results(), data_root=PROJECT_ROOT, identity_map=identity_map)
        wandb_kp.finish()
        
        logger.info("\nResults (With Keypoints):")
        logger.info(str(metrics_with_kp))
    
    # Cleanup
    if temp_gt_path.exists():
        temp_gt_path.unlink()
        
    return True


def run_detection_data_prep(output_dir="data/detector/coco", yaml_path="data/detector/dogs_detection.yaml"):
    """Runs the data preparation and conversion for the detection model."""
    logger.info("\n" + "-" * 60)
    logger.info("STARTING DETECTION DATA PREPARATION")
    logger.info("-" * 60)

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


def run_detection_pipeline(output_dir="data/detector/coco", yaml_path="data/detector/dogs_detection.yaml"):
    """Runs the full detection pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING DETECTION PIPELINE")
    logger.info("=" * 60)

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
    trainer.update_config(
        data=yaml_path, epochs=epochs, batch=batch_size, imgsz=imgsz
    )
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
    os.rename(exported_path, ONNX_DETECTOR_PATH)
    logger.info(f"Detector ONNX exported to: {ONNX_DETECTOR_PATH}")


def run_embedding_data_prep():
    """Runs the data preparation step for the embedding model."""
    logger.info("\n" + "-" * 60)
    logger.info("STARTING EMBEDDING DATA PREPARATION")
    logger.info("-" * 60)
    
    dataset_converter = EmbeddingDatasetConverter(
        source_path=DATA_CONFIG["DOGFACENET_PATH"],
        output_train_json=DATA_CONFIG["TRAIN_JSON_PATH"],
        output_val_json=DATA_CONFIG["VAL_JSON_PATH"]
    )
    dataset_converter.convert()

def run_embedding_pipeline():
    """Runs the full embedding pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING EMBEDDING PIPELINE")
    logger.info("=" * 60)

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
        "training_config": TRAINING_CONFIG,
        "data_config": DATA_CONFIG,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Datasets
    train_dataset = DogIdentityDataset(
        json_path=PROJECT_ROOT / DATA_CONFIG["TRAIN_JSON_PATH"],
        img_size=DATA_CONFIG["IMG_SIZE"],
        is_training=True,
    )
    val_dataset = DogIdentityDataset(
        json_path=PROJECT_ROOT / DATA_CONFIG["VAL_JSON_PATH"],
        img_size=DATA_CONFIG["IMG_SIZE"],
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["HARDWARE_WORKERS"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DATA_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["HARDWARE_WORKERS"],
    )

    # Create Model
    model = DogEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=train_dataset.num_classes,
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
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
        warmup_epochs=TRAINING_CONFIG["WARMUP_EPOCHS"],
        full_epochs=TRAINING_CONFIG["FULL_TRAIN_EPOCHS"],
        head_lr=TRAINING_CONFIG["HEAD_LR"],
        backbone_lr=TRAINING_CONFIG["BACKBONE_LR"],
        full_lr=TRAINING_CONFIG["FULL_TRAIN_LR"],
        patience=TRAINING_CONFIG["EARLY_STOPPING_PATIENCE"],
    )
    logger.info(f"Embedding training complete. Best model: {best_model_path}")

    # 3. Evaluate Best Model and Export
    run_embedding_export(best_model_path, val_loader, device, train_dataset.num_classes)


def run_embedding_export(model_path, val_loader, device, num_classes):
    """Evaluates the best model and exports it to ONNX."""
    
    # Re-instantiate model for evaluation and export
    model = DogEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=num_classes,
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
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
    export_model = DogEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=num_classes,
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
    )
    export_model.load_state_dict(torch.load(model_path, map_location=export_device))
    export_model.to(export_device)
    export_model.eval()

    dummy_input = torch.randn(
        1, 3, DATA_CONFIG["IMG_SIZE"], DATA_CONFIG["IMG_SIZE"], device=export_device
    )
    
    ONNX_EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(ONNX_EMBEDDING_PATH),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(f"Embedding ONNX exported to: {ONNX_EMBEDDING_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Master Training Script for Animal ID Pipeline")
    parser.add_argument("--skip-detection", action="store_true", help="Manually skip detection pipeline")
    parser.add_argument("--skip-embedding", action="store_true", help="Manually skip embedding pipeline")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip full pipeline benchmark")
    parser.add_argument("--skip-trained", action="store_true", help="Automatically skip training a model if its ONNX file already exists.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging during benchmark")
    parser.add_argument("--tag", type=str, default=None, help="Tag for WandB benchmark run")
    args = parser.parse_args()

    # --- Detection Pipeline Execution ---
    if args.skip_detection:
        logger.info("Manually skipping detection pipeline.")
    elif args.skip_trained and ONNX_DETECTOR_PATH.exists():
        logger.info(f"Skipping detection pipeline as {ONNX_DETECTOR_PATH} exists (--skip-trained).")
    else:
        run_detection_pipeline()

    # --- Embedding Pipeline Execution ---
    if args.skip_embedding:
        logger.info("Manually skipping embedding pipeline.")
    elif args.skip_trained and ONNX_EMBEDDING_PATH.exists():
        logger.info(f"Skipping embedding pipeline as {ONNX_EMBEDDING_PATH} exists (--skip-trained).")
    else:
        run_embedding_pipeline()
    
    # --- Benchmark Execution ---
    if not args.skip_benchmark:
        # Check if ONNX files are present for benchmarking
        can_benchmark = ONNX_DETECTOR_PATH.exists() and ONNX_EMBEDDING_PATH.exists()
        if not can_benchmark:
             logger.warning(f"Skipping benchmark because required ONNX models not found.")
        else:
            run_full_pipeline_benchmark(
                no_wandb=args.no_wandb,
                tag=args.tag
            )
    
    # --- Final Summary ---
    # This section is only reached if all steps complete without error.
    print("\n" + "=" * 60)
    print("MASTER SCRIPT COMPLETE")
    print("=" * 60)
    print("All pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
