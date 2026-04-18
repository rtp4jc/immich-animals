# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**immich-animals** is an animal identification system that mirrors Immich's people-detection pipeline. It implements a 3-stage deep learning pipeline (detect → keypoints → embed) to identify individual animals in photos, with current focus on dogs. The end goal is to "hijack" Immich's facial recognition pipeline to work on pets.

## Commands

```bash
# Install (requires uv: https://docs.astral.sh/uv/getting-started/installation/)
uv venv --python 3.12 venv --seed
uv pip install -e .[dev]
venv/bin/python -m pre_commit install

# Lint / format
ruff check .
ruff format .

# Tests
python -m pytest tests/
python -m pytest tests/unit/test_datasets.py   # single file
python -m pytest --cov=animal_id tests/        # with coverage
```

### Training pipeline (numbered scripts, in order)
```bash
python scripts/01_prepare_detection_data.py    # COCO → YOLO format
python scripts/03_train_detection_model.py     # Train YOLO11n detector
python scripts/04_prepare_keypoint_data.py
python scripts/05_train_keypoint_model.py      # Train YOLO11n-pose
python scripts/06_prepare_embedding_data.py    # DogFaceNet → JSON
python scripts/08_train_embedding_model.py     # ResNet50 + ArcFace
python scripts/train_master.py                 # Orchestrates full pipeline

# Export to ONNX
python scripts/10_export_embedding_onnx.py
python scripts/11_export_detector_onnx.py
python scripts/12_export_keypoint_onnx.py

# Benchmark
python scripts/13_run_full_pipeline.py --num-images 50 --num-queries 5
python scripts/13_run_full_pipeline.py --no-wandb   # disable W&B
```

## Architecture

### Three-stage inference pipeline
`animal_id/pipeline/ambidextrous_axolotl.py` is the main orchestrator:
1. **Detection** (`ONNXDetector`) — YOLO11n, outputs bounding boxes
2. **Keypoints** (`ONNXKeypoint`) — YOLO11n-pose, estimates 4 facial landmarks for crop refinement (currently disabled by default — benchmarks show better results without it)
3. **Embedding** (`ONNXEmbedding`) — ResNet50 + ArcFace, 512-dim vectors for identity matching

`pipeline/onnx_models.py` wraps the three ONNX models. `pipeline/models.py` defines the `DetectionModel`, `KeypointModel`, and `EmbeddingModel` Protocol classes for loose coupling.

### Training modules
Each stage has its own subpackage with `trainer.py`, `dataset_converter.py`, and `yolo_converter.py`:
- `animal_id/detection/` — wraps Ultralytics YOLO training
- `animal_id/keypoint/` — YOLO-pose training on Stanford Dogs keypoints
- `animal_id/embedding/` — custom PyTorch: `models.py` (DogEmbeddingModel + ArcFace head), `backbones.py` (ResNet50/MobileNetV3/EfficientNet), `losses.py` (ArcFace/CosFace)

### Shared utilities
`animal_id/common/constants.py` is the single source of truth for all paths (`PROJECT_ROOT`, `MODELS_DIR`, `DATA_DIR`, `ONNX_DIR`). `animal_id/benchmark/evaluator.py` computes MRR, top-k accuracy, and TAR@FAR metrics.

### Key paths
| Path | Purpose |
|------|---------|
| `models/onnx/` | Exported ONNX models consumed by inference pipeline |
| `data/dogfacenet/` | Identity training data (DogFaceNet 224px) |
| `data/coco/` | Detection training data |
| `data/stanford_dogs/` | Keypoint training data |
| `outputs/` | Benchmark results and visualizations |
| `runs/` | Training logs |

## Immich Integration

The goal is to replace Immich's face recognition with animal recognition. This requires:
1. Immich source (fork with `rtp4jc/hijack` branch)
2. Trained ONNX models copied via `copy_models.sh`
3. Custom Docker Compose: `docker-compose.dogs.yml`

See `REPLICATION.md` for full instructions. Scripts 16–17 are Immich integration tests.

## CI

GitHub Actions (`.github/workflows/python-package-conda.yml`) runs on push/PR to main: ruff lint → ruff format check → pytest, using conda + Python 3.12.
