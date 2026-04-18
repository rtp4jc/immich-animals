# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [README.md](README.md) for project overview, setup, training pipeline, and data requirements.

## Commands

```bash
# Lint / format
venv/bin/ruff check .
venv/bin/ruff format .

# Tests (always use venv/bin/ prefix — do not rely on an activated venv)
venv/bin/python -m pytest tests/
venv/bin/python -m pytest tests/unit/test_datasets.py   # single file
venv/bin/python -m pytest --cov=animal_id tests/        # with coverage
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

## CI

GitHub Actions (`.github/workflows/python-package-conda.yml`) runs on push/PR to main: ruff lint → ruff format check → pytest, using conda + Python 3.12.
