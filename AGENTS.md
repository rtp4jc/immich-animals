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

## Working in a Git Worktree

This repo uses git worktrees for isolated feature branches. When Claude Code opens a worktree the working directory is the worktree path, **not** the repo root. Key facts to avoid confusion:

- **The venv lives at the repo root**, not inside the worktree. Use its absolute path:
  ```bash
  /mnt/e/Code/GitHub/immich-animals/venv/bin/python -m pytest tests/
  /mnt/e/Code/GitHub/immich-animals/venv/bin/ruff check .
  ```
- **The venv is shared** across all worktrees. Installing a package via `uv` targets the single shared venv — it applies everywhere.
- **Installing deps from the worktree's pyproject.toml** requires pointing uv at the repo root venv explicitly:
  ```bash
  uv pip install <package> --python /mnt/e/Code/GitHub/immich-animals/venv/bin/python
  # or to sync all deps from worktree's pyproject.toml:
  uv pip install -e "<worktree_path>/.[dev]" --python /mnt/e/Code/GitHub/immich-animals/venv/bin/python
  ```
  Running `uv pip install -e .[dev]` from inside the worktree will pick up the **parent repo's** `pyproject.toml`, not the worktree's.
- **Git commands** work normally from the worktree root — `git status`, `git add`, `git commit`, `git push` all operate on the worktree branch.
- **`CLAUDE.md` commands** like `venv/bin/python ...` are relative to the repo root, not the worktree. Prefix them with the absolute venv path shown above when running from inside a worktree.

## Architecture

### Three-stage inference pipeline
`animal_id/pipeline/animal_pipeline.py` is the main orchestrator (`AnimalPipeline` class):
1. **Detection** (`ONNXDetector`) — YOLO11n, outputs bounding boxes
2. **Keypoints** (`ONNXKeypoint`) — YOLO11n-pose, estimates 4 facial landmarks for crop refinement (currently disabled by default — benchmarks show better results without it)
3. **Embedding** (`ONNXEmbedding`) — ResNet50 + ArcFace, 512-dim vectors for identity matching

`pipeline/onnx_models.py` wraps the three ONNX models. `pipeline/models.py` defines the `DetectionModel`, `KeypointModel`, and `EmbeddingModel` Protocol classes for loose coupling.

### Training modules
Each stage has its own subpackage with `trainer.py`, `dataset_converter.py`, and `yolo_converter.py`:
- `animal_id/detection/` — wraps Ultralytics YOLO training
- `animal_id/keypoint/` — YOLO-pose training on Stanford Dogs keypoints
- `animal_id/embedding/` — custom PyTorch: `models.py` (AnimalEmbeddingModel + ArcFace head), `backbones.py` (ResNet50/MobileNetV3/EfficientNet), `losses.py` (ArcFace/CosFace)

### Shared utilities
`animal_id/common/constants.py` is the single source of truth for all paths (`PROJECT_ROOT`, `MODELS_DIR`, `DATA_DIR`, `ONNX_DIR`). `animal_id/benchmark/evaluator.py` computes MRR, top-k accuracy, and TAR@FAR metrics.

## CI

GitHub Actions (`.github/workflows/python-package-conda.yml`) runs on push/PR to main: ruff lint → ruff format check → pytest, using conda + Python 3.12.
