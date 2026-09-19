# Animal Identification for Immich

Detects and identifies individual animals in photos, mirroring Immich's people pipeline (detect → crop → embed → cluster → user confirms). Initial focus is dogs.

## Just want it working in Immich?

**[sidecar/](sidecar/)** adds your dogs to Immich's People tab. One container,
one setting, no fork of Immich. Start there — the rest of this README is about
training the models.

## Architecture

`AnimalPipeline` in `animal_id/pipeline/animal_pipeline.py` runs three ONNX models:

1. **Detector** (`ONNXDetector`): YOLO11n, outputs animal bounding boxes.
2. **Keypoint estimator** (`ONNXKeypoint`): YOLO11n-pose, finds 4 facial landmarks (eyes, nose, throat) to refine the crop. **Off by default**: benchmarks are better without it.
3. **Embedder** (`ONNXEmbedding`): ResNet50 + ArcFace, 512-dim L2-normalised vectors compared by cosine similarity, matching Immich's face-embedding contract.

`pipeline/onnx_models.py` wraps the three models and `pipeline/models.py` defines the `DetectionModel`, `KeypointModel`, and `EmbeddingModel` Protocols they satisfy.

### Package layout

```
animal_id/
├── pipeline/        # AnimalPipeline orchestrator + ONNX wrappers
├── detection/       # YOLO detector training (Ultralytics)
├── keypoint/        # YOLO-pose training on Stanford Dogs keypoints
├── embedding/       # PyTorch embedding model: backbones.py, models.py (ArcFace head), losses.py, trainer.py
├── identification/  # Immich-like clustering (cosine DBSCAN) + cluster metrics
├── benchmark/       # MRR, top-k accuracy, TAR@FAR
├── tracking/        # Weights & Biases logger
└── common/          # constants.py (single source of truth for paths), datasets, seeding, shared YOLO converter
scripts/             # train_master.py, run_ablation.py, and numbered helper scripts (see below)
tests/               # unit/ and integration/
.planning/           # Dated design docs: production audit (2026-04) and backbone ablation plan (2026-06)
```

Each training subpackage follows the same pattern: `dataset_converter.py` (raw data → COCO), `yolo_converter.py` (COCO → YOLO) where relevant, and `trainer.py`.

## Setup

[mise](https://mise.jdx.dev/) manages the toolchain and [uv](https://docs.astral.sh/uv/) manages dependencies.

```bash
mise install      # Python 3.12 + uv
mise run setup    # uv sync → .venv from uv.lock, including dev deps
pre-commit install
```

Without mise, `uv sync` alone works and installs Python if needed.

mise auto-activates `.venv` in interactive shells. Anywhere it doesn't (CI, scripts, agents), prefix commands with `uv run`. Add dependencies with `uv add <pkg>` so `uv.lock` stays in sync.

**GPU**: on Linux, `uv sync` installs CUDA-enabled PyTorch from PyPI. Verify with `uv run python -c "import torch; print(torch.cuda.is_available())"`. For GPU ONNX inference, swap `onnxruntime` for `onnxruntime-gpu`.

**FiftyOne explorer** (optional): `uv sync --extra viz`.

## Usage

`scripts/train_master.py` is the entry point for training, export, and benchmarking. With no subcommand it runs detection, then embedding, then the benchmark.

```bash
uv run python scripts/train_master.py                          # everything
uv run python scripts/train_master.py all --skip-trained       # skip stages whose ONNX already exists
uv run python scripts/train_master.py detection                # prepare, train, export detector
uv run python scripts/train_master.py embedding                # prepare, train, export embedder
uv run python scripts/train_master.py benchmark --num-images 50 --tag baseline
```

Other subcommands: `detection-data`, `embedding-data`, `export-detector`, `export-embedding`. Keypoints are trained separately via scripts 04, 05, and 12.

```bash
# Immich-like identification: cluster embeddings and score against ground truth
uv run python scripts/18_run_identification.py --split val --num-images 200

# Explore embeddings and clusters visually in FiftyOne
uv run python scripts/19_explore_fiftyone.py --split val --num-images 200

# Backbone ablation (see .planning/6-25-2026-embedding-backbone-ablation/plan.md)
uv run python scripts/run_ablation.py --backbone convnextv2_tiny --mode probe
```

Benchmarks log to Weights & Biases by default; pass `--no-wandb` to disable. In the W&B dashboard, plot metrics like `top_5_accuracy` or `tar_at_far_0_01` against wall time and group by `use_keypoints`. Missed detections and wrong matches show up under Media.

### Helper scripts

| Script | Purpose |
|---|---|
| 02 | Inspect detection datasets |
| 04, 05, 12 | Keypoint data prep, training, ONNX export |
| 07, 09 | Visualise and validate embeddings |
| 14, 15 | Model I/O inspection, two-stage inference |
| 16, 17 | Immich container integration (needs a local Immich fork at `immich-clone/`, not included) |
| 18, 19 | Identification clustering and FiftyOne explorer |
| `run_ablation.py` | Embedding backbone ablation harness |

Exported models land in `models/onnx/` as `detector.onnx`, `keypoint.onnx`, and `embedding.onnx`. `copy_models.sh` and `reload_immich.sh` push them into the `immich-clone/` fork.

## Testing and CI

```bash
uv run pytest                          # everything
uv run pytest tests/unit/test_datasets.py
uv run pytest --cov=animal_id
uv run ruff check . && uv run ruff format .
```

Unit tests cover data loading, models, and converters. Integration tests run short end-to-end training for each stage. GitHub Actions runs ruff lint, ruff format check, and pytest on every push and PR to `main`.

## Data

Download and extract under `data/` (gitignored). Scripts guide preparation after download.

| Dataset | Location | Used for |
|---|---|---|
| [COCO 2017](https://cocodataset.org/#download) | `data/coco/images/{train2017,val2017}/` | Detection |
| [DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet#dataset) | `data/dogfacenet/DogFaceNet_224resized/`, `data/dogfacenet/DogFaceNet_alignment/` | Identity embedding |
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | `data/stanford_dogs/images/`, `data/stanford_dogs/annotation/` | Detection, keypoints |
| [StanfordExtra](https://www.kaggle.com/datasets/ollieboyne/stanfordextra-dogs-dataset) | `data/stanford_dogs/stanford_extra_keypoints.json` | Keypoint labels |
| [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | `data/oxford_pets/images/`, `data/oxford_pets/annotations/` | Additional detection data |
