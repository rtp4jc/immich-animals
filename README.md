# Animal Identification for Immich

Detects and identifies individual animals in photos, mirroring Immich's people pipeline (detect → crop → embed → cluster → user confirms). Initial focus is dogs.

## Just want it working in Immich?

**[sidecar/](sidecar/)** adds your dogs to Immich's People tab. One container,
one setting, no fork of Immich. Start there — the rest of this README is about
training the models.

## Model card

### Embedding
| Embedder | Top-1 | Top-5 | MRR | mAP | TAR@FAR=1% | Params | CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Tiny (default) | 0.953 | 0.983 | 0.967 | 0.624 | 0.826 | 28.7 M | 29 ms |
| ResNet50 (`buffalo_m` and below) | 0.857 | 0.945 | 0.896 | 0.494 | 0.621 | 25.1 M | 23 ms |

Every variation is in [`embedding_benchmarks.csv`](embedding_benchmarks.csv)

**Test set**: DogFaceNet held-out split, 1078 queries over 149
identities. Tight, mostly frontal face crops.

### Detection

80% of in-the-wild dog photos were observed to get a detection with 99% being detected when the dog was a significant portion of the frame. False positives were observed on near-neighbor
quadrupeds: 40% of wolves and dingoes, 14% of cats.

**Test set**: YOLO11n, held out on 1524 photos of 121 individual dogs plus 450
dog-free photos with a min detection score of 30%.

## Architecture

`AnimalPipeline` in `animal_id/pipeline/animal_pipeline.py` runs three ONNX models:

1. **Detector** (`ONNXDetector`): YOLO11n, outputs animal bounding boxes.
2. **Keypoint estimator** (`ONNXKeypoint`): YOLO11n-pose, finds 4 facial landmarks (eyes, nose, throat) to refine the crop. **Off by default**: benchmarks are better without it.
3. **Embedder** (`ONNXEmbedding`): ConvNeXt-Tiny + ArcFace (a ResNet50 variant also ships), 512-dim L2-normalised vectors compared by cosine similarity, matching Immich's face-embedding contract.

`pipeline/onnx_models.py` wraps the three models and `pipeline/models.py` defines the `DetectionModel`, `KeypointModel`, and `EmbeddingModel` Protocols they satisfy.

### Package layout

```
animal_id/
├── pipeline/        # AnimalPipeline orchestrator + ONNX wrappers
├── data/            # Sample contract, source adapters, exports, contact sheets
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

Every dataset is parsed by an adapter in `animal_id/data/sources/` into `Sample`s
(image path, source, licence, and boxes carrying a class and optional identity),
cached as `data/manifests/<source>.jsonl`. Exports in `animal_id/data/exports/`
write them in each trainer's format: `torch_identity.py` for the embedder
(`DataConfig.sources`) and `yolo.py` for the detector (`DETECTION_*` in
`train_master.py`, which picks the sources, classes and negative caps).

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

`embedding` and `all` also take `--backbone`, `--head`, `--seed` and `--epochs`, so a run is fully specified without editing `embedding/config.py`. The run directory's `config.json` records them, and `export-embedding` reads it back to rebuild the right architecture.

```bash
# Immich-like identification: cluster embeddings and score against ground truth
uv run python scripts/18_run_identification.py --split val --num-images 200

# Explore embeddings and clusters visually in FiftyOne
uv run python scripts/19_explore_fiftyone.py --split val --num-images 200

# Backbone ablation (see .planning/6-25-2026-embedding-backbone-ablation/plan.md)
uv run python scripts/run_ablation.py --backbone convnext_tiny --mode probe
uv run python scripts/summarize_ablation.py --mode finetune   # apply the decision rule

# Train and export the production model (folds val in; writes models/onnx/embedding.onnx)
uv run python scripts/train_final.py --backbone convnext_tiny --include-val
```

Backbone weights carry their own licence, which gates deployment separately from
accuracy: `BackboneSpec.license_tier` records it and `summarize_ablation.py`
treats anything non-permissive as a reference ceiling rather than a candidate.

Each exported embedder writes a `.json` sidecar beside it holding the
preprocessing recipe, test metrics, ONNX parity and the DBSCAN `eps` to cluster
at. That `eps` is swept per model (it depends on embedding geometry, so it
cannot be inherited across a backbone swap) and is selected on the test split,
so the clustering scores are best-case; retrieval metrics involve no tuning.

Benchmarks log to Weights & Biases by default; pass `--no-wandb` to disable. In the W&B dashboard, plot metrics like `top_5_accuracy` or `tar_at_far_0_01` against wall time and group by `use_keypoints`. Missed detections and wrong matches show up under Media.

### Helper scripts

| Script | Purpose |
|---|---|
| 02 | Inspect COCO- and YOLO-format exports (keypoint and detection) |
| 04, 05, 12 | Keypoint data prep, training, ONNX export |
| `data.py` | `parse` sources into manifests; `inspect` prints a per-source summary and writes a contact sheet to `outputs/data_inspect/` |
| 09 | Validate embeddings |
| 14, 15 | Model I/O inspection, two-stage inference |
| 16, 17 | Immich container integration (needs a local Immich fork at `immich-clone/`, not included) |
| 18, 19 | Identification clustering and FiftyOne explorer |
| `run_ablation.py` | Embedding backbone ablation harness (resumable; `--dry-run` shows the plan, `--force` re-runs a cell) |
| `ablation_status.py` | Regenerates `outputs/ablation/STATUS.md` from the result CSVs |
| `summarize_ablation.py` | Aggregates seeds and applies the licence/ONNX/latency decision rule |
| `measure_latency.py` | Cost axis for every backbone on one instrument (torch + ONNX Runtime) |
| `stage_d_validate.py` | Deploy gate: exports each finalist and checks the 512-d L2 contract |
| `train_final.py` | Trains and exports the production model, with eps sweep and provenance sidecar |

Exported models land in `models/onnx/` as `detector.onnx`, `keypoint.onnx`, and
`embedding.onnx`, each embedder alongside a `.json` sidecar recording its
backbone, preprocessing recipe, test metrics and DBSCAN `eps`. The embedder is
trained on ImageNet-normalised input while the YOLO stages take raw `[0, 1]`, so
`ONNXEmbedding` normalises and the others do not. `copy_models.sh` and `reload_immich.sh` push them into the `immich-clone/` fork.

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
After adding a source, check it parsed correctly with
`uv run python scripts/data.py inspect <source>`.

| Dataset | Location | Used for |
|---|---|---|
| [COCO 2017](https://cocodataset.org/#download) | `data/coco/images/{train2017,val2017}/` | Detection |
| [DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet#dataset) | `data/dogfacenet/DogFaceNet_224resized/`, `data/dogfacenet/DogFaceNet_alignment/` | Identity embedding |
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | `data/stanford_dogs/images/`, `data/stanford_dogs/annotation/` | Detection, keypoints |
| [StanfordExtra](https://www.kaggle.com/datasets/ollieboyne/stanfordextra-dogs-dataset) | `data/stanford_dogs/stanford_extra_keypoints.json` | Keypoint labels |
| [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | `data/oxford_pets/images/`, `data/oxford_pets/annotations/` | Detection negatives (cats) |

## Notice of AI usage

This project does rely heavily on AI code generation capabilities. I'm somewhat
hesitant to admit that because I'm not just vibe coding this project and turning
my brain off. I do care about the quality of this code and consistently work
to ensure maintain a high quality bar on all produced code via tests, manual
validation, and actual manual reviews of the code. That being said, I have not
read every line of code in this code base and, right now, I don't have time to.
This is currently in a prototyping/beta stage and I indend to do full due
diligence when/if this gets merged into Immich. To make this easier on myself,
I'm consistently having AI sweep for redundant code, bugs, and improper
organization to keep it DRY and minimal.

### How I'm combatting AI slop

In addition to the audits and processes listed above, I'm combatting AI slop
by making each step in the process as small as possible and human verifiable.

For example, in dataset preparation there are several datasets we are pulling
from. If bounding boxes are parsed incorrectly, you'll be training the model
to find random parts of the background, not the animal. Before just telling
an LLM to add this new dataset to the training mix and vibing it out on the
pytorch metrics, I have it parse the results, and visualize several examples.
I go through each to ensure the bounding box, label, and identity looks
correct. You can see how I've done that in
[scripts/02_inspect_detection_datasets.py](scripts/02_inspect_detection_datasets.py)
and
[scripts/04_prepare_keypoint_data.py](scripts/04_prepare_keypoint_data.py).

This is one example, but I'm continually looking for ways to make sure I 1)
understand this code and technology deeply and 2) trust it enough to share
with friends and family.
