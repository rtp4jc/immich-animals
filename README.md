# Animal Identification for Immich

Detects and identifies individual animals in photos, mirroring Immich's people pipeline (detect → crop → embed → cluster → user confirms). Dogs and cats so far.

## Just want it working in Immich?

**[sidecar/](sidecar/)** adds your dogs and cats to Immich's People tab. One container,
one setting, no fork of Immich. Start there — the rest of this README is about
training the models.

## Model card

Release 0.3.0, the first with cats. The better value in each column is bold.
The Embedding and Households tables come from `scripts/evaluate_release.py`,
with the metric definitions in `animal_id/identification/households.py`.

### Embedding
| Embedder | DogReID top-1 | Top-5 | MRR | TAR@FAR=1% | Cat top-1 | Cat TAR@FAR=1% | Params | CPU, 4 threads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DINOv2-B/14 + ArcFace, dogs and cats (0.3.0) | 0.512 | 0.758 | 0.625 | **0.762** | **0.962** | **0.837** | 87 M | 128 ms |
| DINOv2-B/14 + ArcFace, dogs (0.2.0) | **0.534** | **0.766** | **0.638** | 0.758 | 0.923 | 0.564 | 87 M | 128 ms |

**Test sets**: [DogReID-1553](https://doi.org/10.7910/DVN/LVTRLG) open-set test,
777 dogs filmed by their owners on phones; each query is matched against a
gallery from a different scene, cropped to the ground-truth box plus 10%. Cats:
123 held-out shelter cats from [Cat Individual Images](https://www.kaggle.com/datasets/timost1234/cat-individuals),
one photo per near-duplicate burst (667 photos), leave-one-out.

Retraining the 0.2.0 recipe with another seed moved DogReID top-1 by about 5
points, more than the gap between these rows (`.planning/10-2-2026-cats/`). The
0.3.0 seed was chosen on validation identities only.

### Households
| At the shipped Max Distance | Dogs in one correct person | Never grouped | Merged with another pet | Cats in one correct person | Merged |
| --- | --- | --- | --- | --- | --- |
| 0.3.0 | 46% | 34% | **8%** | **70%** | **1%** |
| 0.2.0 | **47%** | **27%** | 10% | 23% | 8% |

**Simulation**: 300 homes per half of the held-out pets, each with 1-4 own pets
(3-15 photos) and 20-200 one-off strangers, clustered the way Immich assigns
faces (`identification.cluster`, minFaces 3). A pet is in one correct person
when the person holding most of its photos is at least 90% that pet. Dogs use
DogReID and Max Distance 0.375 (0.2.0: 0.4); cats use the held-out cats above and 0.35. Every
pet is cropped from its labelled box, so this measures the embedder alone;
0.2.0's detector finds only 8% of cats in the first place.

### Detection
| Detector (conf >= 0.3) | DogReID dogs | Commons + MPDD dogs | Owner cats | Commons cats | Dog- and cat-free photos with a detection |
| --- | --- | --- | --- | --- | --- |
| YOLO11n, dogs and cats (0.3.0) | **0.953** | 0.876 | **0.990** | **0.912** | **0.234** |
| YOLO11n, dogs (0.2.0) | 0.947 | **0.882** | 0.083 | 0.192 | 0.255 |

**Test sets**: the DogReID-1553 open-set test frames (3,755, IoU >= 0.5); the
held-out shelter cats (2,550 photos, IoU >= 0.5 against YOLO11x boxes); and
`sidecar-validation`: 1,199 photos of 88 named dogs, 1,165 photos of cats (50
named), and 325 photos with neither. 0.2.0's cat "detections" are labelled dog.
Most false positives are other animals: 76% of wolf and fox photos, half the
rabbits and a third of the deer.

## Architecture

`AnimalPipeline` in `animal_id/pipeline/animal_pipeline.py` runs three ONNX models:

1. **Detector** (`ONNXDetector`): YOLO, outputs animal bounding boxes.
2. **Keypoint estimator** (`ONNXKeypoint`): YOLO-pose, finds 4 facial landmarks (eyes, nose, throat) to refine the crop. **Off by default**: benchmarks are better without it.
3. **Embedder** (`ONNXEmbedding`): a backbone from `embedding/backbones.py` trained with a margin head, 512-dim L2-normalised vectors compared by cosine similarity, matching Immich's face-embedding contract.

`pipeline/onnx_models.py` wraps the three models and `pipeline/models.py` defines the `DetectionModel`, `KeypointModel`, and `EmbeddingModel` Protocols they satisfy.

### Package layout

```
animal_id/
├── pipeline/        # AnimalPipeline orchestrator + ONNX wrappers
├── data/            # Sample contract, source adapters, exports, contact sheets
├── detection/       # YOLO detector training (Ultralytics)
├── keypoint/        # YOLO-pose training on Stanford Dogs keypoints
├── embedding/       # PyTorch embedding model: backbones.py, models.py, losses.py (margin heads), trainer.py
├── identification/  # Immich-style clustering + cluster metrics
├── benchmark/       # MRR, top-k accuracy, TAR@FAR
├── tracking/        # Weights & Biases logger
└── common/          # constants.py (single source of truth for paths), datasets, seeding, shared YOLO converter
scripts/             # train_master.py, run_ablation.py, and numbered helper scripts (see below)
tests/               # unit/ and integration/
.planning/           # Dated design docs (audit, backbone ablation, sidecar, data contract)
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
uv run python scripts/train_final.py --backbone dinov2_b --include-val
```

Backbone weights carry their own licence, which gates deployment separately from
accuracy: `BackboneSpec.license_tier` records it and `summarize_ablation.py`
treats anything non-permissive as a reference ceiling rather than a candidate.

Each exported embedder writes a `.json` sidecar beside it holding the
preprocessing recipe, test metrics, ONNX parity and the clustering `eps` to cluster
at. That `eps` is swept per model (it depends on embedding geometry, so it
cannot be inherited across a backbone swap) and is selected on the test split,
so the clustering scores are best-case; retrieval metrics involve no tuning.

Benchmarks log to Weights & Biases by default; pass `--no-wandb` to disable. In the W&B dashboard, plot metrics like `top_5_accuracy` or `tar_at_far_0_01` against wall time and group by `use_keypoints`. Missed detections and wrong matches show up under Media.

### Helper scripts

| Script | Purpose |
|---|---|
| 02 | Inspect COCO- and YOLO-format exports (keypoint and detection) |
| 04, 05, 12 | Keypoint data prep, training, ONNX export |
| `data.py` | `fetch` / `prepare` a source's images, `parse` sources into manifests; `inspect` prints a per-source summary and writes a contact sheet to `outputs/data_inspect/` |
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
| `evaluate_release.py` | Model-card Embedding and Households numbers for shipped ONNX embedders |

Exported models land in `models/onnx/` as `detector.onnx`, `keypoint.onnx`, and
`embedding.onnx`, each embedder alongside a `.json` sidecar recording its
backbone, preprocessing recipe, test metrics and clustering `eps`. The embedder is
trained on ImageNet-normalised input while the YOLO stages take raw `[0, 1]`, so
`ONNXEmbedding` normalises and the others do not. `copy_models.sh` and `reload_immich.sh` push them into the `immich-clone/` fork.

Each release's files live in `models/onnx/<version>/release/` (what the sidecar
image ships, matching `sidecar/SHA256SUMS` for the current version), with any
candidates that were compared beside it, e.g. `models/onnx/0.3.0/seed_13/`.

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
| [DogReID-1553](https://doi.org/10.7910/DVN/LVTRLG) | `data/dogreid/images/` (unzip `Images.zip`, rename to lowercase `images`), `data/dogreid/{splits,bounding_boxes,breeds}.tab` | Detection and identity embedding (owner phone photos); only the open-set `train` frames are loaded, so its query/gallery test stays held out |
| [MPDD](https://doi.org/10.17632/v5j6m8dzhv.1) | `data/mpdd/MPDD/pytorch/` | Identity embedding (whole-body); its test identities also go into `sidecar-validation` |
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | `data/stanford_dogs/images/`, `data/stanford_dogs/annotation/` | Detection, keypoints |
| [StanfordExtra](https://www.kaggle.com/datasets/ollieboyne/stanfordextra-dogs-dataset) | `data/stanford_dogs/stanford_extra_keypoints.json` | Keypoint labels |
| [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | `data/oxford_pets/images/`, `data/oxford_pets/annotations/` | Detection (cats and dogs, boxed from the trimaps) |
| [Cat Individual Images](https://www.kaggle.com/datasets/timost1234/cat-individuals) | unzip to `data/cat_individuals/cat_individuals_dataset/`, then `data.py prepare cat_individuals` | Cat detection and identity embedding; a quarter of the cats are held out as the benchmark |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | `data.py fetch open_images --max-per-class N` | Cat and dog detection (not in the default mix) |

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
