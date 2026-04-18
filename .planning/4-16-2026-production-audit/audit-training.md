# Training Code Audit — immich-animals

Auditor's deep-dive evaluating whether the training code under `animal_id/{detection,keypoint,embedding}/`, `scripts/`, and related modules is suitable for adoption into the Immich open-source organization.

Scope: training only. Inference (`animal_id/pipeline/`) and benchmark correctness are out of scope except where they cross into training.

---

## Executive Summary

**Verdict: not yet production-grade, but salvageable.** The repository already knows what "good" looks like — it has a package layout, a master orchestrator, W&B hooks, a pre-commit config, and a CI job running ruff + pytest. What's missing is the *discipline* ML-maintainer-grade code requires: no seeding/determinism, no hyperparameter sweep surface, W&B is only plumbed into the pipeline benchmark and not into the training loop, and the "central" embedding config (`animal_id/embedding/config.py:11`) ships learning rates of `1e-8`/`1e-9` — a strong indicator the last human left it in a debugging state, not a pristine baseline. Dataset code still contains `print()` and `try: ...; return self.__getitem__((idx+1) % len(self))`-style recursion on corrupt images (`animal_id/common/datasets.py:86`) that silently skews the sampled distribution.

The good news: scope is small, abstractions mostly exist, and almost every gap can be closed by ~1–3 day PRs. The bad news: the current code won't enable ablations or third-party reproduction without an engineer babysitting it, which is not what a project like Immich wants to inherit.

Everything below is cross-referenced with `path:line` so it can be acted on.

---

## Findings by Category

### 1. Reproducibility

**What exists.** Individual data-preparation scripts seed a `random.seed(42)` before the train/val split: `animal_id/detection/dataset_converter.py:270`, `animal_id/keypoint/dataset_converter.py:211`, `animal_id/embedding/dataset_converter.py:78`. Validation pair sampling seeds numpy: `animal_id/benchmark/metrics.py:117-118`.

**What's missing.**
- **No global training-run seeding.** Neither `animal_id/embedding/trainer.py` nor any script calls `torch.manual_seed`, `torch.cuda.manual_seed_all`, or `np.random.seed` before instantiating the model or DataLoader. The `DataLoader` in `scripts/train_master.py:357-368` has `shuffle=True` with no `generator=` and no `worker_init_fn`, so batch order and augmentation (`animal_id/common/datasets.py:42-60`) are non-reproducible across runs. Two identical invocations of `python scripts/train_master.py` *cannot* produce the same embedding model. Grep confirms: `torch.manual_seed`/`cudnn.deterministic` do not appear anywhere in the tree.
- **No deterministic CUDA flags** (`torch.backends.cudnn.deterministic`, `torch.use_deterministic_algorithms`). The YOLO trainers (`animal_id/detection/trainer.py:27-48`, `animal_id/keypoint/trainer.py:29-49`) do not pass Ultralytics' `seed=` kwarg, so YOLO's internal seed is left as its default.
- **Split stability across reruns of data prep is weak.** The embedding split in `animal_id/embedding/dataset_converter.py:76-91` shuffles identities with seed 42 *but then splits by order-of-insertion into `val_data`*. If `os.listdir` returns identity folders in a different order (filesystem-dependent: ext4 vs. NTFS vs. Docker overlay), the identity list differs before shuffling, so the split differs — this is a subtle reproducibility hazard. The same pattern appears in `animal_id/detection/dataset_converter.py:266-287`.
- **No dataset versioning / checksum.** Data is pulled from DogFaceNet, Stanford Dogs, COCO and Oxford-IIIT Pets, but nothing records the version / zip hash / download date. A user re-running the pipeline in 6 months has no way to know whether they trained on the same data.
- **No git SHA / config dump in the run directory (partial).** `scripts/train_master.py:333-340` writes a `config.json` with timestamp + `TRAINING_CONFIG` + `DATA_CONFIG`, but no git hash, no `pip freeze`, no hardware info, no environment name. If a metric regresses later, the run is forensically opaque.
- **Dependency floors are too loose** (`pyproject.toml:13-29`): `torch>=2.0.0`, `ultralytics>=8.0.0`, etc. There's no lock file and no `requirements.txt`. Ultralytics in particular ships breaking changes in minor releases; without a pin, future users will train a different architecture under the same script.
- **`environment.yml` is near-empty** (`environment.yml:1-10`) — just `python=3.12` + `pip install -e .`. CI therefore just resolves `>=` floors on every run; identical PRs months apart can run against different toolchains.

### 2. Configuration Management

**What exists.** `animal_id/embedding/config.py:11-32` centralizes embedding hyperparameters into `TRAINING_CONFIG` / `DATA_CONFIG` dicts. Each CLI script (e.g. `scripts/05_train_keypoint_model.py:18-34`) has argparse for `epochs/batch/imgsz`. `animal_id/common/constants.py` centralizes paths.

**What's missing.**
- **No unified config system.** There are three config surfaces: (a) the `TRAINING_CONFIG` dict, (b) the YOLO trainers' `_get_default_config()` dicts (`animal_id/detection/trainer.py:27-48`, `animal_id/keypoint/trainer.py:29-49`), (c) argparse overrides. They share no schema, no validator, no way to dump a single "this is what I ran" YAML. A sweep across all three is impossible without editing source.
- **Hardcoded hyperparameters scattered.**
  - `animal_id/embedding/trainer.py:26`: `label_smoothing=0.1` hardcoded in the trainer constructor, not in `TRAINING_CONFIG`.
  - `animal_id/embedding/trainer.py:58`: `max_norm=1.0` gradient clip.
  - `animal_id/embedding/trainer.py:211`: `warmup_epochs_phase2 = 20` is a magic number *inside* the `train()` method — completely invisible from `TRAINING_CONFIG`.
  - `animal_id/embedding/trainer.py:212-222`: `start_factor=0.01`, cosine T_max, milestones all hardcoded.
  - `animal_id/embedding/losses.py:49`: ArcFace `s=30.0, m=0.50` defaults are hardcoded; there is no way to run a `m=0.3` experiment without editing code.
  - `animal_id/common/datasets.py:42-60`: every augmentation parameter (`brightness=0.2`, `RandomRotation(degrees=15)`, `RandomErasing(p=0.1)`) is hardcoded in the dataset class.
  - `animal_id/detection/dataset_converter.py:298, 310`: `num_negatives=15000` for COCO train and `2000` for COCO val are hardcoded call-site values.
- **`TRAINING_CONFIG` LRs look stale / misconfigured.** `animal_id/embedding/config.py:19-22` has `HEAD_LR=1e-8`, `BACKBONE_LR=1e-9`, `FULL_TRAIN_LR=1e-9`. With label smoothing 0.1 and batch 32 on ArcFace, 1e-9 is effectively a no-op for the backbone. The inline comments ("Back to very conservative", "Extremely small") suggest the user left them after manual fiddling. A maintainer-facing config should have tuned defaults plus a pointer to the sweep that produced them.
- **No config composition / overrides.** There is no Hydra / OmegaConf, no `--config path.yaml`, no nested override. Swapping from DogFaceNet to another dataset requires editing `animal_id/embedding/config.py:28` directly.
- **Relative paths as string literals.** `DATA_CONFIG["TRAIN_JSON_PATH"] = "data/identity_train.json"` — this only works when cwd is the repo root. Script 08 runs fine from root, but integration tests (`tests/integration/test_embedding_pipeline.py`) work around this by constructing absolute paths elsewhere.

### 3. Experiment Tracking (W&B)

**What exists.** A well-structured `WandBLogger` in `animal_id/tracking/wandb_logger.py` that supports grouping, tagging, config logging, failure-image logging.

**What's missing.**
- **W&B is only wired into the pipeline benchmark, not into training.** Grep confirms `wandb.log` / `wandb.init` appear only in `wandb_logger.py`. `EmbeddingTrainer.train()` (`animal_id/embedding/trainer.py:167-245`) does not call W&B at all — it just writes `training_metrics.json` at the end (line 233). There is no per-epoch train/val loss, learning-rate, gradient-norm, epoch-time, or best-metric curve going to W&B during training.
- **YOLO trainers inherit Ultralytics' W&B/TensorBoard defaults**, but the config (`animal_id/detection/trainer.py:27-48`) never sets `project=` or `name=` to match the W&B project used elsewhere, and doesn't set `plots=` / `exist_ok=`. Experiments from the three stages can't be cleanly compared in one W&B project.
- **No artifact logging.** Best checkpoints, ONNX files, and the data-prep JSON manifests are never uploaded as `wandb.Artifact`. There's no lineage from "this benchmark number" back to "this exact `best_model.pt`".
- **Run naming is timestamp-only.** `scripts/train_master.py:326-328` produces `20250907_155221_resnet50` directories, but doesn't set the W&B run name, so W&B auto-names runs incoherently.
- **Config logged to W&B is shallow.** In `run_full_pipeline_benchmark` (`scripts/train_master.py:148-153`) only `{num_images, include_additional, dataset_size, pipeline}` is logged. None of the model-side hyperparameters (ArcFace margin, embedding dim, backbone, training LR) reach W&B.
- **`wandb.init(reinit=True)`** at `animal_id/tracking/wandb_logger.py:55-62` suggests the logger is designed for multiple runs per process. That's fine for the benchmark but fragile — if `start()` fails, `log_metrics` silently no-ops (`:77-78`), making it hard to notice a run lost telemetry.

### 4. Checkpointing

**What exists.** `EmbeddingTrainer.save_checkpoint` (`animal_id/embedding/trainer.py:84-102`) writes `latest_checkpoint.pt` and `best_model.pt`. Global-vs-per-phase best tracking is implemented (`:131-147`).

**What's missing.**
- **No resume-from-checkpoint.** `EmbeddingTrainer.train()` always starts fresh; there is no `--resume` or load-and-continue path. `latest_checkpoint.pt` is written but never read anywhere — confirmed by grep. Interrupted training wastes all progress.
- **Best-model checkpoint is state-dict only, not a full training state.** `best_model.pt` at `animal_id/embedding/trainer.py:99` is `model.state_dict()` — no optimizer state, no scheduler state, no RNG state, no epoch counter. `latest_checkpoint.pt` at `:86-94` adds `epoch` and `val_mAP` but still omits optimizer/scheduler/RNG. Even if resume were implemented, it couldn't resume the LR schedule or Adam moments.
- **Saved config in the run dir does not include the model's `num_classes`.** `scripts/train_master.py:333-340` dumps `TRAINING_CONFIG`/`DATA_CONFIG` but not `train_dataset.num_classes`, which is needed to re-instantiate the ArcFace head. Users of script 10 (`scripts/10_export_embedding_onnx.py:48-64`) work around this by *re-reading the training JSON* to recompute `num_classes` — an O(tens of thousands) JSON load just to learn one integer, and brittle if the data changes between training and export.
- **YOLO checkpointing is delegated to Ultralytics** with `save_period=5` (`animal_id/detection/trainer.py:38`, `animal_id/keypoint/trainer.py:40`), which is fine, but the "best" selection criterion (mAP50-95 by default) is never made explicit in the project — two trainers use different defaults than the embedding trainer and nobody surfaces this.
- **ONNX export reproducibility is weak.** `scripts/train_master.py:435-453` exports with `opset_version=12, do_constant_folding=True` but uses `torch.onnx.export` rather than `dynamo_export`/`torch.onnx.export(..., dynamo=True)` and no `verify=True`. No post-export parity check (ONNX vs. PyTorch on the same batch) is performed — if ONNX silently diverges (e.g., due to an `F.normalize` edge case), nobody knows. The keypoint exporter (`scripts/12_export_keypoint_onnx.py:45`) references `YOLO._version` which is a private attribute and may break on Ultralytics upgrades.
- **Best-model selection criterion.** `EmbeddingTrainer` picks by mAP (`:116`). Reasonable, but TAR@FAR=0.1% is arguably more correct for an open-set identification / verification task. This should be configurable, and the trainer should log both.

### 5. Error Handling / Robustness

- **Corrupt-image fallback is a silent recursion that biases the distribution.** `animal_id/common/datasets.py:82-86`: on image-load exception, the dataset recurses on `(idx+1) % len(self)`, printing a warning. With many corrupt images this can a) cause stack overflow, b) bias sampling toward the neighbor of every corrupt image, c) hide a dataset problem behind `print(...)` that disappears into tqdm output.
- **No GPU OOM handling.** Neither trainer catches `torch.cuda.OutOfMemoryError` or attempts gradient-accumulation / batch-halving. On 24 GB hardware the defaults will OOM at ResNet50 batch 32 + img 224, and the user will get a raw traceback.
- **No handling of interrupt (Ctrl-C).** A `KeyboardInterrupt` mid-epoch leaves `latest_checkpoint.pt` from the *previous* epoch — acceptable — but the trainer doesn't save a "final" snapshot on signal, so the progress inside the interrupted epoch is lost.
- **Missing data is not detected early.** `animal_id/embedding/dataset_converter.py:31-35` prints a `[ERROR]` and `return`s — the master script (`scripts/train_master.py:298-309`) doesn't check the return and proceeds to read the JSON paths, which will then fail with a FileNotFoundError deep in the trainer. There is no top-level "validate all prerequisites" step.
- **Class imbalance is not addressed.** DogFaceNet filtered at `min_images_per_identity=5` (`animal_id/embedding/dataset_converter.py:17`) still leaves a long tail. There is no weighted sampler, no balanced-batch sampler (essential for ArcFace with large `m`), no log of per-class counts. Training will silently under-represent rare identities.
- **Empty batches.** The `DataLoader` uses default `drop_last=False`. ArcFace with batch size 1 will still run but computes a degenerate cosine update. Not disastrous, but a principled trainer should set `drop_last=True`.
- **`prerequisites` check is shallow.** `DetectionTrainer.verify_prerequisites` (`animal_id/detection/trainer.py:50-69`) checks CUDA availability but not disk space, not the data YAML existence, not whether the images in `train.txt` / `val.txt` actually exist.
- **Empty validation set is only partially handled.** `tests/unit/embedding/test_trainer.py:40-81` confirms the trainer *doesn't crash* on empty val metrics, but the patience counter logic in `animal_id/embedding/trainer.py:136-143` compares `0.0 > -1.0` and marks every epoch as "best", disabling early stopping silently.

### 6. Data Pipeline

- **DataLoader is unoptimized.** `scripts/train_master.py:357-368` does not pass `pin_memory`, `persistent_workers`, or `prefetch_factor`. On CUDA this is easily 10–20% throughput left on the table. Grep confirms these options appear nowhere.
- **Augmentation is bolted into `DogIdentityDataset`** (`animal_id/common/datasets.py:42-71`). You cannot swap augmentation policy without subclassing, and the same class is used for train and val via the `is_training` flag. Validation transforms are also coupled (same file, both branches). Proper structure: Albumentations / torchvision v2 transforms composed externally and injected.
- **Train/val leakage protection is identity-based (good) but count-based (brittle).** `animal_id/embedding/dataset_converter.py:86-91` stops *adding identities to val* once `len(val_data) >= val_target_count`. Because identities vary in size, the resulting val ratio can be far from `0.15` and is sensitive to identity order. Better: pre-sort identities by size, then bin-pack to hit the ratio.
- **No explicit caching / preprocessing.** DogFaceNet is loaded fresh each epoch through PIL → RandomCrop/ColorJitter/Rotation. For a ResNet50 at 224 px this is fine but leaves perf on the table — and the detection pipeline's `cache=False` (`animal_id/detection/trainer.py:39`) is set to false without comment.
- **Dataset converters are destructive.** `animal_id/detection/dataset_converter.py:27-30`, `animal_id/keypoint/dataset_converter.py:42-47` do `shutil.rmtree` on the output dir. A user who runs `01_prepare_detection_data.py` then edits the output loses everything silently. Should be guarded by a `--force` flag.
- **COCO-negative sampling is non-deterministic per run.** `animal_id/detection/dataset_converter.py:186-187` shuffles without a local seed; only the outer `random.seed(42)` at line 270 protects later shuffles, but this call happens *before* the outer seed is set (line 270 runs *after* `_load_coco_bbox_only`). The COCO negative subset therefore changes run-to-run. Confirmed by reading lines 253-312.

### 7. Code Quality Across Stages (DRY)

- **`DetectionTrainer` and `KeypointTrainer` are near-duplicate.** `animal_id/detection/trainer.py:16-104` vs. `animal_id/keypoint/trainer.py:16-111`: identical `verify_prerequisites`, `load_model`, `train`, `update_config`. The only meaningful differences are default config dict and the `yolo11n.pt` vs. `.yaml` model. A `BaseYoloTrainer` with two subclasses would save ~70 lines and ensure that any fix in one applies to both.
- **Two `find_latest_*` helpers** in `animal_id/common/utils.py:11-71` plus a near-identical glob in `animal_id/detection/validator.py:23-35`.
- **Two `_load_ground_truth`-style JSON loaders** — the embedding one at `animal_id/common/datasets.py:34-40` and the benchmark one at `animal_id/benchmark/evaluator.py:120-123` — fine, but each reinvents list comprehension/filtering.
- **No shared `DataConverter` ABC.** All three `dataset_converter.py` files reimplement the "clear output dir, build COCO dict, write train/val JSON" loop. Pulling this into a base class in `animal_id/common/` would make new dataset integrations (e.g. Asian Dogs, iWildCam) a 50-line job instead of a new module.
- **Script entry points are wrapper-thin but ad-hoc.** Scripts 01, 03, 06, 08 are 5-line shims over `scripts.train_master` (e.g. `scripts/01_prepare_detection_data.py`), while 04, 05, 09, 10, 11, 12 have their own argparse and their own logic. No consistent pattern for a new contributor.
- **Top-level scripts import from `scripts.train_master`**, which requires cwd=root + `scripts/__init__.py`. `scripts/__init__.py` does not exist (only `__pycache__` is visible in `ls`). This works by virtue of `conftest.py:1-8` adding the root to `sys.path` for pytest, but a plain `python scripts/01_prepare_detection_data.py` without `-m` or a `PYTHONPATH` will ImportError.

### 8. Ablation Ergonomics

This is the single biggest barrier to third-party use. See "Ablation Friction Walkthroughs" below. Short version:

- **Swap backbone**: one-line change in `animal_id/embedding/config.py:9`, but only among the three hardcoded options in `animal_id/embedding/backbones.py:18-21`. Adding a new backbone means editing the `BackboneType` enum + `get_backbone`, shipping a new feature-dim contract.
- **Swap loss**: ArcFace is hardwired in `animal_id/embedding/models.py:113`. CosFace and AdaFace aren't in `animal_id/embedding/losses.py`. The trainer's `self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)` (`animal_id/embedding/trainer.py:26`) also hardwires the outer loss — you cannot swap to pure triplet without rewriting `train_epoch`.
- **Swap augmentation**: edit `animal_id/common/datasets.py:42-60` inline. No config for it.
- **New dataset**: write a new `dataset_converter.py` from scratch (no base class). Edit `scripts/train_master.py:298-309`. Edit `DATA_CONFIG`.

### 9. Observability

- **Logging is inconsistent.** `animal_id/embedding/trainer.py` uses `print(...)` (e.g. `:150-156`). `scripts/train_master.py:74-80` uses `logging` module. `animal_id/detection/trainer.py` uses `print`. There's no project-wide logger, no `--log-level`, no file handler.
- **Progress indication is tqdm-only.** Good for interactive use, ugly in `tee`'d logs and useless in CI.
- **Diagnostics on failure are thin.** The ArcFace forward pass (`animal_id/embedding/losses.py:78-120`) does not guard against `NaN` in `target_cosine` — `torch.sqrt(1 - x**2)` for `|x| > 1` (numerical overflow) produces NaN silently. No loss.isnan() check in the training loop.
- **Training metrics JSON** (`animal_id/embedding/trainer.py:233-242`) is the only persisted training artifact besides checkpoints. No per-epoch images, no confusion-matrix dump, no sample embeddings T-SNE.
- **`evaluate_embedding_model`** (`animal_id/benchmark/metrics.py:181-214`) is called every validation epoch but does an O(N²) cosine-similarity for mAP. On a 5k-image val set this is workable; at 50k it will dominate epoch time. There is no flag to skip mAP and only compute TAR@FAR (which uses sampled pairs).

### 10. Test Coverage

**What's covered.**
- `tests/unit/test_datasets.py`: dataset init, `__getitem__`, augmentation randomness.
- `tests/unit/test_detection_converters.py`: Stanford XML parsing, COCO parsing.
- `tests/unit/test_keypoint_converters.py`: keypoint mapping + validation + process_image.
- `tests/unit/test_models.py`: forward-pass shapes, freeze/unfreeze.
- `tests/unit/test_identity_loader.py`: validation-data loader, additional identities.
- `tests/unit/embedding/test_trainer.py`: trainer handles empty val loader.
- `tests/unit/benchmark/test_metrics.py`: TAR@FAR edge cases.
- `tests/integration/test_detection_pipeline.py`, `test_keypoint_pipeline.py`, `test_embedding_pipeline.py`: 1-epoch end-to-end smoke tests (good — these are the most valuable).

**What's not covered.**
- **ArcFace correctness / numerical stability**: no test for the margin math in `animal_id/embedding/losses.py:78-120` (only a `__main__` block, which isn't run by pytest).
- **ONNX export parity**: no test that the exported ONNX produces the same embedding as the PyTorch model on a fixed input.
- **Reproducibility**: no test that two invocations with the same seed produce the same weights / same first-batch gradients.
- **Dataset split stability**: no test that the 90/10 split is deterministic given a fixed source structure.
- **Master script**: `scripts/train_master.py` is not unit-tested and only exercised transitively by the integration tests — which don't call `run_detection_pipeline` or `run_embedding_pipeline` (they call the trainer directly).
- **YOLO trainer default config**: `_get_default_config` values are never asserted — a drift in Ultralytics default would go unnoticed.
- **W&B logger**: `animal_id/tracking/wandb_logger.py` has no test, including no test that the `enabled=False` path is truly no-op.

Overall the testing tier is *shallow-broad*: present for every module, but light per module. The integration tests are the strongest part.

---

## Top 5 Production-Readiness Gaps

Ranked by (severity × breadth) / effort to fix.

1. **No training-run seeding → zero reproducibility.** Severity: high. Effort: 0.5 day. This is the biggest "can we reproduce any reported metric" blocker. Needs a `set_seed(seed)` utility, a DataLoader `generator`, `worker_init_fn`, `torch.use_deterministic_algorithms(True)` (behind a flag), and YOLO `seed=` plumbing. (`animal_id/embedding/trainer.py`, `animal_id/detection/trainer.py`, `scripts/train_master.py:357-368`.)

2. **Stale / misconfigured embedding LRs shipped as defaults.** Severity: high (any new user who runs `scripts/08_train_embedding_model.py` gets a model that barely trains). Effort: 0.25 day to restore; 1 day to also add a sweep that validates defaults. (`animal_id/embedding/config.py:19-22`.)

3. **W&B disconnected from training loop.** Severity: high (no visibility into training dynamics, no comparable runs, no artifacts). Effort: 1 day. Wire `WandBLogger` into `EmbeddingTrainer.train` for per-epoch metrics + grad norm + LR, upload `best_model.pt` as artifact, set run name to the timestamped directory name. Also pass `project=...` to Ultralytics trainers. (`animal_id/embedding/trainer.py`, `animal_id/detection/trainer.py:27-48`, `animal_id/keypoint/trainer.py:29-49`.)

4. **No resume-from-checkpoint + incomplete `latest_checkpoint.pt`.** Severity: medium-high (any interruption = restart from scratch). Effort: 1 day. Save optimizer+scheduler+RNG state into `latest_checkpoint.pt`, add a `resume_from: Optional[Path]` kwarg to `EmbeddingTrainer.train`, skeleton already in place at `animal_id/embedding/trainer.py:84-102`.

5. **Hardcoded hyperparameters blocking ablations.** Severity: medium (but HIGH for the Immich use case of "let us try CosFace / AdaFace"). Effort: 1–2 days. Move all magic numbers (ArcFace `s/m`, augmentation params, phase-2 warmup epochs, gradient clip, label smoothing) into a single `TrainingConfig` dataclass, read from a YAML, merged with CLI overrides. (`animal_id/embedding/config.py`, `animal_id/embedding/trainer.py:26, 58, 211-222`, `animal_id/embedding/losses.py:49`, `animal_id/common/datasets.py:42-60`.)

---

## Ablation Friction Walkthroughs

### Ablation A — "Try CosFace instead of ArcFace"

Expected: one CLI flag.

Actual changes required:

1. `animal_id/embedding/losses.py`: add `CosFaceLoss` class parallel to `ArcFaceLoss` (~30 lines).
2. `animal_id/embedding/models.py:113`: the line `self.head = ArcFaceLoss(embedding_dim, num_classes)` is hardcoded. Need to pass `loss_type` through `DogEmbeddingModel.__init__`, or inject a factory.
3. `scripts/train_master.py:371-375`: `DogEmbeddingModel(...)` call hardcodes nothing about loss — would need a new kwarg.
4. `animal_id/embedding/config.py`: add `LOSS_TYPE` key.

Total: edit 4 files, no test coverage to guide correctness. A CI job only knows the model instantiates.

### Ablation B — "Try AdaFace"

As above, plus:

1. AdaFace needs per-sample feature norm (not L2-normalized embeddings). `EmbeddingNet.forward` (`animal_id/embedding/models.py:73-83`) unconditionally L2-normalizes. Must add a `normalize: bool` toggle or emit both normalized+unnormalized.
2. AdaFace requires a running mean/std of feature norms. No state-tracking hooks exist; needs a small module.

### Ablation C — "Swap ResNet50 for ConvNeXt-Tiny"

1. `animal_id/embedding/backbones.py:18-21`: add `CONVNEXT_TINY` to the enum.
2. Add a new elif in `get_backbone` with the right `num_features` (768).
3. Verify the `projection_head` in `animal_id/embedding/models.py:57-62` handles the feature-map shape — `AdaptiveAvgPool2d` on ConvNeXt works, but ConvNeXt's final stage is a `LayerNorm+Linear`, so the `feature_extractor = model.features` convention from MobileNet/EfficientNet breaks. A per-backbone adapter is needed. Non-trivial.
4. Update `TRAINING_CONFIG["EMBEDDING_DIM"]` if you want a different embedding head.

Effort: 1–2 days including a working first run.

### Ablation D — "Use a different augmentation recipe (e.g. TrivialAugment)"

1. `animal_id/common/datasets.py:42-60`: augmentation is baked into the dataset class. You must subclass `DogIdentityDataset` or modify in place.
2. No config field exists for "augmentation name". A sweep over `[light, medium, heavy, trivial_augment, randaugment]` requires a `transform_factory(name)` that doesn't exist.

### Ablation E — "Switch from DogFaceNet to a new identity dataset"

1. Write a new `EmbeddingDatasetConverter` subclass (the existing one is not subclass-friendly: it's a single `convert()` method; copy-paste required).
2. Hardcoded paths `"data/dogfacenet/..."` in `animal_id/embedding/config.py:28`.
3. Nothing prevents identity-label collision between datasets — no namespace prefix.
4. The val split logic is identity-size-biased (section 6); may behave surprisingly on a new dataset.

### Ablation F — "Train only detection without pulling COCO"

Not possible cleanly. `animal_id/detection/dataset_converter.py:292-315` unconditionally calls `_load_coco_bbox_only` on `data["coco_train_json"]` and `data["coco_val_json"]`. Passing empty strings causes `os.path.exists` to return False and the function returns empty lists (ok by luck) but the log lines at `:303-318` still print misleading counts.

---

## PR-Sized Recommendations

Each ≤1 day of work. Listed in suggested merge order.

### PR 1 — Introduce `animal_id/common/seeding.py`
Single function `set_seed(seed, deterministic=False)` that seeds Python, NumPy, PyTorch CPU+CUDA, sets `cudnn.deterministic/benchmark`, and returns a torch.Generator. Call it at the top of `run_embedding_pipeline`, `DetectionTrainer.train`, `KeypointTrainer.train`, and wire `seed=` into the Ultralytics config dicts. Add `seed` to `TRAINING_CONFIG`. Add a test that two runs with seed=0 produce the same first-batch output on a MobileNetV3-small model.

Files touched: `animal_id/common/seeding.py` (new), `animal_id/embedding/{config.py,trainer.py}`, `animal_id/{detection,keypoint}/trainer.py`, `scripts/train_master.py`, `tests/unit/test_seeding.py` (new).

### PR 2 — Wire W&B into `EmbeddingTrainer`
Add `wandb_logger: Optional[WandBLogger]` to `EmbeddingTrainer`. In `_train_and_validate_epoch`, log `train_loss`, `val_mAP`, `TAR@FAR=1%`, `TAR@FAR=0.1%`, `lr[0]`, `grad_norm` (returned from the clip_grad_norm call), `epoch_time`. At end of training, upload `best_model.pt` as a `wandb.Artifact`. Add run-name = timestamped dir name. Add `project=` + `name=` to both YOLO trainer configs so their W&B runs land in the same project.

Files touched: `animal_id/embedding/trainer.py`, `animal_id/{detection,keypoint}/trainer.py`, `scripts/train_master.py`.

### PR 3 — Restore a sane embedding-training baseline + document it
Replace the `1e-8 / 1e-9` LRs in `animal_id/embedding/config.py:19-22` with values that trained the published best model (read from `runs/*/config.json` of a known-good run and pin those). Add a `README.md` section "How these defaults were chosen" pointing at the W&B run. Add a test that asserts `HEAD_LR >= 1e-5`.

Files touched: `animal_id/embedding/config.py`, `README.md`, `tests/unit/test_config.py` (new).

### PR 4 — Hydrate `TRAINING_CONFIG` into a dataclass + YAML loader
Convert the dict in `animal_id/embedding/config.py` into an `EmbeddingConfig(dataclass)`. Add `--config path.yaml` to `scripts/08_train_embedding_model.py` that merges over defaults. Pull `label_smoothing`, `grad_clip`, `phase2_warmup_epochs`, `arcface_s`, `arcface_m` into the dataclass. Inject them through `EmbeddingTrainer.__init__`.

Files touched: `animal_id/embedding/{config.py,trainer.py,losses.py,models.py}`, `scripts/08_train_embedding_model.py`, `scripts/train_master.py`.

### PR 5 — Extract augmentation into a factory
New `animal_id/common/transforms.py::build_transform(name, img_size, is_training)` with at least `basic`, `strong`, `trivial_augment`. `DogIdentityDataset.__init__` takes `transform=` injection. Config exposes `AUGMENTATION_POLICY`.

Files touched: `animal_id/common/{datasets.py,transforms.py}`, `animal_id/embedding/config.py`, tests.

### PR 6 — Full-state checkpointing + resume
`save_checkpoint` writes `{model, optimizer, scheduler, torch_rng, cuda_rng, epoch, phase, best_val_metric, num_classes}`. `EmbeddingTrainer.train(resume_from=...)` restores all of it and continues. Add an integration test that trains 2 epochs, restarts, trains 1 more, and asserts the same final loss as a 3-epoch baseline (within tolerance).

Files touched: `animal_id/embedding/trainer.py`, `tests/integration/test_embedding_pipeline.py`.

### PR 7 — Replace corrupt-image recursion with a skip+metric
In `animal_id/common/datasets.py:82-86`, return `None` on failure; add a `collate_fn` that filters None. Log via `logging` (not `print`) and increment a counter published to W&B as `corrupt_image_count`.

Files touched: `animal_id/common/datasets.py`, `scripts/train_master.py`.

### PR 8 — Unify YOLO trainers behind `BaseYoloTrainer`
Extract `_get_default_config` defaults into class-level dicts, make `DetectionTrainer`/`KeypointTrainer` subclass a `BaseYoloTrainer` with a `default_config` attribute + task-specific `model_name`. Delete ~70 lines of duplication.

Files touched: `animal_id/detection/trainer.py`, `animal_id/keypoint/trainer.py`, `animal_id/common/yolo_trainer.py` (new).

### PR 9 — ONNX parity test
`tests/integration/test_onnx_parity.py`: train for 1 epoch on the mock dataset, export to ONNX, run both the torch model and ONNX on the same random batch, assert `np.allclose(atol=1e-5)`. Add this to CI. Covers regressions from ONNX opset drift or `F.normalize` edge cases.

Files touched: `tests/integration/test_onnx_parity.py` (new).

### PR 10 — Record git state + environment in every run dir
In `run_embedding_pipeline` (`scripts/train_master.py:312-395`), after `run_dir.mkdir`, capture:
- `git rev-parse HEAD` (short-circuit if not a git repo)
- `git diff HEAD`
- `pip freeze`
- `torch.cuda.get_device_name(0)` + `torch.__version__`
Write them under `run_dir/env/` alongside `config.json`. Upload as a W&B artifact. Same for YOLO trainers via a shared `capture_env(run_dir)` helper.

Files touched: `animal_id/common/env.py` (new), `scripts/train_master.py`, `animal_id/{detection,keypoint}/trainer.py`.

### PR 11 — Add a `CosFaceLoss` and swap loss via config
Smallest loss-ablation unlock. `ArcFaceLoss` and `CosFaceLoss` share the cosine-head part; refactor the shared bit into a `MarginHead` base, add a `HEAD_TYPE` config key, dispatch in `DogEmbeddingModel.__init__`. No change to the training loop.

Files touched: `animal_id/embedding/{losses.py,models.py,config.py}`.

### PR 12 — Guard destructive dataset-converter behavior
Add `--force` flag to each `02_prepare_*` / `04_prepare_*` / `06_prepare_*`. Refuse to `shutil.rmtree` unless `--force` is set or the directory is empty. Applies to `animal_id/detection/dataset_converter.py:27-30`, `animal_id/keypoint/dataset_converter.py:42-47`.

Files touched: the three converter modules + three scripts.

### PR 13 — Tighten CI
Pin deps in `environment.yml` (or replace with `uv.lock` / `requirements.lock`). Add a pytest run that also executes the ONNX parity and the seeding reproducibility test. Add `ruff format --check` + `ruff check` + `pytest --cov=animal_id` (coverage is already installed). Fail on `<60%` coverage until improved.

Files touched: `.github/workflows/python-package-conda.yml`, `environment.yml`.

---

## Appendix — Quick-win Nits Not Worth Their Own PR

- `animal_id/embedding/trainer.py:193`: `torch.load(...)` without `weights_only=True`. PyTorch 2.6 will warn; 2.7+ default flipped. Update before adoption.
- `scripts/12_export_keypoint_onnx.py:45`: `YOLO._version` is a private attribute.
- `animal_id/common/datasets.py:39`: `self.num_classes = max(all_labels) + 1` assumes dense 0-indexed labels; a gap in labels silently inflates `num_classes` and wastes ArcFace head memory.
- `animal_id/embedding/trainer.py:150`: `Epoch {(epoch % total_epochs) + 1}/{total_epochs}` is a defensive modulo that can mask off-by-one bugs — better to pass the true global epoch and the phase separately.
- `animal_id/detection/validator.py:23-35`: duplicates `find_latest_run` logic from `animal_id/common/utils.py:11-48`.
- `scripts/train_master.py:287-294`: `from ultralytics import YOLO` inside the function; outside the function in other scripts. Pick one.
- `pyproject.toml:27`: `pathlib2>=2.3.7` is a Python-2 backport — unneeded on Python 3.12.
- `environment.yml` doesn't pin `ultralytics`. Breaking YOLO11 API changes in the 8.3.x line have happened.
