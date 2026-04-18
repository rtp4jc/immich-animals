# Structure, Conventions, and Extensibility Audit

**Date**: 2026-04-16  
**Scope**: `/mnt/e/Code/GitHub/immich-animals`  
**Audience**: Immich maintainers evaluating adoption of this project

---

## Executive Summary

`immich-animals` is a well-structured, single-maintainer research project that has outgrown its first-iteration naming and dog-specific assumptions. The pipeline architecture is genuinely clean — Protocol-based model injection, a single constants file for all paths, a logical separation of training stages, and a healthy test-to-code ratio. However, three issues will create friction for Immich OSS adoption:

1. **Dog-specific naming throughout the public API** — `DogEmbeddingModel`, `DogIdentityDataset`, `DogIdentificationSystem` are in the public package, not just scripts. Every class name needs renaming before the project claims to be "animal identification."
2. **A whimsical, unsearchable orchestrator name** — `AmbidextrousAxolotl` is the main pipeline class. It makes code review, issue triage, and search harder than necessary.
3. **No static type checking** — the Immich ML codebase uses mypy; this project does not declare it at all.

Everything else is in good shape or is a small, fixable issue. The testing strategy is solid for a project of this size.

---

## 1. Project Structure Findings

### Packaging (`pyproject.toml`)
- Uses `setuptools` with `find:` auto-discovery — correct for a single-namespace package. `[project]` table is complete.
- **`pathlib2>=2.3.7` is listed as a runtime dependency** (`pyproject.toml:28`). `pathlib` has been in the standard library since Python 3.4; `pathlib2` is a backport for Python 2. This is a dead dependency. Python 3.12 is required, so `pathlib2` serves no purpose and adds a pointless install.
- `wandb>=0.23.1` is a **runtime** (not dev) dependency. Training experiments require W&B, but inference does not. Users who only want to run the inference pipeline are forced to install W&B. Consider moving it to an `[extras]` group, e.g., `pip install animal_id[train]`.
- `transformers>=4.30.0` is a runtime dependency but does not appear to be used anywhere in `animal_id/` (no `import transformers` found). This is either dead or a future placeholder.

### Module Layout Under `animal_id/`
```
animal_id/
├── benchmark/     # evaluator, metrics, visualizer — good separation
├── common/        # constants, datasets, identity_loader, inference (placeholder!), utils, visualization
├── detection/     # trainer, dataset_converter, yolo_converter, validator
├── embedding/     # models, backbones, losses, trainer, dataset_converter, config
├── keypoint/      # trainer, dataset_converter, yolo_converter
├── pipeline/      # ambidextrous_axolotl, models (protocols), onnx_models
└── tracking/      # wandb_logger
```
The layout is logical and follows the three-stage architecture. The symmetry between `detection/`, `keypoint/`, and `embedding/` is intentional and good.

**Issue**: `animal_id/common/inference.py` is entirely placeholder code — every method body is just `# Placeholder - to be implemented in later prompts` with no actual implementation. The class `InferencePipeline` has methods `load_models`, `detect_dogs`, `extract_keypoints`, `generate_embedding`, `process_image` that all return `None` implicitly. The real inference pipeline lives in `animal_id/pipeline/ambidextrous_axolotl.py`. This file is dead code.

**Issue**: `animal_id/__init__.py:8` still describes the system as "3-stage pipeline for detecting and identifying individual dogs" — the multi-species framing from `AnimalClass` is not reflected here.

### Script Organization
Numbered scripts `01`–`17` with gaps (`02`, `07`, `09`, `14`, `15` are inspection/visualization only) provide a clear workflow. The orchestration pattern — thin entry-point scripts calling functions in `train_master.py` — is excellent. Scripts are importable because `train_master.py` exposes named functions.

`scripts/train_master.py:71` re-defines `PROJECT_ROOT = Path(__file__).resolve().parents[1]` instead of importing from `animal_id.common.constants`. This violates the single-source-of-truth principle the project otherwise follows.

### Docs Location
`README.md`, `REPLICATION.md`, `AGENTS.md`, `CLAUDE.md` at the root. The `.planning/` directory contains maintainer planning notes — these should not be in the production tree (or at minimum should be `.gitignore`d for release branches).

---

## 2. Abstractions and Protocol Review

### Protocol Definitions (`animal_id/pipeline/models.py`)
The three protocols are clean and coherent:
- `DetectionModel.predict(image: np.ndarray) -> List[Dict[str, Any]]`
- `KeypointModel.predict(image: np.ndarray) -> List[Dict[str, Any]]`
- `EmbeddingModel.predict(image: np.ndarray) -> np.ndarray`

**What works well**: All three ONNX wrappers (`ONNXDetector`, `ONNXKeypoint`, `ONNXEmbedding` in `onnx_models.py`) satisfy their protocols. The pipeline accepts them by structural subtyping (Python Protocol) without forcing inheritance. Mocks work naturally in tests.

**Missing**: The return type `List[Dict[str, Any]]` is underspecified. A `Detection = TypedDict(...)` for `{'bbox': List[int], 'confidence': float, 'class': AnimalClass}` would make contracts explicit and enable type-checking tools to catch schema violations. Without it, e.g., `ambidextrous_axolotl.py:127` uses `d.get("class", self.target_class)` defensively because the contract is not enforced.

**`AnimalClass` enum location**: Defined in `animal_id/pipeline/models.py:13`. This is the right place since it is a pipeline-level concept, but it is also used in `benchmark/evaluator.py:101` as a backward-compat alias. The relationship is circular-adjacent; `AnimalClass` arguably belongs in `animal_id/common/`.

**`AnimalIdentificationSystem` Protocol** (`benchmark/evaluator.py:74`): This is the interface used by `BenchmarkEvaluator`. `AmbidextrousAxolotl` inherits from it (line 17) rather than satisfying it by structural subtyping. Mixing inheritance and Protocol is unusual — if the intent is a formal interface, use `runtime_checkable` Protocol. If it is just documentation, the inheritance is unnecessary.

---

## 3. Extensibility Walkthroughs

### 3.1 Add a New Dataset (e.g., Cats)

**Target species friction** is the highest-friction scenario in the entire codebase.

Concrete file changes required:

1. **`animal_id/pipeline/onnx_models.py:42`** — `ONNXDetector.predict()` hardcodes `"class": AnimalClass.DOG`. To support cats, either the model output must encode multi-class detections and this code must decode them, or a separate `ONNXCatDetector` must be created. A single trained detector for both species would need class-ID mapping here.

2. **`animal_id/detection/dataset_converter.py:125`** — `_load_coco_bbox_only` filters to `cat["name"] == "dog"` only. Adding cats requires changing this string and the `categories` list at line 329 (`{"id": 1, "name": "dog"}`). The category ID `1` is also hardcoded in `current_annotations` construction at line 97 and 176. A multi-class design would map COCO category IDs to local IDs via a configurable mapping.

3. **`animal_id/detection/dataset_converter.py:217`** — `_load_oxford_pets_negatives` negates on `!= "dog"`. For a cat detector, this logic inverts.

4. **`animal_id/embedding/dataset_converter.py`** — The `EmbeddingDatasetConverter` is relatively generic (directory-scan based), but its docstring and print statements (`line 37`: "Scanning DogFaceNet for dog identities...") are dog-specific. The class name itself is fine.

5. **`animal_id/embedding/config.py:28`** — `"DOGFACENET_PATH"` hardcodes a dog-specific dataset path key name. For cats, a new config section or a parameterized `SPECIES_PATH` would be needed.

6. **`animal_id/common/datasets.py:29`** — `DogIdentityDataset` is dog-named. Since the logic is fully generic (JSON path → dataset), renaming to `IdentityDataset` costs nothing and enables reuse.

7. **`animal_id/embedding/models.py:86`** — `DogEmbeddingModel` is dog-named but generic. Renaming to `AnimalEmbeddingModel` is a one-line change with wide ripple effects across scripts and tests.

8. **`animal_id/detection/trainer.py:32`** — Default config `"data": "data/detector/dogs_detection.yaml"`. This path must change per species.

9. **`scripts/train_master.py`** — Multiple `DogEmbeddingModel` and `DogIdentityDataset` imports, all needing renaming.

**Friction rating**: HIGH. Dog-specific assumptions are embedded in 8+ files. The naming is the main issue; the logic itself is largely generic. A rename+refactor PR could fix this without algorithmic changes.

### 3.2 Swap Embedding Backbone (ResNet50 → ConvNeXt-V2 or DINOv2)

**Target file**: `animal_id/embedding/backbones.py`

Changes required:

1. **Add enum value** to `BackboneType` at `backbones.py:18` — e.g., `CONVNEXT_V2_BASE = "convnext_v2_base"`.
2. **Add an `elif` branch** in `get_backbone()` at `backbones.py:24`. ConvNeXt-V2 and DINOv2 require `torchvision>=0.15` (ConvNeXt is available) or the `timm` library (DINOv2). If using `timm`, add it to `pyproject.toml` dependencies.
3. **No changes needed** to `EmbeddingNet`, `DogEmbeddingModel`, `EmbeddingTrainer`, or `train_master.py` — they all operate on `backbone_type: BackboneType`.
4. **Update default** in `embedding/config.py:9` — `DEFAULT_BACKBONE = BackboneType.RESNET50`.

**Friction rating**: LOW. The backbone factory pattern is correctly abstracted. Only `backbones.py` must change for supported torchvision models. For DINOv2 specifically, the feature extraction API differs (ViT outputs sequence tokens, not feature maps), so `EmbeddingNet.forward()` at `models.py:77` would need to handle the case where `feature_extractor` outputs a tensor that does not need `AdaptiveAvgPool2d`. The `AdaptiveAvgPool2d(1)` + `Flatten` head in `EmbeddingNet` at `models.py:57-60` assumes CNN-style `(B, C, H, W)` output. This is one additional line of code, not a redesign.

### 3.3 Swap Loss Function (ArcFace → AdaFace)

**Target file**: `animal_id/embedding/losses.py`

Changes required:

1. **Add `AdaFaceLoss` class** to `losses.py`. AdaFace is a drop-in replacement for ArcFace — same constructor signature (`in_features`, `out_features`, `s`, `m`), same `forward(embeddings, labels) -> logits` contract.
2. **`animal_id/embedding/models.py:113`** — `DogEmbeddingModel.__init__` hardcodes `ArcFaceLoss`:
   ```python
   self.head = ArcFaceLoss(embedding_dim, num_classes)
   ```
   This is the **only place that needs to change** for loss swapping. Adding a `loss_type: LossType` parameter here and a one-line factory would make it configurable from config.
3. **`animal_id/embedding/config.py`** — Add a `LOSS_TYPE` key to `TRAINING_CONFIG`.
4. **`scripts/train_master.py`** — Pass the loss type through to `DogEmbeddingModel`.

**Friction rating**: LOW-MEDIUM. The loss is cleanly isolated in `losses.py`, but `DogEmbeddingModel` hardcodes `ArcFaceLoss` at instantiation time rather than accepting it as a dependency. A `loss_fn_type` parameter on `DogEmbeddingModel.__init__` solves this.

### 3.4 Change Embedding Dimension (512 → 256)

**Target**: `animal_id/embedding/config.py:14` — `"EMBEDDING_DIM": 512`

Changes required:

1. **Single config change** — `TRAINING_CONFIG["EMBEDDING_DIM"] = 256` in `config.py:14`. This value is read by `train_master.py` and propagated correctly throughout training and ONNX export.
2. **Re-export ONNX** — the ONNX model shape changes; old `embedding.onnx` must be regenerated.
3. **No code changes** — `EmbeddingNet` accepts `embedding_dim` as a constructor parameter. `ArcFaceLoss` is constructed with `embedding_dim` at `models.py:113`. The ONNX export dummy input shape does not depend on embedding dim (it is the input image shape).

**Friction rating**: LOWEST. This is exactly what a config-driven system should look like.

---

## 4. Testing Findings

### Coverage Assessment

**What is tested (unit)**:
- `DogIdentityDataset` — initialization, `__getitem__`, training vs. validation transforms (4 tests, `test_datasets.py`)
- `DogEmbeddingModel` + `EmbeddingNet` — init, training forward, inference forward, freeze/unfreeze, `get_embeddings` (5 tests, `test_models.py`)
- `AmbidextrousAxolotl.generate_embedding` — no detection, wrong class, basic flow, keypoint refinement, gallery similarity (5 tests, `test_pipeline_inference.py`)
- `ONNXDetector`, `ONNXKeypoint`, `ONNXEmbedding` — preprocessing and output parsing, all mocked (`test_onnx_models.py`)
- `calculate_tar_at_far`, `evaluate_embedding_model` — edge cases (3 tests, `test_metrics.py`)
- `EmbeddingTrainer` — empty val loader, empty metrics during training loop (2 tests)
- `CocoDetectorDatasetConverter` — Stanford and COCO parsing (2 tests)
- `CocoKeypointDatasetConverter` — keypoint mapping, validation, image processing (3 tests)
- `IdentityLoader` — base load, per-identity limit, additional scan, augmented dataset (4 tests)
- `find_latest_run`, `find_latest_timestamped_run`, visualization smoke test (3 tests, `test_utils.py`)

**What is tested (integration)**:
- `train_detector` — 1-epoch training on mock YOLO dataset
- `train_keypoint_model` — 1-epoch training on mock pose dataset
- `EmbeddingTrainer.train` — 1-epoch warmup + 1-epoch full on mock dataset

**What is NOT tested**:
- `BenchmarkEvaluator.evaluate` and `_compute_metrics` — the full evaluator logic (detection ranking, MRR computation) has no unit test
- `animal_id/benchmark/visualizer.py` — zero tests
- `animal_id/common/visualization.py` — only a smoke test for `visualize_detection_results`; other visualization functions untested
- `animal_id/detection/validator.py` — `DetectionValidator` untested
- `animal_id/tracking/wandb_logger.py` — untested
- ONNX export pipeline (scripts 10–12) — not covered
- `IdentityLoader.load_validation_data` with `num_images` parameter — not tested (only `max_per_identity`)

### Test Quality

**Good practices observed**:
- `conftest.py` at `tests/` root shares fixtures across all test subdirectories — `mock_image_dataset`, `mock_yolo_dataset`, `mock_keypoint_dataset` are properly scoped.
- Integration tests use `tmp_path`, `pretrained=False`, and `device='cpu'` — they are hermetic and do not require a GPU or network.
- `test_pipeline_inference.py` uses `unittest.mock.MagicMock` for the three protocol-typed models — the Protocol design pays off here.
- Edge cases are tested (empty dataloader, no detections, wrong class filter).

**Issues**:
- `test_datasets.py:47` — `test_dataset_transforms_training` asserts `not torch.allclose(img1, img2)`. This is statistically likely but not guaranteed. With `RandomCrop` on a 224+32 image to 224, if the random crop position is the same both times, the assertion fails. Extremely rare, but a flaky test in theory.
- `conftest.py:113` — `mock_keypoint_dataset` creates labels with 17 keypoints (`kpt_shape: [17, 3]`) — COCO standard — but the project uses 4 keypoints. The mock is mismatched with production data format. This does not cause test failures because the integration test does not validate keypoint outputs, but it reduces the test's fidelity.
- The `conftest.py:84` `mock_yolo_dataset` labels use `"names": ["dog"]` — a string reference that should generalize to whatever class is being trained.

### CI Gates

The workflow at `.github/workflows/python-package-conda.yml` runs: ruff lint → ruff format check → pytest.

**Missing gates**:
- No `pytest --cov` with a minimum coverage threshold — coverage is not enforced
- No `mypy` or `pyright` type check step
- No benchmark regression gate — performance could degrade silently across PRs
- The workflow uses `strategy.max-parallel: 5` but there is only one job — this setting has no effect
- Tests run without explicit timeout — a hanging integration test could block CI indefinitely
- No test split between fast unit tests and slow integration tests — integration tests (which run real YOLO training for 1 epoch) slow down every PR

---

## 5. Configuration Discipline

### What is centralized (`animal_id/common/constants.py`)
Path constants only: `PROJECT_ROOT`, `MODELS_DIR`, `DATA_DIR`, `ONNX_DIR`, and derived ONNX model paths. This is correctly scoped — it is the single source of truth for file locations.

### What is centralized (`animal_id/embedding/config.py`)
Embedding training hyperparameters: `DEFAULT_BACKBONE`, `TRAINING_CONFIG` (LR, epochs, patience, workers, embedding dim), `DATA_CONFIG` (paths, img_size, batch_size). This is good — these are the knobs a researcher changes between runs.

### Remaining scattered config / magic numbers

| Location | Value | Issue |
|----------|-------|-------|
| `animal_id/embedding/trainer.py:211` | `warmup_epochs_phase2 = 20` | Hardcoded inside `train()`, not exposed in `TRAINING_CONFIG` or as a parameter |
| `animal_id/embedding/trainer.py:212` | `start_factor=0.01` | LR warmup start factor is a hyperparameter |
| `animal_id/detection/dataset_converter.py:298` | `15000` (COCO train negatives) | Hardcoded negative sample count |
| `animal_id/detection/dataset_converter.py:310` | `2000` (COCO val negatives) | Hardcoded negative sample count |
| `animal_id/pipeline/ambidextrous_axolotl.py:138` | `0.1` (detection crop padding) | Padding factor is a behavior parameter |
| `animal_id/pipeline/ambidextrous_axolotl.py:162` | `0.2` (keypoint padding) | Keypoint padding factor |
| `scripts/train_master.py:71` | `PROJECT_ROOT = Path(__file__).resolve().parents[1]` | Duplicates `constants.py:12` |
| `animal_id/detection/trainer.py:32` | `"data/detector/dogs_detection.yaml"` | Dog-specific hardcoded path |
| `animal_id/keypoint/trainer.py:32` | `"data/keypoints/dogs_keypoints_only.yaml"` | Dog-specific hardcoded path |

### argparse in scripts
`scripts/13_run_full_pipeline.py` has `--num-images`, `--include-additional`, `--no-wandb`, `--tag`. These are well-named and consistent. `scripts/train_master.py` has `--skip-detection`, `--skip-embedding`, `--skip-benchmark`, `--skip-trained`, `--no-wandb`, `--tag`. Also well-named.

---

## 6. Documentation Findings

### README
`README.md` is comprehensive — architecture overview, setup, training workflow, data requirements, testing instructions. The Quick Start command references `--num-queries 5` which does not exist in the current `scripts/13_run_full_pipeline.py` argparse definition (the parameter was removed; the README was not updated).

### Module-level docstrings
Most modules have module-level docstrings. `animal_id/embedding/losses.py` has an unusually detailed "What it's for / What it does / How to run it / How to interpret" docstring that reads as scaffolding from early development rather than stable documentation. The "How to run it" section points to `training/losses.py` (a path that no longer exists).

### Function-level docstrings
Good coverage in public-facing classes. Protocol methods have docstrings specifying arg types and return schemas — this is the right place for schema documentation given the `Dict[str, Any]` return types.

### Type hints
Partially applied. All Protocol definitions use type hints correctly. `EmbeddingTrainer.train()` at `trainer.py:167` has no return type annotation. Several utility functions in `common/utils.py` are correctly typed.

### No mypy configuration
`pyproject.toml` has no `[tool.mypy]` section. The Immich ML codebase (`immich-clone/machine-learning/pyproject.toml:79`) requires mypy with pydantic plugin. This is a gap for OSS adoption.

---

## 7. Naming and Cohesion

### The "ambidextrous axolotl" problem

`animal_id/pipeline/ambidextrous_axolotl.py` — the main inference pipeline class is named `AmbidextrousAxolotl`. This is the central class that Immich maintainers would review, debug, and extend. The name is:
- Unsearchable (try `grep -r "pipeline" .` vs. `grep -r "axolotl" .`)
- Undescriptive of function
- A cognitive tax in every code review comment and issue title

A descriptive name like `AnimalPipeline`, `IdentificationPipeline`, or `ThreeStagePipeline` costs nothing and pays dividends in perpetuity. The module filename `ambidextrous_axolotl.py` makes the import path `from animal_id.pipeline.ambidextrous_axolotl import AmbidextrousAxolotl` — awkward in any downstream usage.

### Dog-specific public API names
These names appear in the public package API and tests:
- `DogEmbeddingModel` (`animal_id/embedding/models.py:86`) — logic is fully generic
- `DogIdentityDataset` (`animal_id/common/datasets.py:29`) — logic is fully generic
- `DogIdentificationSystem` (`animal_id/benchmark/evaluator.py:101`) — kept as alias, but alias points to `AnimalIdentificationSystem` which coexists — confusing
- `detect_dogs()` method in `animal_id/common/inference.py:40` — in placeholder/dead code

### Good naming
- `AnimalClass` enum — correctly generic
- `AnimalIdentificationSystem` Protocol — correctly generic
- `BenchmarkEvaluator`, `EmbeddingTrainer`, `DetectionTrainer` — clear
- `EmbeddingNet` vs. `DogEmbeddingModel` — inconsistent: `EmbeddingNet` is generic; `DogEmbeddingModel` wraps it with a dog name

---

## 8. Linting / Typing

### Ruff configuration (`pyproject.toml:44-51`)
```toml
[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]
```
- `E` (pycodestyle), `F` (pyflakes), `I` (isort) — a reasonable minimum set
- `E501` (line too long) is disabled — acceptable
- Missing rule sets that would add value: `B` (flake8-bugbear catches common bugs), `UP` (pyupgrade for Python 3.12 compatibility), `RUF` (Ruff-specific rules), `SIM` (simplify)
- No `[tool.ruff.format]` section — default Ruff formatting settings are used

### Type annotation discipline
- Good: Protocol definitions, `common/constants.py`, `pipeline/models.py`, `benchmark/evaluator.py`
- Acceptable: `embedding/trainer.py` (missing return type on `train()`)
- Missing: `common/inference.py` (placeholder), `benchmark/visualizer.py`
- No `mypy` run in CI, no `pyproject.toml` mypy config

### No pyright
The Immich ML project uses mypy. This project uses neither. Adding mypy as a CI step with a permissive initial config (`ignore_errors = true` baseline, then tighten) would be the path to adoption parity.

---

## 9. Dead Code and Duplication

### Dead code

1. **`animal_id/common/inference.py`** — `InferencePipeline` class with 5 stub methods that all return `None` implicitly. Never imported anywhere in the package. This appears to be a first-iteration scaffold from early prompts that was superseded by `AmbidextrousAxolotl`. File should be deleted.

2. **`if __name__ == "__main__":` block in `animal_id/embedding/losses.py:130-189`** — A 60-line self-test that instantiates dummy tensors and runs assertions. Functionality is partially duplicated by `tests/unit/test_models.py`. The script test was never integrated into the test suite. This self-test code is better moved to pytest tests and removed from the source file.

3. **`animal_id/detection/validator.py`** — `DetectionValidator` and `validate_latest_detector()` have no callers in the scripts or tests. They may have been used in early scripts but are currently orphaned.

### Duplication

1. **`verify_prerequisites()` method** is identical in `DetectionTrainer` (`detection/trainer.py:50-69`) and `KeypointTrainer` (`keypoint/trainer.py:51-70`) — 20 lines of exact duplication. This could be a `BaseYOLOTrainer` mixin or a shared utility function in `common/`.

2. **`update_config()` method** is identical in both trainers — 3 lines, trivial but repeated.

3. **`update_config()`** convenience functions `train_detector()` and `train_keypoint_model()` follow the same pattern — acceptable for discoverability.

4. **`_preprocess()` methods** in `ONNXDetector` (`onnx_models.py:47`) and `ONNXKeypoint` (`onnx_models.py:86`) are nearly identical (same resize, same normalize, same transpose, different interpolation method). A shared `_preprocess_yolo_input()` helper would remove duplication.

---

## 10. CI Quality

### Current gates (`.github/workflows/python-package-conda.yml`)
- ruff lint (`ruff check .`)
- ruff format check (`ruff format --check .`)
- pytest (all tests, no flags)

### What passes the bar
- Linting and formatting are enforced — a ruff violation blocks merge. This is correct.
- Tests run on every push and PR to `main`.
- Uses conda for environment reproducibility (matching the project's `environment.yml`).

### Missing gates
1. **Coverage gate** — `pytest --cov=animal_id --cov-fail-under=70 tests/` is not present. Without a minimum, coverage can erode silently.
2. **Type checking** — no mypy or pyright step. The Immich ML codebase requires mypy.
3. **Test splitting** — integration tests (`test_detection_pipeline.py`, `test_keypoint_pipeline.py`, `test_embedding_pipeline.py`) run actual 1-epoch YOLO training. These are slow. Unit and integration tests should be split: `-m unit` vs `-m integration`, with integration tests gated to post-merge or manual trigger.
4. **Benchmark regression** — no automated performance gate. A 10% drop in MRR would not be caught by CI.
5. **`max-parallel: 5`** with a single job is a no-op setting.
6. **Security scanning** — no `pip-audit` or Dependabot configuration.
7. **No timeout on pytest** — a hanging integration test (e.g., YOLO training deadlock) will consume the full GitHub Actions runner time limit.

---

## PR-Sized Recommendations

Listed in priority order for OSS adoption appeal:

### P0 — Must-fix before pitching to Immich

**PR-1: Rename dog-specific public API classes**
- `DogEmbeddingModel` → `AnimalEmbeddingModel` (or `EmbeddingModel` — conflicts with protocol name, so `AnimalEmbeddingModel`)
- `DogIdentityDataset` → `IdentityDataset`
- `DogIdentificationSystem` alias removal (keep `AnimalIdentificationSystem` only)
- Files: `animal_id/embedding/models.py`, `animal_id/common/datasets.py`, `animal_id/benchmark/evaluator.py`, all tests, `scripts/train_master.py`, `scripts/10_export_embedding_onnx.py`, `scripts/09_validate_embeddings.py`

**PR-2: Rename `AmbidextrousAxolotl` → `AnimalPipeline`**
- Rename class and module file: `ambidextrous_axolotl.py` → `pipeline.py` (or `animal_pipeline.py` to avoid shadowing the package name)
- Files: `animal_id/pipeline/ambidextrous_axolotl.py`, all scripts and tests that import it
- Update `AGENTS.md` / `CLAUDE.md`

### P1 — High value, small scope

**PR-3: Delete `animal_id/common/inference.py` (dead code)**
- Verify no imports exist (confirmed: zero callers)
- Remove file

**PR-4: Fix pyproject.toml dependency hygiene**
- Remove `pathlib2` (dead backport)
- Move `wandb` and `transformers` to `[project.optional-dependencies]` training group

**PR-5: Extract `verify_prerequisites()` and `update_config()` duplication**
- Create `animal_id/common/yolo_utils.py` with shared `BaseYOLOTrainer` or a free function
- Removes ~25 lines of identical code

**PR-6: Add mypy to CI**
- Add `mypy>=1.3.0` to dev dependencies
- Add `[tool.mypy]` to `pyproject.toml` with `ignore_errors = true` as starting baseline
- Add `mypy animal_id` step to CI workflow

**PR-7: Add `TypedDict` for detection/keypoint return schemas**
- Define `Detection = TypedDict(...)` and `KeypointDetection = TypedDict(...)` in `animal_id/pipeline/models.py`
- Replace `List[Dict[str, Any]]` in protocol signatures
- Removes defensive `d.get("class", self.target_class)` pattern

### P2 — Good hygiene, lower urgency

**PR-8: Add pytest markers and split CI jobs**
- Mark unit tests with `@pytest.mark.unit`, integration tests with `@pytest.mark.integration`
- Add fast CI job (unit only) that runs on every commit
- Keep integration test job gated to post-merge

**PR-9: Add coverage gate**
- Add `pytest --cov=animal_id --cov-fail-under=65 tests/unit/` to CI
- Establishes a floor without breaking anything

**PR-10: Move `warmup_epochs_phase2 = 20` and magic numbers into config**
- Expose `PHASE2_WARMUP_EPOCHS` in `TRAINING_CONFIG`
- Move detection negative sample counts from `dataset_converter.py` into the converter config

**PR-11: Add ruff rule sets B, UP, RUF**
- `select = ["E", "F", "I", "B", "UP", "RUF"]` in `pyproject.toml`
- Fix resulting issues (likely small given existing code quality)

**PR-12: Fix README / AGENTS.md stale references**
- `--num-queries 5` in README Quick Start does not match current CLI
- `losses.py` "How to run it" section points to nonexistent path
- `animal_id/__init__.py` still describes a "dogs" system

**PR-13: Remove `__main__` self-test from `losses.py`**
- Move assertions to `tests/unit/embedding/test_losses.py`
- Reduces source file length from 190 to ~130 lines

**PR-14: Deduplicate `ONNXDetector` and `ONNXKeypoint` `_preprocess` methods**
- Extract shared `_preprocess_yolo_input(image, input_size) -> Tuple[np.ndarray, Tuple[int,int]]`
- Minor but visible in code review
