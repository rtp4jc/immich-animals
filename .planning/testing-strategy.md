# Testing Strategy & Implementation Plan

This document outlines the plan to establish a comprehensive testing suite for the `animal_id` project. It is designed to be a standalone guide for developers implementing the tests.

## Execution
Run all tests using:
```bash
pytest tests/
```

## Phase 1: Foundation & Integration (Priority)

**Goal**: Scaffold the environment and ensure the three main pipelines (Detection, Keypoint, Embedding) can run end-to-end on minimal data.

### Task 1: Scaffold & Refactor
*   **Goal**: Create `tests/` directory, `conftest.py` with shared fixtures, and refactor code to support CPU testing.
*   **Actions**:
    *   Create `tests/`, `tests/unit/`, `tests/integration/`.
    *   **Refactor Trainers**: Modify `animal_id/detection/trainer.py` and `animal_id/keypoint/trainer.py`:
        *   In `verify_prerequisites()`: Allow execution if `self.config.get('device') == 'cpu'`, bypassing the `torch.cuda.is_available()` check.
        *   In `train()`: Replace `sys.exit(...)` with `raise RuntimeError(...)` to allow tests to catch initialization errors.
    *   **Fixtures (`tests/conftest.py`)**:
        *   `mock_image_dataset`:
            *   Creates a temp directory using `tmp_path`.
            *   Generates 5 dummy JPEG images (random numpy arrays -> PIL Image -> save).
            *   Creates a `dataset.json` with the following schema:
                ```json
                [
                  {"file_path": "/path/to/img1.jpg", "identity_label": 0},
                  {"file_path": "/path/to/img2.jpg", "identity_label": 1}
                ]
                ```
            *   Returns the path to `dataset.json`.
        *   `mock_yolo_dataset`:
            *   Creates `images/train` and `labels/train`.
            *   Generates 5 dummy JPEGs.
            *   Generates 5 matching `.txt` files in `labels/train`.
                *   Content: `0 0.5 0.5 0.2 0.2` (class x_center y_center width height).
            *   Creates `data.yaml`:
                ```yaml
                path: /absolute/path/to/temp/dir
                train: images/train
                val: images/train
                nc: 1
                names: ['dog']
                ```
            *   Returns path to `data.yaml`.
        *   `mock_keypoint_dataset`:
            *   Similar to YOLO dataset but with keypoint labels.
            *   Label format: `class x y w h k1x k1y k1v ...` (17 keypoints standard, or simplified).
            *   `data.yaml` must include `kpt_shape: [num_kpts, 3]`.

### Task 2: Detection Pipeline Integration
*   **Goal**: Verify the Detection pipeline works.
*   **File**: `tests/integration/test_detection_pipeline.py`
*   **Scope**:
    *   Use `mock_yolo_dataset`.
    *   Initialize `DetectionTrainer` with `device='cpu'`.
    *   Run `train()` for 1 epoch.
    *   Assert that `best.pt` or `last.pt` exists in the output directory.

### Task 3: Keypoint Pipeline Integration
*   **Goal**: Verify the Keypoint pipeline works.
*   **File**: `tests/integration/test_keypoint_pipeline.py`
*   **Scope**:
    *   Use `mock_keypoint_dataset`.
    *   Initialize `KeypointTrainer` with `device='cpu'`.
    *   Run `train()` for 1 epoch.
    *   Verify model artifact creation.

### Task 4: Embedding Pipeline Integration
*   **Goal**: Verify the Embedding pipeline works.
*   **File**: `tests/integration/test_embedding_pipeline.py`
*   **Scope**:
    *   Use `mock_image_dataset`.
    *   Initialize `DogEmbeddingModel` and `EmbeddingTrainer` (mocking the `device` to 'cpu').
    *   Run `trainer.train(warmup_epochs=1, full_epochs=0, ...)` with minimal parameters.
    *   Verify checkpoint creation.

---

## Phase 2: Core Logic Unit Tests

**Goal**: Verify the correctness of critical data handling and modeling logic.

### Task 5: Datasets
*   **File**: `tests/unit/test_datasets.py`
*   **Scope**: `animal_id/common/datasets.py`
*   **Tests**:
    *   Init with `mock_image_dataset`.
    *   Verify `len(dataset) == 5`.
    *   Verify `__getitem__` returns `(tensor, label)`.
    *   Verify tensor shape matches `img_size` (e.g., `[3, 224, 224]`).

### Task 6: Identity Loader
*   **File**: `tests/unit/test_identity_loader.py`
*   **Scope**: `animal_id/common/identity_loader.py`
*   **Tests**: Dataset splitting logic, sampling strategies, combining datasets.

### Task 7: Modeling
*   **File**: `tests/unit/test_models.py`
*   **Scope**: `animal_id/embedding/models.py` & `backbones.py`
*   **Tests**:
    *   Model initialization (mock `get_backbone` or use small ResNet).
    *   Forward pass with dummy tensor `(2, 3, 224, 224)`.
    *   Verify output shape `(2, 512)`.
    *   Verify `freeze_backbone()` sets `requires_grad=False`.

### Task 8: Inference Orchestration
*   **File**: `tests/unit/test_pipeline_inference.py`
*   **Scope**: `animal_id/pipeline/ambidextrous_axolotl.py`
*   **Strategy**: Use `unittest.mock.MagicMock` to mock the detector, keypoint model, and embedder.
*   **Tests**:
    *   Feed a dummy numpy image.
    *   Mock detector to return a specific bounding box.
    *   Verify the pipeline crops the image based on that box.
    *   Verify the embedder is called with the crop.

---

## Phase 3: Utilities & Converters Unit Tests

**Goal**: Verify helper functions and data converters to prevent data corruption.

### Task 9: Detection Converters
*   **File**: `tests/unit/test_detection_converters.py`
*   **Scope**: `animal_id/detection/dataset_converter.py`, `yolo_converter.py`
*   **Tests**: COCO to YOLO format conversion, coordinate normalization.

### Task 10: Keypoint Converters
*   **File**: `tests/unit/test_keypoint_converters.py`
*   **Scope**: `animal_id/keypoint/dataset_converter.py`, `yolo_converter.py`
*   **Tests**: Keypoint format conversion, visibility handling.

### Task 11: General Utilities
*   **File**: `tests/unit/test_utils.py`
*   **Scope**: `animal_id/common/utils.py`, `visualization.py`
*   **Tests**: File I/O helpers, path manipulation, visualization function smoke tests.
