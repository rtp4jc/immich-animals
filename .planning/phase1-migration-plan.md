# Phase 1 Code Migration Plan

## Overview
Refactor Phase 1 scripts to follow the standardized folder structure defined in `long-term-folder-structure.md` while preserving critical visualization functionality in numbered scripts that follow a consistent pattern.

## Document Usage Guidelines

### When to Update This Document
- **Automatic Updates**: Update this document whenever completing any prompt from this plan
- **Explicit Updates**: Update when specifically requested by the user
- **Progress Tracking**: Mark completed prompts and update acceptance criteria status
- **Issue Documentation**: Add notes about any deviations or problems encountered during implementation

### Where to Find More Project Information
- **`.planning/`** - All planning documents and project context
- **`.planning/long-term-folder-structure.md`** - Target package structure
- **`.planning/overarching.md`** - High-level project goals and context
- **`.planning/phase-*.md`** - Phase-specific implementation details
- **`scripts/phase1/`** - Current Phase 1 implementation to be migrated
- **`scripts/05-14`** - Current Phase 2+ scripts showing target patterns
- **`dog_id/`** - Existing package structure and modules

## Critical Implementation Context

### Current Project Structure
**Phase 1 Scripts (scripts/phase1/):**
- `convert_detector_coco_to_yolo.py` - Converts COCO detection format to YOLO
- `convert_keypoint_coco_to_yolo.py` - Converts COCO keypoint format to YOLO  
- `create_detector_coco_dataset.py` - Creates COCO detection dataset from source data
- `create_detector_sample_images.py` - Creates sample detection images
- `create_keypoint_coco_dataset.py` - Creates COCO keypoint dataset from source data
- `run_two_stage_inference.py` - Runs detection + keypoint inference pipeline
- `train_detector_only.py` - Trains YOLO detection model
- `train_keypoint_model.py` - Trains YOLO keypoint model
- `validate_detector.py` - Validates detection model performance
- `visualize_generic_dataset.py` - Generic dataset visualization tool

### Dependencies and Imports
**Logging**: Use `print()` statements (current project standard)
**Error Handling**: Follow Python best practices (try/except, proper exceptions)
**CLI Arguments**: Always use `argparse` with intuitive defaults to minimize required arguments

### Data Paths and Configuration
**Source Datasets** (in `data/` directory):
- `data/stanford_dogs/` - Stanford Dogs dataset with annotations and keypoints
- `data/oxford_pets/` - Oxford-IIIT Pet Dataset  
- `data/coco/` - COCO dataset for additional training data
- `data/dogfacenet/` - DogFaceNet dataset for embeddings

**Output Structure**: 
- Primary format: COCO dataset structure (source of truth)
- Secondary formats: Derived from COCO (e.g., YOLO format for Ultralytics)
- Model outputs: `models/` directory
- Visualizations: `outputs/` directory

**Existing Output Examples**:
- `data/detector/coco/` - COCO format detection dataset
- `data/detector/dogs_detection.yaml` - YOLO detection config
- `data/keypoints/coco/` - COCO format keypoint dataset  
- `data/keypoints/dogs_keypoints_only.yaml` - YOLO keypoint config

### Code Patterns and Conventions
**Logging**: Use `print()` statements (current project standard)
**Error Handling**: Follow Python best practices (try/except, proper exceptions)
**CLI Arguments**: Always use `argparse` with intuitive defaults to minimize required arguments
**Code Style**: No specific formatting requirements

### Hardware/Environment
**GPU**: GPU available with PyTorch CUDA support enabled
**Environment**: Linux system with ML libraries pre-installed
**Performance**: Standard hardware expectations for ML workloads

### Testing/Validation
**Approach**: Request user validation by providing specific commands to run
**Training Scripts**: User verification only (due to time constraints)
**Data Scripts**: Full execution and output verification expected

## Updated Migration Strategy

### Core Principles
1. **Numbered Script Workflow**: All scripts follow numbered sequence with consistent patterns
2. **Centralize Core Logic**: Extract reusable logic into `dog_id` package modules
3. **File-Based Visualizations**: Default to saving plots to files, optional display mode
4. **Verification Integration**: Include visualization and verification within training scripts

### Package Structure Additions
```
dog_id/
├── __init__.py                  # NEW
├── common/
│   ├── visualization.py         # NEW - Core visualization utilities
│   └── inference.py             # NEW - Inference pipeline utilities
├── detection/                   # NEW MODULE
│   ├── __init__.py
│   ├── dataset_converter.py     # Extract from create_detector_coco_dataset.py
│   ├── yolo_converter.py        # Extract from convert_detector_coco_to_yolo.py
│   ├── trainer.py               # Extract from train_detector_only.py
│   └── validator.py             # Extract from validate_detector.py
└── keypoint/                    # NEW MODULE
    ├── __init__.py
    ├── dataset_converter.py     # Extract from create_keypoint_coco_dataset.py
    ├── yolo_converter.py        # Extract from convert_keypoint_coco_to_yolo.py
    └── trainer.py               # Extract from train_keypoint_model.py
```

### Final Script Structure
```
scripts/
├── 01_prepare_detection_data.py      # Create + convert detection dataset (no visualization)
├── 02_inspect_datasets.py            # PRESERVE from phase1 - generic dataset inspection tool
├── 03_train_detector.py              # Train + validate + visualize detector results
├── 04_prepare_keypoint_data.py       # Create + convert keypoint dataset (no visualization)
├── 05_train_keypoint_model.py        # Train + validate + visualize keypoint results
├── 06_prepare_embedding_data.py      # RENUMBER from current 05
├── 07_train_embedding_model.py       # RENUMBER from current 07
├── 08_validate_embeddings.py         # RENUMBER from current 08
├── 09_export_embedding_model.py      # RENUMBER from current 09
├── 10_export_detector_onnx.py        # RENUMBER from current 10
├── 11_export_keypoint_onnx.py        # RENUMBER from current 11
├── 12_run_full_pipeline.py           # RENUMBER from current 12
├── 13_test_immich_integration.py     # RENUMBER from current 13
├── 14_visualize_immich_pipeline.py   # RENUMBER from current 14
└── 15_run_two_stage_inference.py     # PRESERVE from phase1 - standalone inference tool
```

## Independent Implementation Prompts

### Prompt 1: Create Package Infrastructure ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` for the target package structure.

**Task**: Create the missing package infrastructure for the `dog_id` package.

**Requirements**:
- Create `dog_id/__init__.py` 
- Create `dog_id/detection/` module with `__init__.py`
- Create `dog_id/keypoint/` module with `__init__.py`
- Create placeholder files: `dog_id/common/visualization.py` and `dog_id/common/inference.py`

**Explicit Acceptance Criteria**:
- [✅] Run `python -c "import dog_id; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.detection import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.keypoint import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.common.visualization import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.common.inference import *; print('Success')"` - must print "Success"
- [✅] All created files must have proper docstrings and basic structure

**Implementation Notes**:
- Created all required `__init__.py` files with proper docstrings
- Added placeholder functions in visualization.py with consistent API design
- Added InferencePipeline class structure in inference.py
- All imports work successfully

---

### Prompt 2: Extract Detection Module Logic ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` and examine `scripts/phase1/create_detector_coco_dataset.py`, `scripts/phase1/convert_detector_coco_to_yolo.py`, `scripts/phase1/train_detector_only.py`, and `scripts/phase1/validate_detector.py`.

**Task**: Extract core logic from Phase 1 detection scripts into reusable modules within `dog_id/detection/`.

**Requirements**:
- Extract `CocoDetectorDatasetConverter` class to `dog_id/detection/dataset_converter.py`
- Extract COCO to YOLO conversion logic to `dog_id/detection/yolo_converter.py`
- Extract training configuration and logic to `dog_id/detection/trainer.py`
- Extract validation logic to `dog_id/detection/validator.py`
- Maintain all existing functionality and configuration options
- Add proper imports and error handling

**Explicit Acceptance Criteria**:
- [✅] Run `python -c "from dog_id.detection.dataset_converter import CocoDetectorDatasetConverter; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.detection.yolo_converter import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.detection.trainer import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.detection.validator import *; print('Success')"` - must print "Success"
- [✅] All original phase1 detection scripts must still run without modification
- [✅] Each module must have comprehensive docstrings and type hints
- [✅] No hardcoded file paths - all paths must be configurable parameters

**Implementation Notes**:
- Created `CocoDetectorDatasetConverter` with full functionality from original script
- Created `CocoToYoloDetectionConverter` class for YOLO format conversion
- Created `DetectionTrainer` class with configurable training parameters
- Created `DetectionValidator` class with model loading and validation capabilities
- All modules include proper type hints and comprehensive docstrings
- Configuration is externalized through constructor parameters and default config functions
- Original phase1 scripts remain functional

---

### Prompt 3: Extract Keypoint Module Logic ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` and examine `scripts/phase1/create_keypoint_coco_dataset.py`, `scripts/phase1/convert_keypoint_coco_to_yolo.py`, and `scripts/phase1/train_keypoint_model.py`.

**Task**: Extract core logic from Phase 1 keypoint scripts into reusable modules within `dog_id/keypoint/`.

**Requirements**:
- Extract keypoint dataset creation logic to `dog_id/keypoint/dataset_converter.py`
- Extract COCO to YOLO conversion logic to `dog_id/keypoint/yolo_converter.py`  
- Extract training configuration and logic to `dog_id/keypoint/trainer.py`
- Maintain all existing functionality and configuration options
- Add proper imports and error handling

**Explicit Acceptance Criteria**:
- [✅] Run `python -c "from dog_id.keypoint.dataset_converter import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.keypoint.yolo_converter import *; print('Success')"` - must print "Success"
- [✅] Run `python -c "from dog_id.keypoint.trainer import *; print('Success')"` - must print "Success"
- [✅] All original phase1 keypoint scripts must still run without modification
- [✅] Each module must have comprehensive docstrings and type hints
- [✅] No hardcoded file paths - all paths must be configurable parameters

**Implementation Notes**:
- Created `CocoKeypointDatasetConverter` with full cropping and keypoint transformation logic
- Created `CocoToYoloKeypointConverter` class for YOLO pose format conversion
- Created `KeypointTrainer` class with keypoint-specific training parameters
- All modules include proper type hints and comprehensive docstrings
- Configuration is externalized through constructor parameters and default config functions
- Keypoint mapping and validation logic preserved from original implementation
- Original phase1 scripts remain functional

---

### Prompt 4: Create Centralized Visualization Logic ✅ COMPLETED
**Background**: Review `scripts/phase1/visualize_generic_dataset.py`, `scripts/06_visualize_embedding_data.py`, and `scripts/14_visualize_immich_pipeline.py` to identify common visualization patterns.

**Task**: Create centralized visualization utilities in `dog_id/common/visualization.py` that can be reused across all scripts.

**Requirements**:
- Extract common plotting/visualization functions
- Support for dataset visualization (COCO, YOLO formats)
- Support for model output visualization (bboxes, keypoints, embeddings)
- Support for pipeline visualization (multi-stage results)
- Default to saving plots to `outputs/` directory, optional display parameter
- Use consistent styling and color schemes
- Maintain backward compatibility with existing visualization scripts

**Explicit Acceptance Criteria**:
- [✅] Run test script that calls each visualization function - must save files to `outputs/` without displaying
- [✅] Run test script with `display=True` parameter - must show plots in windows
- [✅] All visualization functions must accept `save_path` and `display` parameters
- [✅] Functions must handle COCO format datasets without errors
- [✅] Functions must handle YOLO format datasets without errors
- [✅] Functions must handle detection bboxes, keypoints, and embedding visualizations
- [✅] All saved plots must be properly formatted and readable
- [✅] No breaking changes to existing visualization workflows

**Implementation Notes**:
- Created comprehensive visualization functions for all major data formats
- Implemented `visualize_coco_annotations()` for COCO dataset inspection
- Implemented `visualize_yolo_annotations()` for YOLO dataset inspection
- Implemented `visualize_detection_results()` and `visualize_keypoint_results()` for model outputs
- Implemented `visualize_training_metrics()` for training progress
- Implemented `visualize_identity_dataset()` for embedding dataset verification
- Added `print_dataset_statistics()` utility function
- All functions use consistent `save_path` and `display` parameters
- Proper error handling for missing files and directories
- Maintains backward compatibility with existing scripts
- Uses matplotlib with proper backend handling for headless environments

---

### Prompt 5: Create Script 01 - Prepare Detection Data ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` and the extracted `dog_id.detection` modules from previous prompts.

**Task**: Create `01_prepare_detection_data.py` that creates and converts detection dataset without visualization.

**Requirements**:
- Use `dog_id.detection.dataset_converter` to create COCO dataset
- Use `dog_id.detection.yolo_converter` to convert to YOLO format
- No visualization (users will run script 02 for inspection)
- Follow same patterns as existing scripts 05-14
- Include progress logging and error handling

**Explicit Acceptance Criteria**:
- [✅] Run `python scripts/01_prepare_detection_data.py` - must complete successfully
- [✅] Verify COCO dataset files created in `data/detector/coco/`
- [✅] Verify YOLO dataset files created in `data/detector/`
- [✅] Verify YOLO config file `dogs_detection.yaml` is created
- [✅] Script must complete in under 10 minutes on standard hardware
- [✅] No visualization files should be created by this script

---

### Prompt 6: Create Script 02 - Inspect Datasets ✅ COMPLETED
**Background**: Review `scripts/phase1/visualize_generic_dataset.py` and the centralized `dog_id.common.visualization` module.

**Task**: Create `02_inspect_datasets.py` that provides generic dataset inspection capabilities.

**Requirements**:
- Update existing `visualize_generic_dataset.py` to use `dog_id.common.visualization`
- Support inspection of COCO, YOLO, and other dataset formats
- Save all visualizations to `outputs/02_dataset_inspection/`
- Accept command line arguments for dataset path and format
- Default to file-based output, optional display mode
- Include dataset statistics and sample visualizations

**Explicit Acceptance Criteria**:
- [✅] Run `python scripts/02_inspect_datasets.py data/detector/coco` - must complete successfully
- [✅] Run `python scripts/02_inspect_datasets.py data/keypoints/coco` - must complete successfully  
- [✅] Verify visualization files saved to `outputs/02_dataset_inspection/`
- [✅] Check that dataset statistics are displayed/saved (image count, class distribution, etc.)
- [✅] Verify sample images with annotations are properly visualized
- [✅] Script must handle missing datasets gracefully with clear error messages
- [✅] Must support both COCO and YOLO format inspection

---

### Prompt 7: Create Script 03 - Train Detector ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` and the extracted `dog_id.detection` modules from previous prompts.

**Task**: Create `03_train_detector.py` that trains the detector and includes validation/visualization.

**Requirements**:
- Use `dog_id.detection.trainer` for training configuration and execution
- Use `dog_id.detection.validator` for model validation
- Use `dog_id.common.visualization` to visualize training results and validation outputs
- Save all visualizations to `outputs/03_detector_training/`
- Include training metrics plots and sample predictions
- Follow same patterns as existing training scripts

**Explicit Acceptance Criteria**:
- [✅] Run `python scripts/03_train_detector.py` - user must verify training starts successfully (don't wait for completion)
- [✅] Verify training configuration is loaded correctly
- [✅] Verify CUDA/GPU detection works if available
- [✅] Verify training logs are created in expected directory
- [✅] Verify visualization setup saves sample plots to `outputs/03_detector_training/`
- [✅] Script must handle missing GPU gracefully with clear error message
- [✅] Training must use the YOLO config created by script 01

---

### Prompt 8: Create Script 04 - Prepare Keypoint Data ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` and the extracted `dog_id.keypoint` modules from previous prompts.

**Task**: Create `04_prepare_keypoint_data.py` that creates and converts keypoint dataset without visualization.

**Requirements**:
- Use `dog_id.keypoint.dataset_converter` to create COCO keypoint dataset
- Use `dog_id.keypoint.yolo_converter` to convert to YOLO format
- No visualization (users will run script 02 for inspection)
- Follow same patterns as script 01
- Include progress logging and error handling

**Explicit Acceptance Criteria**:
- [✅] Run `python scripts/04_prepare_keypoint_data.py` - must complete successfully
- [✅] Verify COCO keypoint dataset files created in `data/keypoints/coco/`
- [✅] Verify YOLO keypoint dataset files created in `data/keypoints/`
- [✅] Verify YOLO config file `dogs_keypoints_only.yaml` is created
- [✅] Script must complete in under 10 minutes on standard hardware
- [✅] No visualization files should be created by this script

---

### Prompt 9: Create Script 05 - Train Keypoint Model ✅ COMPLETED
**Background**: Review `.planning/long-term-folder-structure.md` and the extracted `dog_id.keypoint` modules from previous prompts.

**Task**: Create `05_train_keypoint_model.py` that trains the keypoint model and includes validation/visualization.

**Requirements**:
- Use `dog_id.keypoint.trainer` for training configuration and execution
- Use `dog_id.common.visualization` to visualize training results and keypoint predictions
- Save all visualizations to `outputs/05_keypoint_training/`
- Include training metrics plots and sample keypoint predictions
- Follow same patterns as script 03

**Explicit Acceptance Criteria**:
- [✅] Run `python scripts/05_train_keypoint_model.py` - user must verify training starts successfully (don't wait for completion)
- [✅] Verify training configuration is loaded correctly
- [✅] Verify CUDA/GPU detection works if available
- [✅] Verify training logs are created in expected directory
- [✅] Verify visualization setup saves sample plots to `outputs/05_keypoint_training/`
- [✅] Script must handle missing GPU gracefully with clear error message
- [✅] Training must use the YOLO config created by script 04

---

### Prompt 10: Migrate Standalone Scripts and Renumber
**Background**: Review the current scripts 05-14 and `scripts/phase1/run_two_stage_inference.py`.

**Task**: Migrate standalone script and renumber existing scripts to accommodate new scripts 01-05.

**Requirements**:
- Move `run_two_stage_inference.py` to `15_run_two_stage_inference.py`
- Update script to use `dog_id.common.visualization` and `dog_id.common.inference` where beneficial
- Renumber scripts: 05→06, 07→07, 08→08, 09→09, 10→10, 11→11, 12→12, 13→13, 14→14
- Update any internal references to script numbers
- Remove `scripts/phase1/` directory after migration

**Explicit Acceptance Criteria**:
- [ ] Run `python scripts/15_run_two_stage_inference.py` - must complete successfully (user verify with sample data)
- [ ] All renumbered scripts (06-14) must run successfully without modification
- [ ] Verify `scripts/phase1/` directory is completely removed
- [ ] Verify no broken imports or references between scripts
- [ ] All scripts must follow consistent logging and output patterns
- [ ] Script workflow must follow logical order: data prep → inspection → training → validation → export → integration

## Success Metrics
- All 15 scripts run successfully in sequence
- Phase1 functionality is preserved and enhanced
- Code reusability is improved through modularization
- Visualization is consistent across all scripts with file-based output
- Clear, numbered workflow from data preparation to deployment
