# Immich Dogs - Project Status

## Overview
Dog identification system for Immich that mirrors the people detection pipeline. Implements a 3-stage pipeline to detect and identify individual dogs in photos.

## Architecture

### 3-Stage Pipeline
1. **Dog Detector** - YOLO11n finds dog bounding boxes
2. **Keypoint Estimator** - YOLO11n-pose finds 4 face keypoints (eyes, nose, throat)
3. **Identity Embedder** - EfficientNet-B0 + ArcFace loss → 512D embeddings

## Current Status

### ✅ Completed (Phases 1-3)
- **Models Trained**: All three models trained and validated
- **Performance**: 83.97% TAR at 1.0% FAR on validation set
- **ONNX Export**: All models exported to `/models/onnx/`
  - `detector.onnx` (10.6MB)
  - `keypoint.onnx` (11.0MB)
  - `embedding.onnx` (18.7MB)
- **Pipeline Verification**: End-to-end pipeline working, embeddings grouping dogs correctly

### 🔄 In Progress (Phase 4)
- **Immich Integration**: Need to implement model classes in `immich-clone/machine-learning`
- **Docker Container**: Custom container build pending
- **API Testing**: HTTP endpoint testing not yet complete

## Code Structure
```
immich-dogs/
├── .planning/           # Project phases and documentation
├── dog_id/             # Core Python package
├── scripts/            # Workflow scripts (05-13)
├── models/onnx/        # Exported models
├── immich-clone/       # Forked Immich ML container
└── outputs/            # Results and visualizations
```

## Environment
- **Platform**: WSL (Linux) - most scripts compatible
- **Python**: Use `python312` conda environment for all scripts
- **Note**: Some scripts may have Windows-specific paths/issues - originally developed for native Windows
- **Data Files**: Do not directly read files in `/data` directory (too large). Use head/tail or jq for parsing

### Running Scripts
```bash
# Pipeline verification
conda run -n python312 python scripts/12_run_full_pipeline.py

# Export models to ONNX
conda run -n python312 python scripts/10_export_detector_onnx.py
conda run -n python312 python scripts/11_export_keypoint_onnx.py
conda run -n python312 python scripts/09_export_embedding_model.py
```

### Script Output Handling
When running scripts with large outputs, save to file and read selectively:
```bash
# Save output to file
conda run -n python312 python scripts/08_validate_embeddings.py > outputs/scripts/validation_output.txt 2>&1

# Read only head/tail of output
head -20 outputs/scripts/validation_output.txt
tail -20 outputs/scripts/validation_output.txt
```

## Next Steps
1. Implement `DogDetector`, `DogKeypoint`, `DogEmbedder` classes in Immich ML container
2. Build custom Docker container with dog models
3. Test integration via HTTP API calls
4. Deploy for production use

## Key Files
- `scripts/12_run_full_pipeline.py` - Full pipeline verification
- `dog_id/common/constants.py` - Project configuration
- `.planning/overarching.md` - Complete project plan
