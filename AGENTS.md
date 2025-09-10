# Immich Dogs - Project Status

## Overview
Dog identification system for Immich that mirrors the people detection pipeline. Implements a 3-stage pipeline to detect and identify individual dogs in photos.

## Architecture

### 3-Stage Pipeline
1. **Dog Detector** - YOLO11n finds dog bounding boxes
2. **Keypoint Estimator** - YOLO11n-pose finds 4 face keypoints (eyes, nose, throat)
3. **Identity Embedder** - EfficientNet-B0 + ArcFace loss → 512D embeddings

## Current Status

### ✅ Completed (Phases 1-4)
- **Models Trained**: All three models trained and validated
- **Performance**: 69.24% TAR at 10.0% FAR on validation set (no data leakage)
  - TAR @ FAR=10.000%: 69.24% (Threshold: 0.4179)
  - TAR @ FAR=1.000%: 39.68% (Threshold: 0.5571)  
  - TAR @ FAR=0.100%: 24.15% (Threshold: 0.6437)
  - TAR @ FAR=0.010%: 15.53% (Threshold: 0.6981)
- **Data Integrity**: Fixed data leakage - validation uses completely separate dog identities
- **ONNX Export**: All models exported to `/models/onnx/`
  - `detector.onnx` (10.6MB)
  - `keypoint.onnx` (11.0MB)
  - `embedding.onnx` (18.7MB)
- **Pipeline Verification**: End-to-end pipeline working, embeddings grouping dogs correctly
- **Benchmark Framework**: Comprehensive evaluation system with metrics and visualization
- **AmbidextrousAxolotl Pipeline**: First iteration showing keypoints degrade performance (50 images)
  - Top-1 accuracy: 27.8% (with keypoints) vs 44.4% (without keypoints)
  - Top-3 accuracy: 50.0% (with keypoints) vs 77.8% (without keypoints)
- **Immich Integration**: Complete with both keypoint and direct approaches (100 images)
  - Immich API with keypoints: 65.0% top-1 accuracy
  - Immich API without keypoints: 75.0% top-1 accuracy
  - Validates local benchmark findings in production environment

### 🔄 In Progress (Phase 4)
- **Docker Container**: Custom container build complete and tested
- **API Testing**: HTTP endpoint testing complete - both approaches working
- **Production Deployment**: Ready for deployment with keypoint-free approach recommended

## Code Structure
```
immich-dogs/
├── .planning/           # Project phases and documentation
├── dog_id/             # Core Python package
│   ├── benchmark/      # Evaluation framework
│   ├── pipeline/       # Pipeline implementations (AmbidextrousAxolotl, etc.)
│   └── common/         # Shared utilities
├── scripts/            # Workflow scripts (05-13)
├── models/onnx/        # Exported models
├── immich-clone/       # Forked Immich ML container
└── outputs/            # Results and visualizations
```

## Coding Guidelines

### Code Style & Architecture
- **Minimal Implementation**: Write only the absolute minimal code needed - avoid verbose implementations
- **Protocol-Based Design**: Use Python protocols for dependency injection and modularity
- **Generic Architecture**: Design for extensibility (animals, not just dogs; multiple model types)
- **Elegant Naming**: Use fun, memorable names with clear progression (AmbidextrousAxolotl → BrilliantBadger)

### Documentation & Comments
- **Brief Docstrings**: Concise function/class descriptions, no verbose explanations
- **No Inline Comments**: Code should be self-explanatory through good naming
- **Focus on Why**: Document design decisions and trade-offs, not implementation details

### Error Handling & Robustness
- **Graceful Degradation**: Handle missing files, failed model inference, etc.
- **Preserve Original Features**: When refactoring, maintain existing functionality (progress bars, visualizations, etc.)
- **Field Name Consistency**: Update all related code when changing data structures
- **Progress Indicators**: Use `tqdm` progress bars for any process that could take more than 1 second

### Testing & Validation
- **Benchmark-Driven**: Use quantitative metrics to validate design decisions
- **Comparative Analysis**: Test multiple approaches (with/without keypoints) side-by-side
- **Visual Validation**: Provide visualization tools for understanding system behavior

### Integration Patterns
- **Refactor for Reuse**: Extract common visualization/utility code to shared modules
- **Backward Compatibility**: Maintain existing script interfaces while adding new capabilities
- **Clean Abstractions**: Separate model implementations from pipeline logic
- **One-line Commits**: Use concise single-line commit messages for small changes

## Environment
- **Platform**: WSL (Linux) - most scripts compatible
- **Python**: Use `python312` conda environment for all scripts
- **Note**: Some scripts may have Windows-specific paths/issues - originally developed for native Windows
- **Data Files**: Do not directly read files in `/data` directory (too large). Use head/tail or jq for parsing

### Running Scripts
```bash
# Pipeline verification and benchmarking
conda run -n python312 python scripts/13_run_full_pipeline.py --num-images 50 --num-queries 5

# Export models to ONNX
conda run -n python312 python scripts/10_export_embedding_model.py
conda run -n python312 python scripts/11_export_detector_onnx.py
conda run -n python312 python scripts/12_export_keypoint_onnx.py

# Benchmark Immich API integration
python scripts/16_visualize_immich_pipeline.py --num-images 100 --num-queries 5
python scripts/16_visualize_immich_pipeline.py --num-images 100 --num-queries 5 --skip-keypoints
```

### Immich Container Management
```bash
# Build and run the custom Immich ML container
cd immich-clone/machine-learning
docker build -f Dockerfile.dogs -t immich-ml-dogs .
docker run -d --name immich-ml-dogs -p 3003:3003 immich-ml-dogs

# Container operations
docker stop immich-ml-dogs
docker start immich-ml-dogs
docker logs immich-ml-dogs --tail 20

# Test API endpoint
curl -X POST http://localhost:3003/predict \
  -F "image=@/path/to/dog/image.jpg" \
  -F 'entries={"dog-identification":{"detection":{"modelName":"dog_detector"},"recognition":{"modelName":"dog_embedder_direct"}}}'
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
1. Deploy keypoint-free approach for production use
2. Consider implementing BrilliantBadger pipeline iteration
3. Evaluate performance on larger datasets
4. Optimize model inference speed

## Key Files
- `scripts/13_run_full_pipeline.py` - Full pipeline verification and benchmarking
- `scripts/16_visualize_immich_pipeline.py` - Immich API benchmarking
- `dog_id/benchmark/evaluator.py` - Comprehensive evaluation framework
- `dog_id/pipeline/ambidextrous_axolotl.py` - First pipeline implementation
- `dog_id/common/constants.py` - Project configuration
- `.planning/overarching.md` - Complete project plan

## Phase 1 Migration - COMPLETE! 🎉

### **Migration Status:**
**✅ Prompts 1-4**: Module extraction and centralized utilities  
**✅ Prompts 5-9**: New numbered scripts using extracted modules
**✅ Prompt 10**: Script migration and renumbering

**🎉 Phase 1 migration is now 100% COMPLETE!**

### **Final Script Workflow (01-17):**
- **01-02**: Data preparation & inspection
- **03-05**: Model training (detection, keypoint)  
- **06-10**: Embedding pipeline
- **11-12**: Model export (ONNX)
- **13-17**: Pipeline validation & integration

### **Production Validation Results**

### **End-to-End Tested Scripts:**
- **Script 01**: ✅ Processed 38,414 training + 4,292 validation images in ~3 minutes
- **Script 02**: ✅ Successfully visualized both COCO and YOLO formats for detection and keypoints
- **Script 04**: ✅ Processed 11,283 training + 1,254 validation keypoint samples

### **Critical Bug Fixed:**
- **YOLO Visualization Issue**: Fixed missing bounding boxes caused by:
  - Incorrect label path construction (`images/` → `labels/` mapping)
  - Random sampling including negative samples (empty label files)
  - Solution: Filter for non-empty label files and fix path mapping

### **Production Readiness:**
- **Scripts 01, 02, 04**: Fully production-tested with real data
- **Scripts 03, 05**: Ready for training (GPU auto-detection implemented)
- **All visualizations**: Verified working with proper annotations displayed

## Lessons Learned

### **Testing Philosophy:**
- **"Help flags aren't enough"** - Always test end-to-end with real data
- **Validate visualizations** - Check that annotations actually appear correctly
- **Test edge cases** - Negative samples, empty files, missing data
- **Commit only verified functionality** - Don't assume untested code works

### **Dataset Handling:**
- **Negative samples are normal** - YOLO datasets include images without annotations
- **Path mapping is critical** - `images/` ↔ `labels/` conversion must be exact
- **Filter for meaningful visualization** - Show only samples with actual annotations

### **Code Architecture Validation:**
- **Module extraction successful** - Clean separation of concerns achieved
- **Centralized utilities work** - Visualization functions properly shared
- **Import patterns consistent** - PROJECT_ROOT + sys.path.append pattern works reliably
