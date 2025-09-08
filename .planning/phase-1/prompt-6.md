# Prompt 6 — Run quick validation & save sample inference outputs

## Date / Agent run id
2025-08-29 / code-xai/grok-code-fast-1

## Goal
Create validation and inference scripts for the trained YOLOv8 pose model, run evaluation on val split and sample images, generate CSV report.

## Prereqs (checked)
- [x] conda env created
- [x] models/phase1/pose_run14/weights/best.pt exists
- [x] outputs/phase1/sample_images/ exists (40 sample images)
- [x] data/dogs_keypoints.yaml exists

## Commands run
- `conda activate python312 && python scripts/phase1/validate_and_infer.py`

## Files created
- `scripts/phase1/validate_and_infer.py` - Script for validation and inference with reporting

## Outputs / Metrics
### Validation Results
- Dataset: val split with 790 images, 0 instances (background images)
- Box Metrics: mAP50: 0.0000, mAP50-95: 0.0000, Precision: 0.0000, Recall: 0.0000
- Pose Metrics: mAP50: 0.0000, mAP50-95: 0.0000, Precision: 0.0000, Recall: 0.0000
- WARNING: No labels found in pose set, metrics invalid
- Visual results saved to runs\pose\val

### Inference Results
- Total sample images: 40
- Images with detected dogs: 0 (0.0%)
- Average keypoints detected: 0.0
- No dogs detected in any sample images
- Visual results saved to outputs/phase1/inference (saved overlaid images though no detections)

## Issues / Errors
- Validation dataset contains only background images (0 instances), rendering metrics meaningless
- Trained model fails to detect any dogs in sample images (0/40 detections)
- Known issue: Validation labels are missing or empty in cache file
- Script executed successfully but model performance is poor (likely due to training data quality or insufficient epochs)

## Next steps / recommended action
1. Investigate why validation split contains only background images
2. Check training data quality and annotation completeness
3. Consider retraining model with better data or increased epochs
4. Verify YOLO dataset YAML configuration
5. Run diagnostics on training process (check Prompt 5 results)
6. If model performance inadequate, may need to revisit dataset preparation (Prompts 2-4)
7. For debugging: Examine sample images to ensure they actually contain dogs
8. Evaluate model architecture choice (YOLO v8n-pose) vs. training data complexity