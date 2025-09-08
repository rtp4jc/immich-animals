# Prompt 3 — Convert available keypoints / masks to COCO-style keypoints annotations

## Date / Agent run id
2025-08-29 / Agent-1

## Goal
Implement conversion utility for COCO keypoints format.

## Prereqs (checked)
- [x] StanfordExtra keypoint files available at data/stanford_dogs/stanford_extra_keypoints.json
- [x] Oxford mask PNGs available if present
- [x] DogFaceNet crops directory (optional)

## Commands run
- python scripts/phase1/convert_to_coco_keypoints.py

## Files created
- scripts/phase1/convert_to_coco_keypoints.py
- data/coco_keypoints/annotations_train.json
- data/coco_keypoints/annotations_val.json
- data/coco_keypoints/debug_annotations/debug_train_first_20.json
- data/coco_keypoints/debug_annotations/debug_val_first_20.json

## Outputs / Metrics
- Total images converted: 19,927
- Images with full keypoints: 12,527
- Images with synthesized keypoints: 7,390
- Images with bbox-only: 10
- Train set: 17,934 images, 17,934 annotations
- Val set: 1,993 images, 1,993 annotations

## Issues / Errors
- None - conversion completed successfully
- Initially debug_val_first_20.json was empty due to train/val split distribution - fixed by adjusting debug sampling logic to take min(20, len(val_images))

## Next steps / recommended action
- Proceed to prompt 4 for YOLOv8 training with the generated COCO annotations