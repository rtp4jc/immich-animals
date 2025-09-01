# Phase 1 Implementation Summary & Learnings

This document outlines the final, stable implementation for Phase 1 of the project and serves as a log of the issues encountered and the steps taken to resolve them.

## 1. Original Goal (from master.md)

The initial plan was to fine-tune a pre-trained YOLOv8-pose model on a combined dataset to detect dog bounding boxes and a 5-point keypoint schema.

## 2. Evolution of the Data Pipeline

Our attempts to follow the original plan revealed several critical issues that required significant revisions to the data preparation pipeline.

### Issue 2.1: Initial Training Failures (Zero-Loss)
*   **Symptom:** The initial training runs failed to learn, with all key loss metrics remaining at zero.
*   **Root Cause:** A bug in `convert_to_coco_keypoints.py` was failing to correctly link COCO annotations to their corresponding images.
*   **Resolution:** The script was corrected to properly map the `image_id` for all annotations.

### Issue 2.2: Incorrect Keypoint Schema & Data
*   **Symptom:** A visualization script revealed that the keypoints being used were incorrect.
*   **Resolution:** We revised the keypoint strategy to a more robust **4-point schema**: `['nose', 'chin', 'left_ear_base', 'right_ear_base']` and corrected the mapping function.

### Issue 2.3: Lack of True Negative Examples
*   **Symptom:** The model was likely to overfit because it was only being trained on images that contained dogs.
*   **Resolution:** The data pipeline was updated to include thousands of images from the COCO dataset that do **not** contain dogs, providing the model with essential negative examples.

### Issue 2.4: Framework-Specific Data Formatting
*   **Symptom:** The YOLOv8 trainer requires a specific `.txt` label file for each image, and cannot directly use the standard COCO JSON format.
*   **Resolution:** We adopted a two-script pipeline: `convert_to_coco_keypoints.py` to create a master COCO JSON dataset, and `convert_coco_to_yolo.py` to convert that into the required YOLO `.txt` format.

### Issue 2.5: Incorrect Handling of Oxford Pets Dataset
*   **Symptom:** The model was incorrectly detecting cats as dogs.
*   **Root Cause:** The pipeline was treating all animals from the Oxford-IIIT Pets dataset as positive "dog" samples. Furthermore, the bounding boxes for this dataset were for faces only, which is inconsistent with our whole-body objective.
*   **Resolution:** The `convert_to_coco_keypoints.py` script was updated to parse the Oxford XML annotations. It now intelligently uses images of cats (and other non-dog animals) as **negative samples** and completely ignores the dog images from this dataset, preventing data contamination.

### Issue 2.6: Label File Path Mismatch
*   **Symptom:** The model trained, but the `pose_loss` was always zero, indicating that the keypoint data was not being seen by the trainer.
*   **Root Cause:** A bug in `convert_coco_to_yolo.py` was generating the `.txt` label files for the Stanford Dogs dataset in the wrong directory, so the trainer could not find them.
*   **Resolution:** The path generation logic was fixed to create a parallel `labels` directory that mirrors the `images` directory structure, which is the standard convention the trainer expects.

## 3. Final Data Pipeline

Our robust, debugged data pipeline now consists of the following key scripts:

*   **`scripts/phase1/convert_to_coco_keypoints.py`**: The primary data processing script. It takes the raw source datasets (COCO, Stanford Dogs, Oxford-IIIT Pets), resolves their differences, and converts them into a single, unified dataset in the standard **COCO JSON format**. It automatically clears old data on each run.
*   **`scripts/phase1/convert_coco_to_yolo.py`**: A build-step script that prepares the data for the Ultralytics framework. It reads the master COCO JSON files and converts them into the **YOLO `.txt` format**. It automatically clears old labels on each run.
*   **`scripts/phase1/train_detector.py`**: Runs the model training process.
*   **`scripts/phase1/validate_and_infer.py`**: Runs validation and performs inference on sample images, providing both visualized images and detailed console output of detection coordinates.
*   **`scripts/phase1/visualize_dataset.py`**: A crucial debugging tool to visualize annotations from both COCO and YOLO formats on an image, allowing for detailed data validation.
*   **`scripts/phase1/create_sample_images.py`**: Creates a representative set of sample images from all data sources for inference testing.

## 4. Current Status & Next Steps

Phase 1 is now functionally complete. The data pipeline is stable and robust, and the model is training successfully. The key learning from this phase was the critical importance of **data validation**. The visualization script we created was essential for uncovering subtle but critical issues in the source data and our processing logic.

The immediate next steps are:
1.  Continue training the model on the clean dataset for a sufficient number of epochs to achieve good performance.
2.  Use the `validate_and_infer.py` script to evaluate the final model.
3.  Use the `export_model.py` script to convert the final model to ONNX or other formats for deployment.
4.  Proceed to **Phase 2** of the project.

