# Phase 3 Plan: Model Export and Inference Pipeline

## 1. Background & Goal

The goal of this phase is to bridge the gap between our trained models and a real-world application. We will convert our PyTorch models into a standardized, portable format (ONNX) and create a unified Python script that validates the entire end-to-end inference logic.

This phase is critical for ensuring our three models (detector, keypoint estimator, and embedder) work together as expected before we attempt to integrate them into the complex Immich environment. It also produces the final model artifacts that will be used in Phase 4.

The pipeline logic is as follows:
1.  An image is loaded.
2.  The **detector** model finds the full-body bounding box of a dog.
3.  The image is cropped to the dog.
4.  The **keypoint** model is run on the crop to find face keypoints.
5.  A final crop is made based on the results:
    *   If keypoints are found, a tight "face chip" is cropped.
    *   If keypoints are not found, the full-body crop is used as a fallback.
6.  The **embedding** model is run on the final crop to produce a 512-d vector.

## 2. Acceptance Criteria

This phase will be considered complete when:

1.  All three models (detector, keypoint, embedder) are successfully exported to the ONNX format and saved in the `/models/onnx/` directory.
2.  Scripts to perform these exports exist in the `/scripts/` directory, allowing for easy re-export in the future.
3.  A single script, `scripts/12_run_full_pipeline.py`, can be executed. It must take a directory of test images as input, run the full three-stage pipeline on each, and produce a meaningful analysis of the resulting embeddings. This includes printing a similarity matrix and, for a few query images, showing the most similar images from the set to visually verify that the embeddings are grouping correctly.

## 3. Step-by-Step Prompts

### Prompt 3.1: Export Detector Model to ONNX

**Action:** Create a new script `scripts/10_export_detector_onnx.py`. This script will:
1.  Load the best-performing detector model checkpoint (`.pt` file) from the training runs.
2.  Use the `export` functionality provided by the `ultralytics` library.
3.  Export the model to ONNX format.
4.  Save the resulting file to `models/onnx/detector.onnx`.

### Prompt 3.2: Export Keypoint Model to ONNX

**Action:** Create a new script `scripts/11_export_keypoint_onnx.py`. This script will:
1.  Load the best-performing keypoint model checkpoint (`.pt` file).
2.  Use the `ultralytics` library's `export` function.
3.  Ensure the export includes the pose-estimation head.
4.  Save the resulting file to `models/onnx/keypoint.onnx`.

### Prompt 3.3: Verify/Update Embedding Model Export

**Action:** Review the existing `scripts/09_export_embedding_model.py` script.
1.  Ensure it correctly loads our trained `EfficientNet-B0` embedding model.
2.  Confirm that it exports the model to ONNX format.
3.  Modify it to save the output to `models/onnx/embedding.onnx`.
4.  If the script is insufficient, update it to meet these requirements.

### Prompt 3.4: Create Unified Inference and Verification Script

**Action:** Create the final validation script, `scripts/12_run_full_pipeline.py`. This script must:
1.  Import `onnxruntime`, `numpy`, `PIL`, `glob`, and `sklearn.metrics.pairwise.cosine_similarity`.
2.  Load the three `.onnx` models from the `models/onnx/` directory.
3.  Accept a command-line argument for a directory of test images (e.g., `data/test_sets/`). This directory should contain subdirectories for each dog, e.g., `data/test_sets/dog_A/`, `data/test_sets/dog_B/`.
4.  Iterate through all images, run the full multi-stage inference pipeline, and store the resulting embedding for each image.
5.  After processing all images, calculate the pairwise cosine similarity matrix for all embeddings.
6.  For each dog identity in the test set, select one image as a query and print its top 5 most similar images from the entire set, along with their similarity scores. This will serve as a qualitative check.
7.  (Optional but recommended) Save debug images for each processed photo, showing the final bounding box and keypoints used for the embedding crop.
