# Phase 1 Implementation Plan: Two-Stage Detector and Pose Estimator

## 1. Background & Goal

Our initial attempt to train a single `yolov8-pose` model revealed a critical limitation: the trainer cannot handle datasets that mix annotations with keypoints and those with only bounding boxes. This resulted in the model ignoring a large portion of our data (the COCO dataset).

To resolve this, we are adopting a more robust, two-stage pipeline. This plan details the implementation of this new approach, adhering to the project's existing data processing workflow (`source -> COCO -> YOLO`).

The new goal is to produce two separate, specialized models:

- **Phase 1-A:** A robust dog detector.
- **Phase 1-B:** A specialized keypoint estimator.

## 2. Phase 1-A: The Robust Dog Detector

**Objective:** Train a `yolo11n` model using all available bounding box data to make it highly effective at finding dogs in varied scenes.

*This phase is complete. The prompts below are retained for documentation.* 

### Revisions during Implementation (Phase 1-A)

Based on initial training results and analysis, we made the following improvements to the data pipeline to enhance the detector's robustness:

1.  **Integrated Full Stanford Dogs Dataset:** The `create_detector_coco_dataset.py` script was updated to parse the XML annotations from the entire Stanford Dogs dataset, not just the `StanfordExtra` subset, significantly increasing the number of positive training examples.
2.  **Corrected Data Errors:** The script was also improved to automatically handle known data errors in the Stanford Dogs dataset, including skipping entries with corrupt filenames and clamping out-of-bounds bounding box coordinates to the image dimensions.
3.  **Increased Negative Samples:** To help the model better distinguish dogs from other objects and backgrounds, the number of negative samples drawn from the COCO dataset was tripled from 5,000 to 15,000.

## 3. Phase 1-B: The Specialized Keypoint Estimator

**Objective:** Train a standard `yolo11n-pose` model that is highly accurate at finding keypoints on images that are already cropped around a dog. By training on pre-cropped images, the model's bounding box prediction task becomes trivial, forcing it to focus its learning capacity on high-precision keypoint estimation.

### Prompt 4: Create a Cropped Keypoint COCO Dataset

"For the keypoint model, we need a new dataset composed of cropped images. This requires a new data generation script.

**Action:** Create a new script `scripts/phase1/create_keypoint_coco_dataset.py`. This script is responsible for:

1.  Reading only from the source dataset that contains keypoints (i.e., `data/stanford_dogs/stanford_extra_keypoints.json`).
2.  For each dog annotation, cropping the original image using the ground-truth bounding box. It's important to add some padding (e.g., 20%) to the crop to provide context for the model.
3.  **This is the most critical step:** Transforming the keypoint coordinates from the original image's coordinate system to the new cropped image's coordinate system.
4.  Saving the newly created cropped images to a directory like `data/keypoint_dataset/images/`.
5.  After processing all images, generating the corresponding COCO-format `annotations_train.json` and `annotations_val.json` files. These files will be saved to `data/coco_keypoints_cropped/` and must reference the file paths of the new cropped images and contain the transformed keypoint data."

### Prompt 5: Convert the Keypoint Dataset to YOLO Format

"Now, we convert the new cropped COCO dataset to the YOLO format.

**Action:** Create a new script `scripts/phase1/convert_keypoint_coco_to_yolo.py`. This script will:

1.  Read from `data/coco_keypoints_cropped/`.
2.  Generate YOLO `.txt` labels. Each label will contain the transformed keypoint data and a bounding box that corresponds to the full cropped image (i.e., class `0`, center `0.5 0.5`, width `1.0`, height `1.0`).
3.  Generate a `data/dogs_keypoints_only.yaml` configuration file, which **must** include our custom `kpt_shape: [4, 3]` and the corresponding `flip_idx`.
4.  The output image list files should be named `data/keypoints_train.txt` and `data/keypoints_val.txt`."

### Prompt 6: Train the Keypoint Model

"Finally, we train the specialized keypoint estimator.

**Action:** Create a new script `scripts/phase1/train_keypoint_model.py`. This script will:

1.  Load a pre-trained `yolo11n-pose.pt` model. This allows us to leverage transfer learning.
2.  Use the `data/dogs_keypoints_only.yaml` file, which contains our custom 4-keypoint configuration, to guide the training.
3.  Save the results to a new project directory, e.g., `models/phase1/keypoint_run`."

## 4. Phase 1-C: The Inference Pipeline

**Objective:** Combine the detector and keypoint estimator into a single, end-to-end inference pipeline.

### Prompt 7: Implement the Two-Stage Inference Script

"The final step is to create a script that uses our two models to process images.

**Action:** Create a new script `scripts/phase1/run_two_stage_inference.py`. This script must:

1.  Load the best trained detector model from Phase 1-A.
2.  Load the best trained keypoint model from Phase 1-B.
3.  For each sample image, first run the detector to get bounding boxes.
4.  For each bounding box found, crop the original image (with padding).
5.  Run the keypoint model on the crop.
6.  Translate the keypoints found in the crop back to the original image's coordinate system.
7.  Visualize and save the final results, showing the detector's bounding box and the translated keypoints."

## 5. Phase 2: Next Steps for Improving Robustness

Once the two-stage pipeline is functional, the following steps can be taken to further improve the detector's performance on "in-the-wild" images:

1.  **Integrate the Open Images Dataset:** This is the highest priority for improving robustness. The Open Images Dataset by Google is massive and contains a vast, diverse collection of images with a "Dog" class. Integrating this would involve:
    *   Using a tool like `FiftyOne` or `odtk` to download the relevant subset of the dataset.
    *   Adding a new loader function to `create_detector_coco_dataset.py` to process the Open Images annotations and add them to our training set.

2.  **Experiment with Model Size:** If performance is still not sufficient, we can experiment with larger model backbones (e.g., `yolo11s` or `yolo11m`). This offers a trade-off between higher accuracy and slower inference speed.

3.  **Advanced Augmentation:** We can explore more advanced data augmentation techniques, such as copy-paste augmentation, to create more complex scenes and further improve the model's ability to handle occlusions and varied contexts.