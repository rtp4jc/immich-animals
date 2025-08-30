# Detection Dataset Implementation Notes

This document outlines the process for creating the aggregated dataset used to train the YOLOv8-pose model for dog detection and facial keypoint estimation.

## 1. Dataset Construction

The final dataset is a combination of three distinct sources, each providing unique features to enhance the model's robustness and accuracy.

### Dataset Sources

1.  **COCO 2017 Dataset**
    *   **Purpose**: Provides a large and diverse set of images for learning robust dog detection in various environments. Crucially, it also serves as a source of **implicit negative examples**, as the model learns to ignore other objects (people, cars, etc.) in the images.
    *   **Contribution**: Bounding boxes for dogs.

2.  **Stanford Dogs Dataset (with StanfordExtra Keypoints)**
    *   **Purpose**: This is the primary source of high-quality keypoint annotations.
    *   **Contribution**: Bounding boxes and detailed facial keypoints (eyes, nose, ears), which are essential for training the pose estimation capabilities of the model.

3.  **Oxford-IIIT Pets Dataset**
    *   **Purpose**: Provides additional examples of dog faces, particularly for improving head detection and keypoint estimation under different conditions.
    *   **Contribution**: Bounding boxes and *synthesized* keypoints derived from pixel-level segmentation masks. This acts as a form of weak supervision to augment the high-quality Stanford data.

## 2. Prerequisites

To reconstruct the dataset, the source datasets must be downloaded and placed in the correct directory structure within the project.

### Directory Structure

The `data/` directory must be structured as follows:

```
data/
├── coco/
│   ├── annotations/
│   │   └── instances_train2017.json
│   └── train2017/
│       └── (*.jpg images)
├── oxford_pets/
│   ├── annotations/
│   │   └── trimaps/
│   │       └── (*.png masks)
│   └── images/
│       └── (*.jpg images)
└── stanford_dogs/
    ├── stanford_extra_keypoints.json
    └── Images/
        └── (*.jpg images)
```

## 3. Key Scripts

The primary script responsible for processing these sources and generating the final annotation files is:

*   `scripts/phase1/convert_to_coco_keypoints.py`

This script performs the following actions:
*   Loads data from the three sources.
*   Filters the COCO dataset to only include images and annotations for the "dog" category (ID: 18).
*   Maps the detailed keypoints from the Stanford dataset to our simplified 5-point schema.
*   Synthesizes keypoints from the Oxford Pets masks.
*   Combines all data and splits it into `train` and `val` sets.
*   Adds metadata to each image entry, including the source dataset and a full relative file path.

## 4. How to Reconstruct the Dataset

Once the prerequisites are met (i.e., the source datasets are in the correct locations), you can reconstruct the entire dataset by running a single command from the project root:

```bash
python scripts/phase1/convert_to_coco_keypoints.py
```

This command will generate the final annotation files in the `data/coco_keypoints/` directory:

*   `annotations_train.json`
*   `annotations_val.json`

These two files, along with the source image directories, are all that is needed for the training process.
