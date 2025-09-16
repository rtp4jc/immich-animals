# Long-Term Folder Structure

This document outlines the standardized folder structure for the project, designed to accommodate multiple models (detector, keypoint, embedding) and potentially multiple ML frameworks.

## Top-Level Structure

- `/data`: Raw and processed datasets.
- `/models`: Trained model output files (`.pt`, `.onnx`, etc.).
- `/outputs`: Visualizations, reports, and other script outputs.
- `/animal_id`: Core Python source code, structured as a package.
- `/scripts`: Top-level, numbered executable scripts that define the end-to-end workflow.

## `animal_id` Source Package Structure

The `animal_id` package is organized by machine learning task.

```
animal_id/
├── __init__.py
├── common/                  <-- Code shared across all tasks
│   ├── __init__.py
│   └── datasets.py
├── detection/               <-- All logic for detection models (YOLO)
│   ├── __init__.py
│   └── ...
├── embedding/               <-- All logic for the embedding model (PyTorch)
│   ├── __init__.py
│   ├── backbones.py         <-- Backbone factory for easy swapping
│   ├── losses.py
│   ├── models.py
│   └── trainer.py           <-- Generic two-phase PyTorch trainer
└── keypoint/                <-- All logic for keypoint models (YOLO)
    ├── __init__.py
    └── ...
```

## `scripts` Workflow Structure

The `scripts` folder contains simple, numbered Python files that import logic from the `animal_id` package and execute it. This shows the clear, step-by-step workflow for the entire project.

```
scripts/
├── 01_prepare_detection_data.py
├── 02_train_detector.py
├── 03_prepare_keypoint_data.py
├── 04_train_keypoint_model.py
├── 05_prepare_embedding_data.py
├── 06_train_embedding_model.py
├── 07_validate_embeddings.py
└── 08_export_embedding_model.py
```
