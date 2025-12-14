# Animal Identification for Immich

Animal identification system for Immich that mirrors the people detection pipeline. Implements a 3-stage pipeline to detect and identify individual animals in photos, with initial focus on dogs.

## Architecture

### 3-Stage Pipeline

1. **Animal Detector** - YOLO11n finds animal bounding boxes
2. **Keypoint Estimator** - YOLO11n-pose finds facial keypoints (eyes, nose, throat)
3. **Identity Embedder** - ResNet50 + ArcFace loss → 512D embeddings

**Note**: Current testing shows the pipeline without keypoint detection is more accurate than the full 3-stage approach.

## Setup

### Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, install PyTorch with CUDA support and replace `onnxruntime` with `onnxruntime-gpu`. See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

## Quick Start

### Run Full Pipeline

```bash
# Benchmark with 50 images, 5 queries per identity
python scripts/13_run_full_pipeline.py --num-images 50 --num-queries 5
```

### Tracking & Benchmarking

We use **Weights & Biases (W&B)** to track experiments and visualize performance metrics over time.

### Usage

The pipeline script automatically logs metrics to W&B. Use tags to organize your experiments:

```bash
# Run with a custom tag
python scripts/13_run_full_pipeline.py --num-images 50 --tag "baseline-v1"

# Disable W&B logging
python scripts/13_run_full_pipeline.py --num-images 10 --no-wandb
```

### Dashboard Tips

To compare results over time in the W&B dashboard:

1.  **Create a Line Chart:** Set X-axis to "Wall Time" and Y-axis to your metric (e.g., `top_5_accuracy`, `tar_at_far_0_01`).
2.  **Group by Variant:** Under "Grouping", select "Group by" and choose `use_keypoints`.
3.  **Inspect Failures:** Check the "Media" section to see images of missed detections and incorrect identifications (including Rank and Predicted Identity).

## Data Preparation and Training

```bash
# Prepare detection dataset
python scripts/01_prepare_detection_data.py

# Inspect datasets
python scripts/02_inspect_datasets.py

# Train models (requires GPU for reasonable training time)
python scripts/03_train_detection_model.py
python scripts/05_train_keypoint_model.py
python scripts/08_train_embedding_model.py

# Export to ONNX
python scripts/10_export_embedding_model.py
python scripts/11_export_detector_onnx.py
python scripts/12_export_keypoint_onnx.py
```

## Project Structure

```
immich-animals/
├── animal_id/             # Core Python package
│   ├── benchmark/      # Evaluation framework
│   ├── pipeline/       # Pipeline implementations
│   ├── detection/      # Detection model utilities
│   ├── keypoint/       # Keypoint model utilities
│   ├── embedding/      # Embedding model utilities
│   └── common/         # Shared utilities
├── scripts/            # Workflow scripts (01-17)
├── models/onnx/        # Exported ONNX models
└── outputs/            # Results and visualizations
```

## Key Scripts

- **01-05**: Data preparation and model training
- **06-09**: Embedding pipeline and validation
- **10-12**: ONNX model export
- **13**: Full pipeline verification and benchmarking
- **14-15**: Model inspection and inference testing

**Note**: Scripts 16-17 (Immich integration) require a local fork of Immich and are not currently functional for external users.

## Testing

We use `pytest` for testing. The test suite includes:
- **Unit Tests**: Verify core logic (data loading, modeling, converters).
- **Integration Tests**: Verify end-to-end training pipelines (Detection, Keypoint, Embedding).

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_datasets.py

# Run with verbose output
python -m pytest -v tests/
```

### Coverage Report

To see code coverage statistics (ensure dependencies are installed):

```bash
python -m pytest --cov=animal_id tests/
```

## Models

Exported models in `/models/onnx/`:

- `detector.onnx` (~11MB) - Animal detection
- `keypoint.onnx` (~11MB) - Facial keypoint estimation
- `embedding.onnx` (~19MB) - Identity embedding generation

## Data Requirements

### Dataset Downloads

The system requires several datasets for training. Download and extract to the specified locations:

**COCO 2017** (detection training): https://cocodataset.org/#download

- Place under `data/coco/images/train2017/` and `data/coco/images/val2017/`

**DogFaceNet** (identity embedding): https://github.com/GuillaumeMougeot/DogFaceNet#dataset

- Place under `data/dogfacenet/DogFaceNet_224resized/` and `data/dogfacenet/DogFaceNet_alignment/`

**Stanford Dogs** (additional training): http://vision.stanford.edu/aditya86/ImageNetDogs/

- Images: `data/stanford_dogs/images/`
- Annotations: `data/stanford_dogs/annotation/`

**StanfordExtra Dogs** (keypoint annotations): https://www.kaggle.com/datasets/ollieboyne/stanfordextra-dogs-dataset

- Place JSON file at `data/stanford_dogs/stanford_extra_keypoints.json`

**Oxford Pets** (additional data): https://www.robots.ox.ac.uk/~vgg/data/pets/

- Images: `data/oxford_pets/images/`
- Annotations: `data/oxford_pets/annotations/`

Scripts will guide you through dataset preparation after download.

## Development Notes

- Pipeline architecture supports different animal classes through the `AnimalClass` enum
- Keypoint-free approach currently shows better performance than the full 3-stage pipeline
