# Phase 5: Simplified Immich Integration (Hijack Method)

## 1. Objective

To rapidly integrate the animal identification capabilities into Immich by "hijacking" the existing Facial Recognition pipeline. Instead of creating parallel "Pet" structures (Database, API, UI), we will swap the underlying Machine Learning models. The system will "think" it is detecting people, but it will actually be detecting and recognizing animals.

## 2. Strategy

We will modify the `immich-machine-learning` service to recognize a new `ModelSource` ("animals") and map specific model names (`dog_detector`, `dog_embedder`) to our custom ONNX models. We will adapt the output of our models to match the schemas expected by the existing `facial-recognition` task (bounding boxes, landmarks, embeddings).

## 3. Implementation Plan

### 3.1. Machine Learning Service Modifications (`immich-clone/immich-app/machine-learning`)

#### 3.1.1. Schema Updates
*   **File**: `immich_ml/schemas.py`
*   **Action**: Add `ANIMALS` to the `ModelSource` enum.
    ```python
    class ModelSource(StrEnum):
        ANIMALS = "animals"
        # ... existing sources
    ```

#### 3.1.2. Model Registration
*   **File**: `immich_ml/models/constants.py`
*   **Action**: Update `get_model_source` to return `ModelSource.ANIMALS` for our specific model names.
    ```python
    _ANIMAL_MODELS = {"dog_detector", "dog_embedder"}
    
    def get_model_source(model_name: str) -> ModelSource | None:
        if model_name in _ANIMAL_MODELS:
            return ModelSource.ANIMALS
        # ... existing logic
    ```

#### 3.1.3. New Model Implementations
Create a new directory `immich_ml/models/animals/` with two files:

1.  **`detector.py`** (Implements `AnimalDetector`)
    *   Inherits from `InferenceModel`.
    *   **Input**: Image.
    *   **Logic**: Uses `detector.onnx` (YOLOv11) to find bounding boxes.
    *   **Output**: Adapts YOLO output to `FaceDetectionOutput`.
        *   `boxes`: YOLO bboxes.
        *   `scores`: YOLO confidence scores.
        *   `landmarks`: **Crucial Step**. Since our embedder doesn't require alignment via 5 facial landmarks like ArcFace, we can provide dummy landmarks (e.g., center of bbox) or 0s to satisfy the type definition, provided the Recognizer ignores them.

2.  **`recognizer.py`** (Implements `AnimalRecognizer`)
    *   Inherits from `InferenceModel`.
    *   **Input**: Image + Bounding Boxes (from Detector).
    *   **Logic**: 
        *   Overrides `_crop`: Performs standard cropping/resizing of the bounding box (no affine transformation/alignment based on landmarks).
        *   Uses `embedding.onnx` to generate embeddings for crops.
    *   **Output**: `FacialRecognitionOutput` (List of objects with `embedding`, `score`, `boundingBox`).

#### 3.1.4. Model Factory Update
*   **File**: `immich_ml/models/__init__.py`
*   **Action**: Update `get_model_class` to instantiate our new classes when the source is `ANIMALS`.
    ```python
    case ModelSource.ANIMALS, ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION:
        return AnimalDetector
    case ModelSource.ANIMALS, ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION:
        return AnimalRecognizer
    ```

### 3.2. Configuration Changes

#### 3.2.1. Server Default Configuration
*   **File**: `immich-clone/immich-app/server/src/config.ts`
*   **Action**: Change the default model names for `facialRecognition`.
    ```typescript
    facialRecognition: {
      enabled: true,
      modelName: 'dog_detector', // Was 'buffalo_l'
      // ...
    }
    ```
    *   *Note*: The recognition model name is typically inferred or configured alongside. We might need to ensure `dog_embedder` is used for the recognition step. The `MachineLearningRepository` uses the same `modelName` for both detection and recognition in some calls, or splits them. We need to verify if `modelName` in config applies to both. 
    *   *Correction*: `FaceDetectionOptions` in `machine-learning.repository.ts` sends:
        ```typescript
        [ModelType.DETECTION]: { modelName, ... },
        [ModelType.RECOGNITION]: { modelName }
        ```
        This implies the *same* name is used for both by default in the current logic. We might need to split this in `config.ts` or `machine-learning.repository.ts` if we want distinct names (e.g. `dog_detector` vs `dog_embedder`), OR we can name our models such that they share a base name or handle the mapping in python. 
        *   *Simpler Approach*: Use a single config string `dog_models` and map it to `dog_detector` and `dog_embedder` inside Python `get_model_class` or just use the same name `dog_model` and have the factory return the correct class based on `ModelType`. Let's go with using `dog_model` as the name for both in config, and the Python factory decides which class (Detector vs Recognizer) to load based on the `ModelType` requested.

### 3.3. Asset Management

*   **Location**: `immich-clone/immich-app/machine-learning/ann/models/` (or wherever the cache is).
*   **Action**: We need to ensure our `.onnx` files are available to the container.
    *   **Dev**: Volume mount our local `models/onnx` directory to the container's model cache location.
    *   **Prod**: We would upload them to HF, but for this phase, volume mount is sufficient.

## 4. Execution Steps

1.  **Prepare Models**: Ensure `detector.onnx` and `embedding.onnx` are exported and valid.
2.  **Apply Code Changes**: Modify the Python files in `immich-clone`.
3.  **Build/Run Container**:
    *   Build the modified `immich-machine-learning` image.
    *   Run it with the necessary volume mounts for the models.
4.  **Configure Immich**:
    *   Set the environment variable `IMMICH_MACHINE_LEARNING_FACIAL_RECOGNITION_MODEL_NAME=dog_model`.
    *   Restart Immich Server.
5.  **Test**:
    *   Upload images of dogs.
    *   Check the "People" tab. It should show clusters of dogs.
    *   Verify merging/naming works as expected.

## 5. Risks & Mitigations

*   **Landmark Dependency**: If the server or web UI relies on facial landmarks for cropping thumbnails (smart cropping), the dummy landmarks might cause weird crops.
    *   *Mitigation*: We can try to approximate landmarks (e.g. eyes) from keypoints if we include the keypoint model, or just center them.
*   **Model Performance**: Our models are trained on specific data. Generalization might vary.
*   **Database Pollution**: "People" faces will be mixed with Dog faces if the DB isn't cleared.
    *   *Mitigation*: Use a fresh DB or re-scan all assets.

