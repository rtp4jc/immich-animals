# Phase 4 Plan: Immich Backend Integration

## 1. Background & Goal

The goal of this phase is to take our exported ONNX models and integrate them into a custom build of the `immich-machine-learning` service. This involves writing new model classes that conform to Immich's internal `InferenceModel` API and testing them via direct network requests.

This phase focuses *only* on the machine learning container. We will not be running a full Immich instance, nor will we be dealing with the main backend server or database. The process is as follows:

1.  **The `immich-machine-learning` Container:** This is a FastAPI web service that exposes a `/predict` endpoint. It accepts an image and a JSON payload describing a pipeline of models to run.
2.  **Stateless Operation:** The container is stateless. It receives a request, runs the models, and immediately returns the result (e.g., an embedding vector) in the HTTP response. It does not store any data.
3.  **Our Goal:** We will add our three ONNX models and the corresponding logic to a fork of this container. We will then build and run this container in isolation.
4.  **Testing:** We will test our custom container by sending it images using `curl` or a Python script. We will verify that it correctly returns a 512-d embedding for each dog it finds.

Clustering of embeddings and saving them to a database is the responsibility of the main `immich-server` and is out of scope for this phase.

## 2. Acceptance Criteria

This phase will be considered complete when:

1.  The necessary Python files for our new `DogDetector`, `DogKeypoint`, and `DogEmbedder` model classes are created within our local clone of the `immich-machine-learning` source code.
2.  A custom `Dockerfile` is prepared that correctly builds a container with our new models and logic included.
3.  The custom container builds and runs successfully using `docker build` and `docker run`.
4.  Sending a request to the container's `/predict` endpoint with an image and the correct JSON payload for our dog pipeline successfully returns a JSON response containing a 512-dimensional embedding vector.
5.  The embeddings generated for a small test set of known dogs show high cosine similarity for images of the same dog and low similarity for images of different dogs, proving the pipeline works correctly in the Immich environment.

## 3. Step-by-Step Prompts

### Prompt 4.1: Create Integration Scaffolding

**Action:** First, prepare the Immich source code for our new models.
1.  In our local `immich-clone/machine-learning/immich_ml/schemas.py` file, add a new `ModelTask` enum member called `DOG_IDENTIFICATION`. We will place our models under the `facial_recognition` task umbrella to align with the existing structure and potentially reuse logic.
2.  Create three new empty files inside the `immich-clone/machine-learning/immich_ml/models/facial_recognition/` directory:
    *   `dog_detector.py`
    *   `dog_keypoint.py`
    *   `dog_embedder.py`

### Prompt 4.2: Implement the `DogDetector` Model Class

**Action:** In `facial_recognition/dog_detector.py`, create the `DogDetector` class.
1.  It must inherit from `immich_ml.models.base.InferenceModel`.
2.  The class should be initialized with `ModelTask.DOG_IDENTIFICATION` and `ModelType.DETECTION`.
3.  The `predict` method will take an image, run inference with the `detector.onnx` model, and return a list of bounding boxes.

### Prompt 4.3: Implement the `DogKeypoint` Model Class

**Action:** In `facial_recognition/dog_keypoint.py`, create the `DogKeypoint` class.
1.  It must inherit from `InferenceModel`.
2.  It will have a dependency on the output of the `DogDetector` model.
3.  The `predict` method will take the original image and the bounding boxes from the detector, crop the image for each box, run inference with the `keypoint.onnx` model, and return keypoints for each detected dog.

### Prompt 4.4: Implement the `DogEmbedder` Model Class

**Action:** In `facial_recognition/dog_embedder.py`, create the `DogEmbedder` class.
1.  It must inherit from `InferenceModel`.
2.  It will have dependencies on the outputs of both the detector and keypoint models.
3.  The `predict` method will implement our core cropping logic (use keypoints if present, fallback to bbox if not), run the `embedding.onnx` model on the resulting crop, and return the final 512-d embedding vector.

### Prompt 4.5: Build and Test the Custom Container

**Action:** Create a `Dockerfile.dogs` and a test script.
1.  Create a `Dockerfile.dogs` in the `immich-clone/machine-learning` directory. This will be a copy of the original `Dockerfile` but modified to copy our new model files and the `.onnx` artifacts into the container.
2.  Write a script `scripts/13_test_immich_integration.py` that uses the `requests` library to:
    *   POST a test image to `http://localhost:3003/predict`.
    *   Include the JSON `entries` payload required to trigger our `DOG_IDENTIFICATION` pipeline.
    *   Receive the response, validate that it contains an embedding, and print it.
3.  Provide the `docker build` and `docker run` commands needed to execute this test.
