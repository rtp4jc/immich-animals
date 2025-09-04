Here’s a revised set of **Phase 2 prompts** for building the **dog embedding model**, updated with learnings from the Phase 1 data pipeline implementation. These follow the same structure as Phase 1: each prompt has prerequisites, an overarching context block, and a progress-tracking instruction. The goal of this phase is to implement a prototype dog embedding network trained on identity-labeled data.

---

# Overarching Prompt (include at the top of every task)

You are an LLM agent with access to local code files. You are developing a **prototype dog embedding model** for identity recognition. The code lives in:

```
E:\Code\GitHub\immich-dogs
```

## Development constraints:

*   System: **Windows**, you are running on WSL so you can use bash commands, but the python scripts will be run separately in a windows command prompt.
*   Environment: **conda** virtual environment with **Python 3.10+**, **CUDA-enabled GPU**.
*   Frameworks: **PyTorch** preferred for embeddings (concise and well-supported), but **TensorFlow** acceptable if necessary.
*   Prototype code must be **as concise as possible** while still functional. Avoid unnecessary abstractions.
*   Directory structure (shared with Phase 1 detector work):

    ```
    immich-dogs/
      data/
        dogfacenet/    # identity-labeled dataset
        stanford_dogs/ # for breed-aware hard negatives
      models/
      scripts/
        phase2/
      training/
      .planning/
    ```
*   Every time you execute a prompt, **log progress in** `.planning/prompt-X.md` (replace X with the prompt number).

## Goals of Phase 2:

1.  Implement a **dog embedding model** (e.g., MobileNetV3-small or EfficientNet-lite backbone).
2.  Train it using a **unified, verifiable identity dataset** created from source files.
3.  Use **metric learning losses** (ArcFace, or Triplet with hard-negative mining).
4.  Export embeddings in a standard format (e.g., `.pt` file, ONNX).
5.  Validate with similarity search and basic verification metrics.

---

# Prompt 2.1 – Unified Identity Dataset Preparation

**Task:** Create a unified JSON-based dataset from source identity images and a corresponding PyTorch data loader.

**Prerequisites:**

*   Download **DogFaceNet** dataset into `E:\Code\GitHub\immich-dogs\data\dogfacenet`.
*   Download **Stanford Dogs** dataset into `E:\Code\GitHub\immich-dogs\data\stanford_dogs` for breed data.
*   Ensure **PyTorch** is installed.

**Instructions:**

*   In `scripts/phase2/prepare_identity_dataset.py`, implement a script that scans `data/dogfacenet` and `data/stanford_dogs`.
*   The script should create `data/identity_train.json` and `data/identity_val.json` files with a defined train/val split.
*   Each JSON entry should contain `file_path`, an integer `identity_label`, and a `breed_label` (derived from Stanford Dogs if possible). This decouples data aggregation from the training framework.
*   In `training/datasets.py`, implement an `IdentityDataset` class that reads from these JSON files. It should return `(image_tensor, label)` and apply augmentations (random crop, flip, color jitter).
*   Create a validation script `scripts/phase2/visualize_identity_dataset.py` to display sample images with their labels from the JSON files to verify correctness before training.

**Log Progress:** Save notes in `.planning/prompt-2.1.md`.

---

# Prompt 2.2 – Embedding Model Architecture

**Task:** Create a lightweight embedding network.

**Prerequisites:**

*   Completed Prompt 2.1 (dataset available).

**Instructions:**

*   In `models/embedding.py`, implement a **MobileNetV3-small** backbone truncated before classification.
*   Add a projection head:
    *   Global Average Pooling → Fully Connected (512-D embedding) → L2 normalization.
*   Make the code modular so backbones can be swapped (EfficientNet, ResNet).
*   Write a test script `notebooks/test_model_forward.ipynb` to confirm embeddings shape `(batch, 512)`.

**Log Progress:** Save notes in `.planning/prompt-2.2.md`.

---

# Prompt 2.3 – Metric Learning Loss Functions with Hard-Negative Support

**Task:** Add support for identity-preserving loss functions, including a strategy for hard-negative mining.

**Prerequisites:**

*   Completed Prompt 2.2.

**Instructions:**

*   In `training/losses.py`, implement:
    *   **ArcFace** (with scale and margin hyperparameters).
    *   **Triplet loss** (with a sampler that supports semi-hard or breed-aware negative mining).
*   Default: use ArcFace for training.
*   Write unit tests in `notebooks/test_losses.ipynb` to validate losses produce non-zero gradients.

**Log Progress:** Save notes in `.planning/prompt-2.3.md`.

---

# Prompt 2.4 – Training Script with Hard-Negative Sampling

**Task:** Train the embedding model on the unified identity dataset.

**Prerequisites:**

*   Dataset prepared (Prompt 2.1).
*   Model defined (Prompt 2.2).
*   Loss implemented (Prompt 2.3).

**Instructions:**

*   In `training/train_embedding.py`:
    *   Load the `IdentityDataset` with the train/val JSON files.
    *   Implement a custom batch sampler to be used with the DataLoader. This sampler will enable breed-aware hard-negative mining for the Triplet Loss.
    *   Use Adam optimizer, learning rate scheduler.
    *   Train for \~10 epochs as prototype.
    *   Save best model as `models/dog_embedding.pt`.
*   Enable GPU training with `torch.cuda.is_available()`.
*   Print training logs (loss curves, accuracy).

**Log Progress:** Save notes in `.planning/prompt-2.4.md`.

---

# Prompt 2.5 – Embedding Validation and Export

**Task:** Validate embeddings with quantitative metrics and qualitative visualization.

**Prerequisites:**

*   Trained model checkpoint from Prompt 2.4.

**Instructions:**

*   In `notebooks/validate_embeddings.ipynb`:
    *   Load validation embeddings.
    *   Compute cosine similarity matrix.
    *   Report **verification metrics**: TAR\@FAR=1e-3.
*   Extend `scripts/phase2/visualize_identity_dataset.py` to:
    *   Load a query image and its embedding.
    *   Display its nearest neighbors from the validation set to visually inspect embedding quality.
*   Export model to **ONNX** format (`models/dog_embedding.onnx`) for portability.

**Log Progress:** Save notes in `.planning/prompt-2.5.md`.

---

# Prompt 2.6 – Pipeline Integration Prototype

**Task:** Provide a minimal end-to-end prototype for embeddings.

**Prerequisites:**

*   Completed Prompts 2.1–2.5.

**Instructions:**

*   In `utils/embed_images.py`:
    *   Load model.
    *   Take an image folder as input.
    *   Output embeddings (`.npy` or `.pkl`) keyed by filename.
*   Provide a CLI example in PowerShell:

    ```powershell
    python utils/embed_images.py --input data/dogfacenet/identity_001 --output embeddings.pkl
    ```
*   Verify embeddings can be clustered (DBSCAN/HDBSCAN).

**Log Progress:** Save notes in `.planning/prompt-2.6.md`.