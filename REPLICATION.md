# Replicating the Animal Identification Setup

This guide explains how to set up the "Pet Hijack" feature in Immich, allowing facial recognition to detect and cluster dogs (and potentially other animals) instead of humans.

**DO NOT FOLLOW THESE INSTRUCTIONS WITH ANY NON-DEV IMMICH INSTANCES. THIS IS NOT STABLE AND COULD SEVERELY DAMAGE YOUR IMMICH DATABASE**

## 1. Prerequisites

*   Linux or WSL2 environment
*   Docker & Docker Compose
*   Python 3.10+ (for training scripts)

## 2. Directory Structure

We recommend the following directory structure:

```
workspace/
├── immich-animals/      (This repository)
└── immich-app/          (The official Immich source code)
```

## 3. Setup Steps

### 3.1. Clone Repositories

```bash
mkdir workspace && cd workspace

# 1. Clone this repository
git clone https://github.com/rtp4jc/immich-animals.git
cd immich-animals

# 2. Clone Immich source code nearby
git clone https://github.com/immich-app/immich ../immich-app
```

### 3.2. Apply the "Pet Hijack" Changes

We have created a branch in the Immich repository that contains the necessary code changes to "hijack" the facial recognition pipeline.

```bash
cd ../immich-app

# Add our fork as a remote
git remote add rtp4jc https://github.com/rtp4jc/immich-app.git
git fetch rtp4jc

# Checkout the hijack branch
git checkout rtp4jc/hijack
```

*Note: This branch includes the necessary "Foundation" classes (`AnimalDetector`, `AnimalRecognizer`) and the "Hijack" configuration that redirects human face detection requests to our animal models.*

### 3.3. Train or Download Models

You need trained ONNX models (`detector.onnx` and `embedding.onnx`). Follow the training instructions in `README.md` to run `scripts/train_master.py`. The output models will be saved to `models/onnx/`.

### 3.4. Deploy Models

Copy the ONNX models from `immich-animals` to the `immich-app` build context.

```bash
cd ../immich-animals
./copy_models.sh ../immich-app
```

### 3.5. Build and Run

Navigate to the Immich Docker directory and start the stack using the custom configuration.

```bash
cd ../immich-app/docker

# Start the stack (this will build the custom 'immich-ml-dogs' container)
docker compose -f docker-compose.dev.yml -f docker-compose.dogs.yml up -d --build
```

## 4. Verification

1.  Open Immich Web at `http://localhost:3001`.
2.  Login (default admin/admin or create an account).
3.  Go to **Administration > Jobs > Facial Recognition**.
4.  Run "All".
5.  Watch the "People" tab. Your dogs should start appearing as detected "people".

## 5. Development Notes

*   **Foundation Branch**: The branch `feat/animal-foundation` contains the clean core logic without the hijack hacks, suitable for future PRs.
*   **Hijack Branch**: The branch `rtp4jc/hijack` builds on the foundation and adds the configuration overrides to force `buffalo_l` requests to use the animal models.
