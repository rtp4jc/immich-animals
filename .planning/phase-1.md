Below is a comprehensive, ready-to-run set of LLM-agent prompts designed to implement **Phase 1 — basic dog detection model with landmarks** in the `E:\Code\GitHub\immich-dogs` repository on Windows (Conda + one CUDA GPU). Each prompt includes: an explicit task, exactly-stated prerequisites the user must satisfy before running the agent, the commands/files the agent should create, what the agent must log to `.planning/prompt-X.md`, and short guidance to keep code concise (prototype-first). An **overarching prompt** is provided that should be prepended to every prompt so the agent has consistent background (frameworks, folder layout, style expectations, constraints).

Use these prompts by providing them to an LLM agent that has file-system access to the local repo. Each prompt is self-contained and instructs the agent to update progress in a planning file. Be explicit: every prompt writes its progress and results into `.planning/prompt-X.md`.

---

# Overarching prompt (prelude; include with every prompt)

> You are an autonomous LLM-based code agent with access to the repository at `E:\Code\GitHub\immich-dogs` on a Windows machine. All actions you take must be deterministic and logged to the repository. Work concisely: this is a prototype; prefer minimal, well-commented scripts over heavy frameworks. Use Python 3.12+, PyTorch with CUDA, and **Ultralytics YOLOv8** (pose/keypoint variant) for a single model that outputs bounding boxes and landmarks (keypoints). Export artifacts under `E:\Code\GitHub\immich-dogs\outputs\phase1\` and place scripts under `E:\Code\GitHub\immich-dogs\scripts\phase1\`. Whenever you run commands, they will be run on the windows command prompt, not bash. Use conda environment name `python312`. Always create or update `.planning/prompt-X.md` describing: (1) what you did, (2) commands run, (3) files created, (4) test results / metrics, (5) next steps / blockers. Commit code changes on branch `phase1/detection` with small commits. Keep scripts short (≤ 200 LOC each) and well-documented. Do not modify other repository areas without logging. Assume one CUDA GPU is available as `torch.cuda.is_available()`; verify early. If you need additional files/data the user must download, specify exact local destination paths and do not attempt to fetch them yourself. Use relative repo paths from `E:\Code\GitHub\immich-dogs`.

NEVER read a full file under the data directory. Ask the user to provide sample content if you need to see the contents. You are able to write scripts that parse and work with files in the data directory, but DO NOT read the full file content yourself.

---

# Prompt 1 — Setup dev environment & verify GPU

**Task (agent):** Create a short PowerShell setup script that creates/activates the `python312` conda environment, installs dependencies (PyTorch + CUDA, ultralytics, pycocotools, opencv-python, pillow, tqdm, pandas), and runs a GPU test. Add an easy-to-run README snippet and write progress to `.planning/prompt-1.md`.

**Prerequisites (user must do first):**

* Install Anaconda/Miniconda and ensure `conda` is on PATH.
* CUDA toolkit and drivers already installed and compatible with the installed PyTorch (user says GPU is present).
* Ensure repo path `E:\Code\GitHub\immich-dogs` exists and you have write permissions.
* Have at least 20 GB free disk space.

**What to create (agent):**

1. `scripts/phase1/setup_env.ps1` — PowerShell script that:

   * Creates conda env:

     ```powershell
     conda create -n python312 python=3.12 -y
     conda activate python312
     ```
   * Installs PyTorch with the right CUDA for the system. Use the recommended `pip` command but include a guard to use `torch` with CUDA if available. Example (concise):

     ```powershell
     pip install -U pip
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install ultralytics==8.* opencv-python pycocotools pillow tqdm pandas
     ```

     (Note: agent should include a comment instructing user to adjust `cu121` to their CUDA version if needed.)
   * Prints `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
2. `README_SETUP.md` snippet added to repo root with single-line run instructions.
3. `.planning/prompt-1.md` with a template describing commands to run, sample expected outputs, and places to paste the GPU print output.

**What to log (in `.planning/prompt-1.md`):**

* Commands executed (copied verbatim).
* Output of `python -c "import torch; print(torch.cuda.is_available())"` and `torch.version`.
* Any errors, and the recommended remediation (e.g., change wheel index for CUDA version).
* Files created (`scripts/phase1/setup_env.ps1`, `README_SETUP.md`).

**Notes for agent:**

* Keep the script compact and robust: detect CUDA compatibility and advise if mismatch.
* Do not attempt to install enormous extras (no heavy dev tools).
* Commit created files to `phase1/detection` branch.

---

# Prompt 2 — Verify dataset presence & create sample manifest

**Task (agent):** Create a verification script that checks for required datasets at expected local paths, enumerates number of images and annotation files, and produces a small sample manifest (10 images per dataset) under `data/manifest/phase1_samples.csv`. Log in `.planning/prompt-2.md`.

**Prerequisites (user must do first):**

* Download or place the datasets at these exact paths (agent must refuse to download external data):

  * COCO 2017 images & annotations (or smaller subset) at `E:\Code\GitHub\immich-dogs\data\coco\` (images in `E:\Code\GitHub\immich-dogs\data\coco\train2017` and `val2017`, annotations in `E:\Code\GitHub\immich-dogs\data\coco\annotations\instances_train2017.json`).
  * Stanford Dogs at `E:\Code\GitHub\immich-dogs\data\stanford_dogs\` (images folder).
  * DogFaceNet / identity-labeled dog head crops (if available) at `E:\Code\GitHub\immich-dogs\data\dogfacenet\` (images/ + metadata if present).
  * Oxford-IIIT Pets at `E:\Code\GitHub\immich-dogs\data\oxford_pets\`.
* If user cannot download everything, at minimum COCO + Stanford Dogs should exist.

**What to create (agent):**

1. `scripts/phase1/verify_datasets.py` — python script that:

   * Accepts dataset root paths as args or defaults to the required paths.
   * Verifies presence and counts images and annotations for each dataset.
   * Writes `data/manifest/phase1_samples.csv` with columns: `dataset,rel_image_path,abs_path,label_hint` (label\_hint can be `dog`, `unknown`, or `dog-head` if known).
   * Saves a small sample images copy (or symlink list) to `outputs/phase1/sample_images/` (if copy, do up to 10 images per dataset).
2. `.planning/prompt-2.md` capturing outputs and any missing dataset notes.

**What to log (in `.planning/prompt-2.md`):**

* Counts of images and annotations per dataset.
* Paths missing / steps the user needs to finish.
* The absolute path of `data/manifest/phase1_samples.csv` and the sample output images list.

**Notes for agent:**

* Keep code concise — just checks + CSV output.
* Do not attempt heavy I/O copy if datasets are huge; list and create small sample extracts (<10 images per dataset).
* Commit the script and manifest.

---

# Prompt 3 — Convert available keypoints / masks to COCO-style keypoints annotations

**Task (agent):** Implement a conversion utility that generates a COCO-format annotation file containing: for each image a `bbox` (x,y,w,h) and `keypoints` array for the chosen dog landmarks. Landmarks should be a compact set: `[left_eye_x, left_eye_y, v, right_eye_x, right_eye_y, v, nose_x, nose_y, v, left_ear_x, left_ear_y, v, right_ear_x, right_ear_y, v]` where `v` is visibility (0/1/2). Use sources that include landmarks (StanfordExtra) or synthesize approximate keypoints when you can (see rules below). Log everything into `.planning/prompt-3.md`.

**Prerequisites (user must do first):**

* StanfordExtra keypoint files available at `E:\Code\GitHub\immich-dogs\data\stanford_dogs\stanford_extra_keypoints.json` OR the user can provide a smaller keypoint CSV.
* Oxford mask PNGs in `E:\Code\GitHub\immich-dogs\data\oxford_pets\annotations\` if available.
* DogFaceNet crops directory (optional) at `E:\Code\GitHub\immich-dogs\data\dogfacenet\images\`.

**Conversion rules (agent must follow):**

* If dataset provides explicit landmarks (StanfordExtra), map them to our 5-point schema (approximate or map nearest named points).
* If only masks are available (Oxford), compute a coarse head-centroid and set `nose` to centroid; set eyes/ears as heuristic offsets from centroid based on mask geometry (this is weak supervision — label `v = 1`).
* If dataset only provides bounding boxes (COCO `dog`), include bbox and set all keypoints to zeros (`v = 0`) — still useful for detector training.
* Always set `category_id` to a single category `dog` with `id = 1`.
* Output COCO JSON to `data/coco_keypoints/annotations_train.json` and `annotations_val.json` (split by 90/10 rule across images from all available datasets).
* For each synthesized keypoint, set `v=1` (visible but approximate) except where truly missing (`v=0`).

**What to create (agent):**

1. `scripts/phase1/convert_to_coco_keypoints.py` — concise script implementing the rules above. It must:

   * Accept a small config at top to list input datasets and their types.
   * Produce `data/coco_keypoints/annotations_train.json` and `annotations_val.json`.
   * Generate small per-image debug JSON outputs for first 20 images to `outputs/phase1/debug_annotations/` showing original meta and final annotation.
2. `.planning/prompt-3.md` documenting conversion counts: #images with full keypoints, #with synthesized keypoints, #with only bbox.

**What to log (in `.planning/prompt-3.md`):**

* Exact command run.
* Number of images converted, and breakdown by annotation quality (full keypoints / synthesized / bbox-only).
* Paths to train/val JSON files.

**Notes for agent:**

* Keep conversion deterministic and short. Use standard COCO fields: `images`, `annotations`, `categories`.
* Ensure each annotation has `id` and image `id` consistent.
* Do not invent identities; this is detection+landmarks only.

---

# Prompt 4 — Create YOLOv8 dataset YAML and small validation split

**Task (agent):** Create a minimal YOLOv8 dataset config YAML referencing images and the COCO-style annotation JSON files created by step 3. Also create a script to create a lightweight validation split (200 images max) and to write `data/dogs_keypoints.yaml`. Log to `.planning/prompt-4.md`.

**Prerequisites (user must do first):**

* The COCO keypoints JSON files produced in Prompt 3 must exist at `data/coco_keypoints/annotations_train.json` and `annotations_val.json`.
* Image directories must be reachable (manifest from Prompt 2).

**What to create (agent):**

1. `data/dogs_keypoints.yaml` with keys:

   ```yaml
   train: E:/data/combined_images/train
   val: E:/data/combined_images/val
   nc: 1
   names: ['dog']
   ```

   (Use forward/back slashes consistently for YOLO; Windows paths okay.)
2. `scripts/phase1/make_yolo_splits.py` — script that:

   * Reads `data/coco_keypoints/annotations_train.json`.
   * Builds on-disk `train` and `val` lists limited to a validation pool of up to 200 images, creating `train/` and `val/` directories with symlink files or small copies if symlink unsupported (Windows: create small copies for sample).
   * Writes `data/dogs_keypoints.yaml`.
3. `.planning/prompt-4.md` to list images in train/val and any issues.

**What to log (in `.planning/prompt-4.md`):**

* Exact path to `data/dogs_keypoints.yaml`.
* Number of images in train and val.
* Any missing images or inconsistent annotations.

**Notes for agent:**

* Keep the YAML minimal and valid for Ultralytics `yolo` CLI.
* If copying images, copy only the small validation subset to avoid huge I/O; for training, prefer using the original image dirs in-place.

---

# Prompt 5 — Train the basic YOLOv8 pose model (detection + keypoints)

**Task (agent):** Create a concise training script that uses Ultralytics YOLOv8's CLI or Python API to train a `yolov8n-pose` (nano pose model) on `data/dogs_keypoints.yaml`. Save model artifacts to `models/phase1/`. Provide a short training wrapper PowerShell script `scripts/phase1/train_detector.ps1` that runs the training with sensible defaults (small epochs for prototype). Log to `.planning/prompt-5.md`.

**Prerequisites (user must do first):**

* Conda env installed & activated (Prompt 1).
* Data YAML and annotation JSONs exist (Prompt 3 & 4).
* GPU available and `torch.cuda.is_available()` returns True.

**What to create (agent):**

1. `scripts/phase1/train_detector.py` — minimal Python wrapper for Ultralytics API:

   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n-pose.pt')  # pretrained pose model
   model.train(data='data/dogs_keypoints.yaml', epochs=30, imgsz=640, batch=16, project='models/phase1', name='run1')
   ```

   (Make epochs small for prototype — user can extend.)
2. `scripts/phase1/train_detector.ps1` — PowerShell runner that activates conda env and runs `python train_detector.py` with real-time streaming logs to `outputs/phase1/logs/train.log`.
3. `.planning/prompt-5.md` with: training command, top-line metrics from last epoch, path to best.pt, and training logs path.

**What to log (in `.planning/prompt-5.md`):**

* Training start time, GPU device name, CUDA version.
* Final training metrics: bbox mAP50, keypoint mAP (if Ultralytics prints it), loss curves snapshot (path to plot).
* Full path to best weights (e.g., `models/phase1/run1/weights/best.pt`).

**Notes for agent:**

* Keep training script ≤ 80 LOC.
* If Ultralytics prints keypoint mAP, capture it; otherwise run a separate validation job (Prompt 6).
* Commit training scripts and PS wrapper.

---

# Prompt 6 — Run quick validation & save sample inference outputs

**Task (agent):** Implement a concise validation/inference script that (a) runs `yolo val` or uses the Ultralytics `model.val()` API on the `val` split and (b) runs inference on the `outputs/phase1/sample_images/` from Prompt 2 and writes visualized outputs into `outputs/phase1/inference/`. Update `.planning/prompt-6.md`.

**Prerequisites (user must do first):**

* Training completed and `best.pt` exists.
* Validation images exist (Prompt 4).

**What to create (agent):**

1. `scripts/phase1/validate_and_infer.py` which:

   * Loads `models/phase1/*/weights/best.pt`.
   * Calls `model.val(data='data/dogs_keypoints.yaml')` and captures metrics.
   * Runs `model.predict(source='outputs/phase1/sample_images', save=True, save_dir='outputs/phase1/inference')`.
   * Saves CSV `outputs/phase1/inference_report.csv` summarizing for each sample image: `image_path,detected_dog(bool),num_boxes,num_keypoints_detected,avg_score`.
2. `.planning/prompt-6.md` with results (val metrics, sample inference CSV path), and noting images where no dog was detected.

**What to log (in `.planning/prompt-6.md`):**

* Full validation metrics (mAP box/mAP keypoint).
* Top 10 images where detection failed (no box) and top 10 with low keypoint counts.
* Paths to visualized output images.

**Notes for agent:**

* Keep code small and friendly; rely on Ultralytics API for plotting/saving results.
* Ensure `save=True` writes overlaid keypoints for manual inspection.

---

# Prompt 7 — Export the trained model to ONNX and TFLite (basic)

**Task (agent):** Add a short script that exports `best.pt` to ONNX and TFLite using Ultralytics `model.export()` if available, or the minimal recommended steps. Save exported files to `models/phase1/export/`. Log steps to `.planning/prompt-7.md`.

**Prerequisites (user must do first):**

* `best.pt` must exist.
* Python packages needed for export (Ultralytics exports typically handle ONNX/TFLite; ensure appropriate packages installed from Prompt 1).

**What to create (agent):**

1. `scripts/phase1/export_model.py`:

   * Load best weights and run:

     ```python
     model = YOLO('models/phase1/run1/weights/best.pt')
     model.export(format='onnx', imgsz=640, save_dir='models/phase1/export')
     model.export(format='tflite', imgsz=640, save_dir='models/phase1/export')
     ```
   * Validate that `models/phase1/export/model.onnx` and `model.tflite` exist.
2. `.planning/prompt-7.md` with export logs, file sizes, and a short note: “Test these exported models on-device later with appropriate delegates.”

**What to log (in `.planning/prompt-7.md`):**

* Export commands run and stdout.
* Output file paths and sizes.
* Any warnings (e.g., missing ops during export).

**Notes for agent:**

* Keep script minimal. If TFLite export fails due to unsupported ops, record the failure and save ONNX at minimum.

---

# Prompt 8 — Minimal verification tests & commit

**Task (agent):** Add a simple test script and a final planning file summarizing Phase 1 status. Run the test locally if resources permit. Commit all changes on branch `phase1/detection`. Write `.planning/prompt-8.md`.

**Prerequisites (user must do first):**

* Completed all earlier steps; best weights and exported model exist.
* Python test runner available in conda env.

**What to create (agent):**

1. `scripts/phase1/run_smoke_tests.py`:

   * Loads the exported ONNX or TFLite model (prefer ONNX if simpler).
   * Runs inference on 3 sample images and asserts outputs saved to `outputs/phase1/inference_test/`.
   * Exit code non-zero on failure.
2. `.planning/prompt-8.md` — master summary for Phase 1:

   * Checklist of tasks 1–7 with status (done / todo / blocked).
   * Key artifacts & their paths:

     * `models/phase1/run1/weights/best.pt`
     * `models/phase1/export/model.onnx`
     * `data/coco_keypoints/annotations_train.json`
     * `data/dogs_keypoints.yaml`
     * Sample inference visualizations
   * Recommended next steps and any blockers (e.g., missing dataset portions).
3. Create a Git commit with message `phase1(detection): initial detection+keypoints prototype` and push branch `phase1/detection` (if remote configured). If push fails, log instructions.

**What to log (in `.planning/prompt-8.md`):**

* Smoke test pass/fail and stdout.
* A final statement: "Phase 1 prototype complete — ready for Phase 2."

**Notes for agent:**

* Keep tests tiny; their goal is reproducible smoke checks, not full CI.

---

# Example `.planning/prompt-X.md` template (agent must use for each prompt)

Each `.planning/prompt-X.md` must include the following sections:

```
# Prompt X — <short title>

## Date / Agent run id
YYYY-MM-DD / <agent id or run name>

## Goal
One-line task description.

## Prereqs (checked)
- [x] conda env created
- [x] datasets present at <paths>  (if not: list missing)

## Commands run
- <exact command 1>
- <exact command 2>

## Files created
- path/to/file1
- path/to/file2

## Outputs / Metrics
- metric1: value
- metric2: value
- sample images: outputs/phase1/...

## Issues / Errors
- <short note with stack traces if relevant>

## Next steps / recommended action
- <what to do next>
```

Agents must fill this template with accurate content.

---

# Additional implementation constraints and clarifications

* **Prototype minimalism:** Aim for the smallest working pipeline — correctness over completeness. Examples: small epoch counts, `yolov8n-pose` instead of larger backbones, small validation splits.
* **Windows specifics:** Use forward slashes or escaped backslashes consistently in Python. For shell commands in scripts, use PowerShell syntax in `*.ps1` wrappers. Example to activate conda in PowerShell:

  ```powershell
  conda activate python312
  python scripts/phase1/train_detector.py
  ```
* **Logging and reproducibility:** All runtime commands must be echoed to `.planning/prompt-X.md`. Save log files under `outputs/phase1/logs/`.
* **Privacy & data handling:** Do not transmit any dataset off the machine. The agent must not try to download dataset files; it should only verify and instruct the user.
* **Code style:** Keep each script short, with helpful docstrings and a `--help` flag (optparse/argparse) where appropriate.
* **Commits:** The agent must create small commits (one per prompt, ideally) on branch `phase1/detection` and include brief commit messages describing created artifacts.

---

# Final notes — what you (the human operator) must do before running the agent prompts

1. Place datasets in the exact paths listed in Prompt 2 prerequisites. If you use different local paths, edit the `verify_datasets.py` config in Prompt 2 or pass paths as args to the scripts.
2. Ensure conda is available and CUDA drivers are installed.
3. Provide permission for the agent to create files and commit to the repo.
4. Run Prompt 1 first. After it completes successfully and you confirm `torch.cuda.is_available()` is true, run Prompts 2→8 in order.

---

If you want, I will now produce the **exact content** for `scripts/phase1/setup_env.ps1`, `scripts/phase1/verify_datasets.py`, `scripts/phase1/convert_to_coco_keypoints.py`, `scripts/phase1/make_yolo_splits.py`, `scripts/phase1/train_detector.py`, `scripts/phase1/validate_and_infer.py`, `scripts/phase1/export_model.py`, `scripts/phase1/run_smoke_tests.py`, and example `.planning/prompt-X.md` files ready to drop into `E:\Code\GitHub\immich-dogs`. Tell me if you want those files generated now and I will create them one-by-one (or all at once) in the repository.
