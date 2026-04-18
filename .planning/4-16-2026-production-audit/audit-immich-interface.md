# Immich Interface Contract Audit

**Date:** 2026-04-16  
**Scope:** Model I/O contract between immich-animals ONNX outputs and the Immich ML face-recognition pipeline. Out of scope: `copy_models.sh`, `docker-compose.dogs.yml`, `REPLICATION.md`, hijack integration branch.

---

## Executive Summary

The immich-animals pipeline and Immich's facial-recognition pipeline share a superficial similarity (detect → align/crop → embed) but differ fundamentally in almost every contract detail. **No animal-ID ONNX model can drop into Immich natively without code changes.** The most critical gaps are:

1. Immich's recognition model is driven by **InsightFace's `ArcFaceONNX.get_feat()`**, which expects **BGR, `(N, 112, 112, 3)` HWC uint8 crops** produced by a specific 5-landmark affine warp (`insightface.utils.face_align.norm_crop`). Our embedding model expects **RGB, `(1, 3, 224, 224)` CHW float32, [0,1]-normalised**.
2. Immich's detector is **RetinaFace** (via InsightFace), fixed at **640×640**, returning both bounding boxes **and 5 facial landmarks** (`(N, 5, 2)`). Our detector is **YOLO11n** (opset-12 ONNX with NMS baked in), output shape `(1, N, 6)`, and returns **no landmarks at all**.
3. Immich wraps everything in a strict model-registry/cache system keyed on known model names (`buffalo_s`, `buffalo_l`, …). Our ONNX files have arbitrary filenames and are not registered anywhere in Immich's codebase.
4. The final embedding output from Immich is serialised as a **JSON string** (`orjson.dumps`) and stored as text. Our pipeline returns a raw **numpy float32 array**; the serialisation layer is completely absent.

---

## 1. Immich's Contract

### 1.1 Face Detection (`FaceDetector`)

**File:** `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/facial_recognition/detection.py`

| Property | Value |
|---|---|
| Backing library | `insightface.model_zoo.RetinaFace` (InsightFace ≥ 0.7.3) |
| ONNX loaded via | `OrtSession` wrapping `ort.InferenceSession` |
| Input format | `NDArray[np.uint8]` or raw `bytes`; decoded to BGR `cv2.Mat` by `decode_cv2()` |
| Input color order | **BGR** (OpenCV default; `decode_cv2` converts PIL→BGR at line 47 of `transforms.py`) |
| Model input size | Fixed **(640, 640)** — set at `detection.py:23` via `model.prepare(input_size=(640,640))` |
| Input tensor shape | `(1, 3, 640, 640)` float32, handled internally by InsightFace |
| Input normalization | InsightFace internal: subtract 127.5, divide by 128 (standard RetinaFace preprocessing) |
| Output (raw) | `model.detect()` returns `(bboxes, landmarks)` tuple |
| `bboxes` shape | `(N, 5)` — columns `[x1, y1, x2, y2, score]` |
| `landmarks` shape | `(N, 5, 2)` — **5 facial keypoints**, xy coordinates, used directly for alignment |
| Python output dict | `{"boxes": NDArray[float32] (N,4), "scores": NDArray[float32] (N,), "landmarks": NDArray[float32] (N,5,2)}` (see `detection.py:31-35`) |
| Score threshold | Default 0.7; configurable via `minScore` |

**Key point:** The `landmarks` tensor is not optional — it is the **required input** to `norm_crop` in the recognition step.

### 1.2 Face Recognition / Embedding (`FaceRecognizer`)

**File:** `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/facial_recognition/recognition.py`

| Property | Value |
|---|---|
| Backing library | `insightface.model_zoo.ArcFaceONNX` |
| Crop method | `insightface.utils.face_align.norm_crop(image, landmark)` per face — 5-landmark affine warp to **112×112** (`recognition.py:77`) |
| Crop input format | BGR `cv2.Mat` uint8 |
| Crop output shape | `(112, 112, 3)` uint8 BGR — confirmed by `test_main.py:776` |
| Model input (batched) | `(batch, 3, 112, 112)` float32 — ArcFaceONNX transposes HWC→CHW and normalises internally |
| ArcFaceONNX normalisation | Subtract 127.5, divide by 127.5 (standard ArcFace/InsightFace) |
| Embedding dim | **512** — confirmed by `test_main.py:769` |
| L2 normalised? | **Yes** — ArcFaceONNX calls `sklearn.preprocessing.normalize` on outputs |
| Distance metric | Cosine similarity (equivalent to dot product on L2-normalised vectors) |
| Batch axis handling | If model has static batch axis = 1, `_add_batch_axis()` patches ONNX proto to `"batch"` dim — `recognition.py:79-87` |
| Output stored as | **JSON string** via `orjson.dumps(embedding)` (`transforms.py:80`, `recognition.py:70`) |
| Final output per face | `{"boundingBox": {x1,y1,x2,y2}, "embedding": "<json_str>", "score": float}` |

### 1.3 Model Registry / Discovery

**File:** `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/constants.py:70-75`

Immich recognises only four InsightFace model names: `antelopev2`, `buffalo_s`, `buffalo_m`, `buffalo_l`. Models are downloaded from HuggingFace Hub as `immich-app/<model_name>` and placed at a path derived from `model_type` + `model_name`. ONNX files must be at `<cache>/<task>/<model_name>/detection/model.onnx` and `…/recognition/model.onnx`.

---

## 2. Our Contract

### 2.1 Detection (`ONNXDetector`)

**File:** `/mnt/e/Code/GitHub/immich-animals/animal_id/pipeline/onnx_models.py:10-53`

| Property | Value |
|---|---|
| Model | YOLO11n exported with `model.export(format="onnx", opset=12, nms=True)` (`train_master.py:290`) |
| Input shape | `(1, 3, H, W)` where H×W is whatever the YOLO model was trained with (640×640 default); read dynamically from session at `onnx_models.py:15` |
| Input format | RGB float32 `[0, 1]` — `onnx_models.py:51-53` |
| Color order | **RGB** (image converted from BGR→RGB by `cv2.cvtColor` in `ambidextrous_axolotl.py:112`) |
| Normalisation | Divide by 255.0 only — no mean/std subtraction |
| Output shape | `(1, N, 6)` — columns `[x1, y1, x2, y2, conf, class_id]` (YOLO NMS-included format) |
| Output coordinates | Absolute pixel coords at input resolution (scaled back to original at `onnx_models.py:31-35`) |
| **Landmarks** | **None — not present anywhere in the detector output** |
| Post-processing | Simple confidence threshold at 0.1; no NMS (baked into ONNX export) |

### 2.2 Keypoint Model (`ONNXKeypoint`)

**File:** `/mnt/e/Code/GitHub/immich-animals/animal_id/pipeline/onnx_models.py:56-92`

| Property | Value |
|---|---|
| Model | YOLO11n-pose exported with `nms=True` (`scripts/12_export_keypoint_onnx.py:66`) |
| Input shape | `(1, 3, H, W)`, RGB, [0,1] float32 |
| Output | `(1, N, 7+)` — `[x,y,w,h, conf, cls, kp1x, kp1y, kp1c, …]`, 4 facial keypoints |
| Landmarks format | 4 keypoints `(x, y, conf)` per detection |
| Status | **Disabled by default** (`use_keypoints=False`, `ambidextrous_axolotl.py:47`); benchmarks show it hurts metrics |

### 2.3 Embedding (`ONNXEmbedding`)

**File:** `/mnt/e/Code/GitHub/immich-animals/animal_id/pipeline/onnx_models.py:95-115`; exported in `train_master.py:401-454`

| Property | Value |
|---|---|
| Model | ResNet50 + projection head; exported via `torch.onnx.export` opset 12 |
| Input name | `"input"` (set at `train_master.py:449`) |
| Output name | `"output"` (set at `train_master.py:450`) |
| Input shape | `(1, 3, 224, 224)` float32 (dynamic batch axis via `dynamic_axes`) |
| Input format | RGB, `[0, 1]`, **no mean/std normalisation** (`onnx_models.py:112-114`) |
| Resize method | `cv2.INTER_AREA` (`onnx_models.py:112`) |
| Color order | RGB (pipeline converts BGR→RGB at pipeline entry) |
| Embedding dim | **512** (`embedding/config.py:EMBEDDING_DIM`) |
| L2 normalised? | **Yes** — `F.normalize(embeddings, p=2, dim=1)` baked in at forward pass (`embedding/models.py:81`) |
| Distance metric | **Cosine similarity** (`sklearn.metrics.pairwise.cosine_similarity`, `ambidextrous_axolotl.py:84`) |
| Output type | Raw `np.ndarray` float32; no serialisation |
| Input crop source | 10%-padded bbox crop from detector, then resized to 224×224 (`ambidextrous_axolotl.py:138-141`) |

---

## 3. Side-by-Side Contract Comparison

| Aspect | Immich Expects | We Produce | Match? |
|---|---|---|---|
| **Detection — model architecture** | RetinaFace (InsightFace) | YOLO11n | No |
| **Detection — input size** | 640×640 | 640×640 (default) | Yes |
| **Detection — input dtype** | uint8 BGR (converted internally to float) | float32 RGB [0,1] | No |
| **Detection — normalisation** | InsightFace: (x-127.5)/128 | YOLO: /255.0 | No |
| **Detection — output: bboxes** | `(N, 4)` float32, pixel coords | `(N, 4)` int, pixel coords (after scaling) | Structurally compatible |
| **Detection — output: scores** | `(N,)` float32 | float inside list-of-dicts | Different container |
| **Detection — output: landmarks** | `(N, 5, 2)` float32 **required** | **Absent — not produced** | **Hard mismatch** |
| **Recognition — crop method** | `norm_crop()` affine warp to 112×112 BGR uint8 | Simple bbox crop + 10% padding, resized to 224×224 RGB float | No |
| **Recognition — crop size** | **112×112** | **224×224** | No |
| **Recognition — input dtype** | uint8 HWC BGR → internally to CHW float | float32 CHW RGB [0,1] | No |
| **Recognition — normalisation** | (x-127.5)/127.5 (ArcFace standard) | /255.0 only, no mean/std | No |
| **Recognition — input name** | determined by ONNX model (ArcFaceONNX introspects) | `"input"` | Cosmetic (Immich is flexible) |
| **Recognition — output name** | determined by ONNX model | `"output"` | Cosmetic (Immich is flexible) |
| **Recognition — embedding dim** | 512 | 512 | Yes |
| **Recognition — L2 normalised** | Yes (ArcFaceONNX) | Yes (EmbeddingNet.forward) | Yes |
| **Recognition — distance metric** | Cosine (dot product on unit vectors) | Cosine | Yes |
| **Recognition — output serialisation** | `orjson.dumps` → JSON string | Raw numpy array | No (integration layer missing) |
| **Batch axis** | Dynamic `"batch"` dim injected if needed | Dynamic via `dynamic_axes` | Compatible |
| **Model names / registry** | `buffalo_s/l`, `antelopev2` only | Arbitrary filenames | No |
| **Keypoints — count** | **5** facial landmarks | **4** keypoints (custom) | No |
| **Keypoints — usage** | Mandatory for alignment (norm_crop) | Optional, disabled by default | Incompatible role |

---

## 4. Mismatches and Severity

### 4.1 Hard Incompatibilities (Immich would crash or produce wrong output)

**M1 — No landmarks from detector (Severity: CRASH)**  
Immich `FaceRecognizer._crop()` calls `norm_crop(image, landmark)` for every detected face (`recognition.py:77`). Without a `landmarks` key in the detection output, this raises a `KeyError`/crash. Our `ONNXDetector` returns only `bbox` and `confidence` — there is no landmark tensor anywhere in the YOLO pipeline.  
- *Our file:* `animal_id/pipeline/onnx_models.py:25-44`  
- *Immich file:* `machine-learning/immich_ml/models/facial_recognition/recognition.py:77`

**M2 — Detector output structure mismatch (Severity: CRASH)**  
Immich `FaceDetector._predict()` returns a Python dict `{"boxes": NDArray, "scores": NDArray, "landmarks": NDArray}`. Our `ONNXDetector.predict()` returns a `List[Dict]` with keys `bbox`, `confidence`, `class`. The containers and key names are completely different. Passing our output directly to `FaceRecognizer._predict()` would crash immediately.  
- *Our file:* `animal_id/pipeline/onnx_models.py:38-44`  
- *Immich file:* `machine-learning/immich_ml/models/facial_recognition/detection.py:31-35`, `machine-learning/immich_ml/schemas.py:82-85`

**M3 — Recognition model input contract incompatible (Severity: WRONG OUTPUT / CRASH)**  
`ArcFaceONNX.get_feat()` expects `(112, 112, 3)` BGR uint8 crops obtained from affine-warped face landmarks. Our ONNX embedding model expects `(1, 3, 224, 224)` RGB float32 `[0,1]` input. Feeding the wrong format to either model produces garbage embeddings or a shape error.  
- *Our file:* `animal_id/pipeline/onnx_models.py:110-115`  
- *Immich file:* `machine-learning/immich_ml/models/facial_recognition/recognition.py:56-58`, `test_main.py:776`

**M4 — Model not in Immich's registry (Severity: CRASH)**  
Immich `ModelCache.get()` and `from_model_type()` only accept names in `_INSIGHTFACE_MODELS = {"antelopev2","buffalo_s","buffalo_m","buffalo_l"}` for facial-recognition tasks. Any other name raises `ValueError`. Our models have no registered name.  
- *Immich file:* `machine-learning/immich_ml/models/constants.py:70-75`

### 4.2 Subtle Semantic Mismatches (runs but metrics differ)

**M5 — Normalisation mismatch for embedding (Severity: WRONG OUTPUT)**  
ArcFaceONNX normalises crops as `(x - 127.5) / 127.5`. Our `ONNXEmbedding._preprocess()` normalises as `x / 255.0`. These produce the same values only if mean = 0, which it isn't. Even though both produce numbers in `[-1, 1]` for ArcFace vs `[0, 1]` for ours, feeding ArcFace-normalised crops to our model (or vice versa) would produce shifted embeddings.  
- *Our file:* `animal_id/pipeline/onnx_models.py:112`  
- *Immich file:* ArcFaceONNX internal (insightface library)

**M6 — Crop geometry mismatch undermines embedding quality (Severity: SUBTLE)**  
Immich crops via affine warp (`norm_crop`) based on 5 facial landmarks, producing a geometrically normalised 112×112 face image. We crop using a 10%-padded axis-aligned bounding box, then resize to 224×224. Even with a perfect detector, the crop fed to our model contains significantly more background and is not pose-normalised. This degrades embedding quality for identity matching.  
- *Our file:* `animal_id/pipeline/ambidextrous_axolotl.py:138-141`  
- *Immich file:* `machine-learning/immich_ml/models/facial_recognition/recognition.py:76-77`

**M7 — Keypoint count and role (Severity: SUBTLE)**  
Immich requires 5 facial landmarks for affine alignment. We have a keypoint model producing 4 dog facial keypoints, and it is disabled by default. These cannot substitute for Immich's 5-landmark scheme — even if enabled, `norm_crop` requires exactly 5 points in a specific order (left eye, right eye, nose tip, left mouth, right mouth).  
- *Our file:* `animal_id/pipeline/onnx_models.py:76` (4 keypoints), `ambidextrous_axolotl.py:47` (disabled)  
- *Immich file:* `recognition.py:77`

### 4.3 Cosmetic Issues

**M8 — ONNX tensor names (Severity: LOW)**  
Our embedding ONNX exports with names `"input"` and `"output"` (`train_master.py:449-450`). Immich's `ArcFaceONNX` introspects `session.get_inputs()[0].name` dynamically, so names are not hardcoded — compatible as long as there is exactly one input and output.

**M9 — Missing embedding serialisation (Severity: INTEGRATION ONLY)**  
Immich serialises embeddings to JSON strings via `serialize_np_array()` (`transforms.py:79-80`). Our pipeline returns raw `np.ndarray`. This is not a model-contract issue per se but must be added in any adapter layer.

---

## 5. Required Changes for Drop-In Compatibility

For our animal ONNX models to be consumed by Immich's existing `FaceDetector` + `FaceRecognizer` classes **without modifying Immich**, the following would be required:

### 5.1 Detection model must be replaced by a RetinaFace-compatible model

Immich's `FaceDetector` hard-wires InsightFace's `RetinaFace` class. Replacing our YOLO detector requires either:
- **Option A:** Train or adapt a RetinaFace model on dog images, exporting to the InsightFace-compatible ONNX format (input: 640×640, output: anchored predictions decoded by InsightFace). This is the only path that requires zero Immich changes.
- **Option B (simpler):** Write a custom `FaceDetector` subclass that wraps our YOLO ONNX model and produces the `{"boxes":..., "scores":..., "landmarks":...}` dict with fake/approximate 5-point landmarks derived from bounding box corners. This would work if the landmark alignment quality is acceptable.

### 5.2 Landmarks must be synthesised or produced (Hard requirement)

A 5-landmark output is mandatory for `norm_crop`. At minimum, four corners + centre of the bounding box could be used as proxy landmarks, but the affine warp will produce poor crops. The proper fix is to produce real anatomical landmarks (eyes, nose, mouth corners) from our keypoint model — but this requires:
- Expanding keypoints from 4 to 5 (or remapping)
- Re-enabling the keypoint pipeline in inference

### 5.3 Recognition model must accept 112×112 BGR uint8 crops

Our embedding model currently expects 224×224 RGB float32. To be called by `ArcFaceONNX`, we would need to:
- Re-export with input `(N, 3, 112, 112)` (or swap to HWC; ArcFaceONNX converts internally)
- Change preprocessing in training/export to use ArcFace normalisation `(x-127.5)/127.5`
- Retrain on 112×112 crops with ArcFace normalisation to avoid distribution shift

### 5.4 Model must be registered in Immich's name registry

Add a custom model name to `_INSIGHTFACE_MODELS` in `constants.py:70` and ensure the ONNX files are placed at the expected cache paths.

### 5.5 Adapter path (minimal Immich changes)

If modifying Immich source is acceptable, the minimum viable adapter is:
1. Add an `AnimalDetector` subclass of `InferenceModel` that wraps the YOLO ONNX and returns the `FaceDetectionOutput` dict (with synthesised landmarks).
2. Add an `AnimalRecognizer` subclass that wraps our embedding ONNX, calls the correct preprocessing (resize to 224, /255.0, RGB→CHW), and returns `FacialRecognitionOutput`.
3. Register the new model names.
4. Wire them into the model task routing in `models/cache.py`.

---

## 6. File Reference Index

| File | Purpose |
|---|---|
| `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/facial_recognition/detection.py` | Immich face detector (RetinaFace wrapper) |
| `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/facial_recognition/recognition.py` | Immich face recognizer (ArcFaceONNX wrapper) |
| `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/transforms.py` | Immich image decode/normalise utils |
| `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/schemas.py` | Immich type contracts (FaceDetectionOutput, etc.) |
| `/mnt/e/Code/GitHub/immich-app/machine-learning/immich_ml/models/constants.py` | Immich model registry (known model names) |
| `/mnt/e/Code/GitHub/immich-app/machine-learning/test_main.py:718-776` | Tests confirming crop shape (112,112,3), embedding dim 512 |
| `/mnt/e/Code/GitHub/immich-animals/animal_id/pipeline/onnx_models.py` | Our ONNX wrappers (detector, keypoint, embedding) |
| `/mnt/e/Code/GitHub/immich-animals/animal_id/pipeline/ambidextrous_axolotl.py` | Our inference orchestrator (crop geometry, cosine sim) |
| `/mnt/e/Code/GitHub/immich-animals/animal_id/embedding/models.py` | EmbeddingNet (512-dim, L2-normalised output) |
| `/mnt/e/Code/GitHub/immich-animals/animal_id/embedding/config.py` | Embedding config (IMG_SIZE=224, EMBEDDING_DIM=512) |
| `/mnt/e/Code/GitHub/immich-animals/scripts/train_master.py:279-295,401-454` | ONNX export logic (input/output names, opset, dynamic axes) |
| `/mnt/e/Code/GitHub/immich-animals/animal_id/common/constants.py:29-31` | ONNX file paths |
