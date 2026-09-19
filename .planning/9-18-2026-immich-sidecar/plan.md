# Immich integration via a sidecar ML server

**Date:** 2026-09-18
**Supersedes:** the Phase 5 "hijack" plan and `REPLICATION.md` on the unmerged
`replication-beta` branch, plus the forked `immich-clone/machine-learning` tree.

## Decision

Do not fork Immich. Serve Immich's ML HTTP contract from our own tiny service and
point Immich at it. Immich's server talks to machine learning over HTTP, so the
integration seam is one JSON shape, not a Python package.

The previous attempt put `DogDetector`/`DogEmbedder` *inside* `immich_ml` and died
on the first `git pull` (reflog: rebase started, aborted 42s later, Dec 14 2025,
never touched again). Nothing about that code needed to live inside Immich.

## The contract (verified against v3.2.2)

`server/src/repositories/machine-learning.repository.ts` `detectFaces()` POSTs
multipart to `<url>/predict`:

- `entries` (form field, JSON):
  ```json
  {"facial-recognition": {
     "detection":   {"modelName": "buffalo_l", "options": {"minScore": 0.7}},
     "recognition": {"modelName": "buffalo_l"}}}
  ```
- `image` (file): raw bytes of the asset preview.

Expected response:
```json
{"facial-recognition": [
   {"boundingBox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
    "embedding": "[0.1, 0.2, ...]",
    "score": 0.99}],
 "imageHeight": 1080, "imageWidth": 1920}
```

`embedding` is a **JSON string**, not an array — `serialize_np_array()` in
`immich_ml/models/transforms.py` is `orjson.dumps(arr).decode()`. Immich stores it
as text and casts to a pgvector.

Also required: `GET /ping` returning 200. The server polls it for health.

Unchanged from the Dec 2025 clone apart from added OCR/Paddle enum members, so
this contract is stable across two major versions.

### What this buys us

The `audit-immich-interface.md` blockers were all *internal* to `immich_ml`. At the
HTTP boundary they disappear:

- **Landmarks (M1, M2, M7)** — gone. Landmarks are consumed by Immich's own
  `ArcFaceONNX._crop()`. We replace the whole task, so nothing asks for them. No
  dummy 5-point synthesis, no keypoint model needed.
- **112x112 BGR / ArcFace normalisation (M3, M5, M6)** — gone. We do our own
  cropping and preprocessing; only the embedding vector crosses the wire.
- **Model registry (M4)** — gone. `modelName` is a string we ignore.
- **JSON serialisation (M9)** — one `orjson.dumps().decode()`.

Person thumbnails are cropped server-side from `boundingBox`, so they work too.

## Design

One container, one Python file. It answers `facial-recognition` itself and
forwards every other task (CLIP search, OCR) to the stock
`immich-machine-learning` container untouched.

The proxy is needed because Immich's `urls` list is failover, not routing — every
task goes to whichever URL is healthy, so our service must be able to answer all
of them.

```
immich-server ──▶ animal-ml (ours) ──facial-recognition──▶ our ONNX
                        └─────────everything else────────▶ immich-machine-learning
```

## Implementation

### `serve.py` (~120 lines, the whole thing)

```python
app = FastAPI()

@app.get("/ping")
def ping(): return PlainTextResponse("pong")

@app.post("/predict")
async def predict(entries: str = Form(), image: bytes = File(None), text: str = Form(None)):
    parsed = orjson.loads(entries)
    if "facial-recognition" not in parsed:
        return await proxy_upstream(entries, image, text)   # httpx, forward verbatim
    min_score = parsed["facial-recognition"]["detection"]["options"]["minScore"]
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    faces = []
    for det in detector.predict(img):
        if det["confidence"] < min_score:
            continue
        x1, y1, x2, y2 = pad_bbox(det["bbox"], w, h, 0.1)
        vec = embedder.predict(img[y1:y2, x1:x2])
        faces.append({
            "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "embedding": orjson.dumps(vec, option=orjson.OPT_SERIALIZE_NUMPY).decode(),
            "score": det["confidence"],
        })
    return ORJSONResponse({"facial-recognition": faces, "imageHeight": h, "imageWidth": w})
```

`detector` and `embedder` are `ONNXDetector` / `ONNXEmbedding` from
`animal_id/pipeline/onnx_models.py` — already written, already the right
preprocessing (RGB, `/255.0`, NCHW), already L2-normalised in the graph. The
bbox padding is the same 10% as `animal_pipeline.py:136`.

New code is glue only: the endpoint, the proxy, the crop loop.

### `Dockerfile`

`FROM python:3.12-slim`, install `fastapi uvicorn onnxruntime opencv-python-headless
httpx orjson python-multipart`, copy `animal_id/pipeline/` + `serve.py` +
`models/onnx/*.onnx`. No Immich base image, no Immich source.

### Deploy

Add one service to your existing Immich compose and set the ML URL in
**Administration → Settings → Machine Learning → URL** to `http://animal-ml:3003`.
Set `UPSTREAM_ML_URL=http://immich-machine-learning:3003` on our container.

Set **Max Distance** to `clustering.eps` from `models/onnx/embedding.json` (the
sidecar `train_final.py` now writes) — that value is exactly Immich's
`maxDistance`, swept on our embedding geometry.

## Scope

**Not doing:** no Immich fork, no branch to rebase, no server/web TypeScript
changes, no `ModelTask.DOG_IDENTIFICATION`, no keypoint stage, no `depends`
dependency graph, no model registry entries, no `Dockerfile.dogs` layered on
Immich's build, no upstream PR.

Immich thinks it is finding human faces. It gets dogs. That is the whole trick,
and at the HTTP boundary it costs no forked code.

**Accepted consequences:** real human faces stop being detected while this is
pointed at an instance, and the People tab mixes dogs into whatever is already
there. Use a test instance or a fresh library, as the old REPLICATION.md warned.

**Upgrade exposure:** one JSON shape. If Immich changes it, the fix is editing a
dict literal, not resolving a rebase.

## Steps

1. Write `serve.py` and its `Dockerfile`; run it locally, POST a dog photo with a
   hand-built `entries` field, assert the response shape.
2. Add the compose service; point Immich's ML URL at it.
3. Run Administration → Jobs → Face Detection → All. Check the People tab.
4. Tune Max Distance from the model sidecar if clusters over- or under-merge.

If step 3 works, this is done. There is no phase 2.

## Outcome (2026-09-19)

Done, and it works. Built as `sidecar/` — `serve.py` (~130 lines), `Dockerfile`,
`docker-compose.yml`, `smoke_test.py`, `README.md`.

Verified against a freshly pulled Immich **v3.2.2** stack (server, ML, postgres,
valkey) with 31 Wikimedia photos of five named dogs:

- 28 faces across 25 photos; 12 people clustered; thumbnails correctly cropped
  from our `boundingBox`.
- Clusters track identity, including splitting the two dogs in a single photo
  into separate people.
- CLIP smart search and OCR still work: both model types loaded in the stock
  container via the proxy. `'a black dog'` returns the Portuguese Water Dogs,
  `'a german shepherd'` returns the shepherds.

Three things the plan did not anticipate:

1. **The embedder needs ImageNet normalisation.** `models/onnx/embedding.json`
   says so; `ONNXEmbedding` only did `/255.0`. The plan's "already the right
   preprocessing" was wrong for the current convnext_tiny export. Fixed in
   `animal_id/pipeline/onnx_models.py` (it now reads mean/std from the sidecar
   JSON) — this was also silently affecting `18_run_identification.py`,
   `19_explore_fiftyone.py` and `train_master.py`. Measured on 528 pairs:
   different-identity cosine similarity drops 0.2034 → 0.1412 while
   same-identity barely moves, widening the margin ~16% relative.
2. **`minScore` 0.7 is tuned for human faces.** At Immich's default we lost a
   third of the dogs. 0.3 is the right setting; it is an admin setting, so no
   code change.
3. **`minFaces` moved to a per-user preference** in v3
   (`preferences.people.minimumFaces`, default 3). The admin ML setting still
   exists and still gates clustering, but the People page hides anything below
   the user's own threshold — which is why 11 clusters in the database showed as
   2 on screen.

The contract itself needed no adjustment: what the plan documented is what
v3.2.2 sends and accepts.
