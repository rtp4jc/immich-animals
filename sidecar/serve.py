"""Serve Immich's machine-learning HTTP contract with our dog models.

Immich asks for human faces and gets dogs. Every other task (CLIP, OCR) is
forwarded verbatim to the stock immich-machine-learning container, because
Immich's `urls` list is failover, not routing: whichever server answers has to
answer everything.

KEEP_HUMAN_FACES (on by default) returns upstream's human faces alongside the
dogs, and rescales our embeddings so Immich's human-tuned Max Distance also
works for dogs. Set it false to serve dogs only, and set Max Distance yourself.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnxruntime as ort
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse, Response

TASK = "facial-recognition"
BBOX_PAD = 0.1  # matches AnimalPipeline's crop, which the embedder was tuned on

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/onnx"))
UPSTREAM_URL = os.environ.get("UPSTREAM_ML_URL", "").rstrip("/")
KEEP_HUMAN_FACES = os.environ.get("KEEP_HUMAN_FACES", "true").lower() in {
    "1",
    "true",
    "yes",
}
# Our embeddings cluster best at DOG_MAX_DISTANCE, Immich's humans at its own
# default. One setting cannot serve both, so we move ours onto theirs.
DOG_MAX_DISTANCE = float(os.environ.get("DOG_MAX_DISTANCE", "0.35"))
IMMICH_MAX_DISTANCE = float(os.environ.get("IMMICH_MAX_DISTANCE", "0.5"))


def _load(name: str) -> tuple[ort.InferenceSession, str, tuple[int, int]]:
    session = ort.InferenceSession(
        str(MODEL_DIR / name), providers=["CPUExecutionProvider"]
    )
    spec = session.get_inputs()[0]
    return session, spec.name, tuple(spec.shape[2:])


detector, _det_input, _det_size = _load("detector.onnx")
embedder, _emb_input, _emb_size = _load("embedding.onnx")

# The embedder expects ImageNet normalisation; skipping it costs accuracy
# silently, so read it from the sidecar JSON the exporter writes.
_prep = json.loads((MODEL_DIR / "embedding.json").read_text())["preprocessing"]
_MEAN = np.array(_prep["mean"], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(_prep["std"], dtype=np.float32).reshape(3, 1, 1)


def _blob(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    resized = cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]


def _detect(rgb: np.ndarray, min_score: float) -> list[tuple[float, list[int]]]:
    """Detections above min_score, as (score, bbox) in source-image pixels."""
    height, width = rgb.shape[:2]
    scale_x, scale_y = width / _det_size[1], height / _det_size[0]
    raw = detector.run(None, {_det_input: _blob(rgb, _det_size, cv2.INTER_LINEAR)})
    return [
        (
            float(score),
            [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ],
        )
        for x1, y1, x2, y2, score, _ in raw[0][0]
        if score >= min_score
    ]


# Mixing a unit vector with an independent random one maps cosine distance
# affinely: d' = (1 - a) + a*d, since random high-dimensional vectors are nearly
# orthogonal. Solving d' = IMMICH_MAX_DISTANCE at d = DOG_MAX_DISTANCE gives a.
_SHIFT = (1 - IMMICH_MAX_DISTANCE) / (1 - DOG_MAX_DISTANCE)
_RESCALE = KEEP_HUMAN_FACES and 0 < _SHIFT < 1


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _rescale(vector: np.ndarray) -> np.ndarray:
    """Move our distances onto Immich's threshold, deterministically per face."""
    seed = hashlib.blake2b(vector.tobytes(), digest_size=8).digest()
    noise = np.random.default_rng(int.from_bytes(seed, "big")).normal(size=vector.shape)
    mixed = np.sqrt(_SHIFT) * _unit(vector) + np.sqrt(1 - _SHIFT) * _unit(noise)
    return _unit(mixed).astype(np.float32)


def _embed(crop: np.ndarray) -> np.ndarray:
    blob = (_blob(crop, _emb_size, cv2.INTER_AREA) - _MEAN) / _STD
    vector = embedder.run(None, {_emb_input: blob})[0][0]
    return _rescale(vector) if _RESCALE else vector


def _pad(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad = int((x2 - x1) * BBOX_PAD)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


app = FastAPI()
logging.getLogger("uvicorn.error").info(
    "animal-ml: humans=%s rescale=%s (Max Distance %.2f behaves as %.2f) upstream=%s",
    KEEP_HUMAN_FACES,
    _RESCALE,
    IMMICH_MAX_DISTANCE,
    DOG_MAX_DISTANCE if _RESCALE else IMMICH_MAX_DISTANCE,
    UPSTREAM_URL or "unset",
)


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


@app.post("/predict")
async def predict(request: Request) -> Response:
    body = await request.body()  # cached, so form() below can still parse it
    form = await request.form()
    content_type = request.headers["content-type"]
    entries = orjson.loads(form["entries"])
    if TASK not in entries:
        upstream = await _post_upstream(body, content_type)
        return Response(
            upstream.content,
            upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    min_score = entries[TASK]["detection"]["options"]["minScore"]
    buffer = np.frombuffer(await form["image"].read(), np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "Could not decode image")  # same status as upstream
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    faces = []
    for score, bbox in _detect(rgb, min_score):
        x1, y1, x2, y2 = _pad(bbox, width, height)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        faces.append(
            {
                "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                # Immich expects the vector as a JSON string, not an array.
                "embedding": orjson.dumps(
                    _embed(crop), option=orjson.OPT_SERIALIZE_NUMPY
                ).decode(),
                "score": score,
            }
        )

    if KEEP_HUMAN_FACES and UPSTREAM_URL:
        upstream = await _post_upstream(body, content_type)
        upstream.raise_for_status()
        faces += orjson.loads(upstream.content).get(TASK, [])

    return ORJSONResponse({TASK: faces, "imageHeight": height, "imageWidth": width})


async def _post_upstream(body: bytes, content_type: str) -> httpx.Response:
    if not UPSTREAM_URL:
        raise HTTPException(503, "UPSTREAM_ML_URL is not set")
    async with httpx.AsyncClient(timeout=120) as client:
        return await client.post(
            f"{UPSTREAM_URL}/predict",
            content=body,
            headers={"content-type": content_type},
        )
