"""Answer Immich's machine-learning HTTP contract with dog models.

Immich has one face-detection pipeline, so dogs ride along in it: we answer the
facial-recognition task with dogs and forward it upstream for people as well.
Every other task (CLIP, OCR) is passed through untouched, because Immich's
`urls` list is failover rather than routing — whichever server answers has to
answer everything.

Immich's own Min Detection Score and Max Distance stay at whatever the user has
them set to. Dogs want different values, so the sidecar applies its own
threshold and maps its embeddings onto Immich's, rather than asking the user to
retune settings that are already right for people.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import httpx
import numpy as np
import onnxruntime as ort
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse, Response

TASK = "facial-recognition"
BBOX_PAD = 0.1  # matches AnimalPipeline's crop, which the embedder was tuned on
# Immich's smaller face models; its picker doubles as ours.
SMALL_FACE_MODELS = {"buffalo_s", "buffalo_m"}

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/onnx"))
UPSTREAM_URL = os.environ.get("UPSTREAM_ML_URL", "").rstrip("/")
KEEP_HUMAN_FACES = os.environ.get("KEEP_HUMAN_FACES", "true").lower() in {
    "1",
    "true",
    "yes",
}
# Immich's default suits people; dogs need roughly 0.3 or half of them are lost.
DOG_MIN_SCORE = float(os.environ.get("DOG_MIN_SCORE", "0.3"))
# Where our embeddings cluster best, and the Max Distance Immich is set to.
DOG_MAX_DISTANCE = float(os.environ.get("DOG_MAX_DISTANCE", "0.35"))
IMMICH_MAX_DISTANCE = float(os.environ.get("IMMICH_MAX_DISTANCE", "0.5"))


def _session(name: str) -> tuple[ort.InferenceSession, str, tuple[int, int]]:
    session = ort.InferenceSession(
        str(MODEL_DIR / name), providers=["CPUExecutionProvider"]
    )
    spec = session.get_inputs()[0]
    return session, spec.name, tuple(spec.shape[2:])


def _channel_constant(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(3, 1, 1)


class _Embedder(NamedTuple):
    session: ort.InferenceSession
    input_name: str
    size: tuple[int, int]
    mean: np.ndarray
    std: np.ndarray


def _load_embedder(stem: str) -> _Embedder:
    """Load an embedder plus the preprocessing its exporter recorded.

    ImageNet normalisation is not baked into the graph and skipping it costs
    accuracy silently, so it is read rather than assumed.
    """
    session, input_name, size = _session(f"{stem}.onnx")
    prep = json.loads((MODEL_DIR / f"{stem}.json").read_text())["preprocessing"]
    return _Embedder(
        session,
        input_name,
        size,
        _channel_constant(prep["mean"]),
        _channel_constant(prep["std"]),
    )


detector, _det_input, _det_size = _session("detector.onnx")
_large = _load_embedder("embedding")
_small = _load_embedder("embedding_resnet50")


def _embedder_for(face_model: str) -> _Embedder:
    """Follow Immich's face-model choice: smaller model, faster embedder."""
    return _small if face_model in SMALL_FACE_MODELS else _large


def _blob(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    resized = cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]


def _detect(rgb: np.ndarray) -> list[tuple[float, list[int]]]:
    """Detections above DOG_MIN_SCORE, as (score, bbox) in source pixels."""
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
        if score >= DOG_MIN_SCORE
    ]


# Mixing a unit vector with an independent random one maps cosine distance
# affinely: d' = (1 - a) + a*d, since random high-dimensional vectors are nearly
# orthogonal. Solving d' = IMMICH_MAX_DISTANCE at d = DOG_MAX_DISTANCE gives a.
_SHIFT = (1 - IMMICH_MAX_DISTANCE) / (1 - DOG_MAX_DISTANCE)
_RESCALE = 0 < _SHIFT < 1


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _rescale(vector: np.ndarray) -> np.ndarray:
    """Move our distances onto Immich's threshold, deterministically per face."""
    seed = hashlib.blake2b(vector.tobytes(), digest_size=8).digest()
    noise = np.random.default_rng(int.from_bytes(seed, "big")).normal(size=vector.shape)
    mixed = np.sqrt(_SHIFT) * _unit(vector) + np.sqrt(1 - _SHIFT) * _unit(noise)
    return _unit(mixed).astype(np.float32)


def _embed(crop: np.ndarray, embedder: _Embedder) -> np.ndarray:
    blob = _blob(crop, embedder.size, cv2.INTER_AREA)
    blob = (blob - embedder.mean) / embedder.std
    vector = embedder.session.run(None, {embedder.input_name: blob})[0][0]
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
    "animal-ml: people=%s, dog min score %.2f, Max Distance %.2f acts as %.2f, upstream=%s",
    KEEP_HUMAN_FACES,
    DOG_MIN_SCORE,
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

    embedder = _embedder_for(entries[TASK].get("recognition", {}).get("modelName", ""))
    buffer = np.frombuffer(await form["image"].read(), np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "Could not decode image")  # same status as upstream
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    faces = []
    for score, bbox in _detect(rgb):
        x1, y1, x2, y2 = _pad(bbox, width, height)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        faces.append(
            {
                "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                # Immich expects the vector as a JSON string, not an array.
                "embedding": orjson.dumps(
                    _embed(crop, embedder), option=orjson.OPT_SERIALIZE_NUMPY
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
