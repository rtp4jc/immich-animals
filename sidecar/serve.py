"""Answer Immich's machine-learning HTTP contract with animal models.

Immich has one face-detection pipeline, so animals ride along in it: we answer
the facial-recognition task with animals and forward it upstream for people as well.
Every other task (CLIP, OCR) is passed through untouched, because Immich's
`urls` list is failover rather than routing — whichever server answers has to
answer everything.

Immich's own Min Detection Score and Max Distance stay at whatever the user has
them set to. Each species wants different values, so the sidecar applies its own
thresholds and maps its embeddings onto Immich's, rather than asking the user to
retune settings that are already right for people.
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

import cv2
import httpx
import numpy as np
import onnxruntime as ort
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from preprocess import embedder_blob, letterbox

TASK = "facial-recognition"
BBOX_PAD = 0.1  # matches AnimalPipeline's crop, which the embedder was tuned on

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/onnx"))
UPSTREAM_URL = os.environ.get("UPSTREAM_ML_URL", "").rstrip("/")
KEEP_HUMAN_FACES = os.environ.get("KEEP_HUMAN_FACES", "true").lower() in {
    "1",
    "true",
    "yes",
}
# (<SPECIES>_MIN_SCORE, <SPECIES>_MAX_DISTANCE) defaults. Immich's min score suits
# people; dogs need roughly 0.3 or half of them are lost. Max distance is where
# that species' embeddings cluster best. Detector classes not listed are ignored.
SPECIES_DEFAULTS = {"dog": (0.3, 0.375), "cat": (0.3, 0.35)}
IMMICH_MAX_DISTANCE = float(os.environ.get("IMMICH_MAX_DISTANCE", "0.5"))
UPSTREAM_ATTEMPTS = 3


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


class _Species(NamedTuple):
    name: str
    min_score: float
    max_distance: float
    embedder_stem: str
    # Mixing a unit vector with an independent random one maps cosine distance
    # affinely: d' = (1 - a) + a*d, since random high-dimensional vectors are nearly
    # orthogonal. Solving d' = IMMICH_MAX_DISTANCE at d = max_distance gives a.
    shift: float


def _species(name: str) -> _Species:
    min_score, max_distance = SPECIES_DEFAULTS[name]
    prefix = name.upper()
    max_distance = float(os.environ.get(f"{prefix}_MAX_DISTANCE", max_distance))
    stem = f"embedding_{name}"
    return _Species(
        name,
        float(os.environ.get(f"{prefix}_MIN_SCORE", min_score)),
        max_distance,
        stem if (MODEL_DIR / f"{stem}.onnx").exists() else "embedding",
        (1 - IMMICH_MAX_DISTANCE) / (1 - max_distance),
    )


detector, _det_input, _det_size = _session("detector.onnx")
# Ultralytics writes class names as a Python dict literal, e.g. "{0: 'dog'}".
_class_names = ast.literal_eval(detector.get_modelmeta().custom_metadata_map["names"])
species = {
    class_id: _species(name)
    for class_id, name in _class_names.items()
    if name in SPECIES_DEFAULTS
}
embedders = {
    stem: _load_embedder(stem) for stem in {s.embedder_stem for s in species.values()}
}


def _detect(rgb: np.ndarray) -> list[tuple[float, list[int], _Species]]:
    """Known-species detections above that species' min score, bbox in source pixels."""
    height, width = rgb.shape[:2]
    blob, scale, left, top = letterbox(rgb, _det_size)
    raw = detector.run(None, {_det_input: blob})
    return [
        (
            float(score),
            [
                int(np.clip((x1 - left) / scale, 0, width)),
                int(np.clip((y1 - top) / scale, 0, height)),
                int(np.clip((x2 - left) / scale, 0, width)),
                int(np.clip((y2 - top) / scale, 0, height)),
            ],
            species[int(class_id)],
        )
        for x1, y1, x2, y2, score, class_id in raw[0][0]
        if int(class_id) in species and score >= species[int(class_id)].min_score
    ]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _rescale(vector: np.ndarray, shift: float) -> np.ndarray:
    """Move our distances onto Immich's threshold, deterministically per face."""
    seed = hashlib.blake2b(vector.tobytes(), digest_size=8).digest()
    noise = np.random.default_rng(int.from_bytes(seed, "big")).normal(size=vector.shape)
    mixed = np.sqrt(shift) * _unit(vector) + np.sqrt(1 - shift) * _unit(noise)
    return _unit(mixed).astype(np.float32)


def _embed(crops: list[tuple[np.ndarray, _Species]]) -> list[np.ndarray]:
    """One session run per embedder, however many animals the photo has."""
    vectors: list[np.ndarray] = [np.empty(0)] * len(crops)
    by_stem = defaultdict(list)
    for i, (_, kind) in enumerate(crops):
        by_stem[kind.embedder_stem].append(i)
    for stem, indices in by_stem.items():
        embedder = embedders[stem]
        batch = np.concatenate(
            [
                embedder_blob(crops[i][0], embedder.size, embedder.mean, embedder.std)
                for i in indices
            ]
        )
        for i, vector in zip(
            indices, embedder.session.run(None, {embedder.input_name: batch})[0]
        ):
            shift = crops[i][1].shift
            vectors[i] = _rescale(vector, shift) if 0 < shift < 1 else vector
    return vectors


def _pad(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad = int((x2 - x1) * BBOX_PAD)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


_client: httpx.AsyncClient


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _client
    async with httpx.AsyncClient(timeout=120) as _client:
        yield


app = FastAPI(lifespan=_lifespan)
logging.getLogger("uvicorn.error").info(
    "animal-ml: people=%s, upstream=%s, %s",
    KEEP_HUMAN_FACES,
    UPSTREAM_URL or "unset",
    "; ".join(
        f"{s.name}: min score {s.min_score:.2f}, Max Distance {IMMICH_MAX_DISTANCE:.2f}"
        f" acts as {s.max_distance if 0 < s.shift < 1 else IMMICH_MAX_DISTANCE:.2f},"
        f" {s.embedder_stem}.onnx"
        for s in species.values()
    ),
)


@app.get("/ping")
async def ping() -> PlainTextResponse:
    # Immich reads a pong as "ML works"; without upstream, CLIP search, OCR and
    # human faces would all fail, so report that rather than hide it. Immich
    # gives the whole check 2 s by default.
    if UPSTREAM_URL:
        try:
            response = await _upstream("GET", "/ping", timeout=1.5)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(503, f"Upstream ML unavailable: {error!r}") from error
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

    rgb = await run_in_threadpool(_decode, await form["image"].read())
    # ONNX runs in a worker thread so the event loop keeps serving pings, CLIP and
    # the upstream face request while animals are inferred.
    animals, humans = await asyncio.gather(
        run_in_threadpool(_animal_faces, rgb), _human_faces(body, content_type)
    )
    height, width = rgb.shape[:2]
    return ORJSONResponse(
        {TASK: animals + humans, "imageHeight": height, "imageWidth": width}
    )


def _decode(data: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "Could not decode image")  # same status as upstream
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _animal_faces(rgb: np.ndarray) -> list[dict]:
    height, width = rgb.shape[:2]
    boxes, crops = [], []
    for score, bbox, kind in _detect(rgb):
        x1, y1, x2, y2 = _pad(bbox, width, height)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        boxes.append(({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, score))
        crops.append((crop, kind))
    return [
        {
            "boundingBox": box,
            # Immich expects the vector as a JSON string, not an array.
            "embedding": orjson.dumps(
                vector, option=orjson.OPT_SERIALIZE_NUMPY
            ).decode(),
            "score": score,
        }
        for (box, score), vector in zip(boxes, _embed(crops))
    ]


async def _human_faces(body: bytes, content_type: str) -> list[dict]:
    if not (KEEP_HUMAN_FACES and UPSTREAM_URL):
        return []
    upstream = await _post_upstream(body, content_type)
    upstream.raise_for_status()
    return orjson.loads(upstream.content).get(TASK, [])


async def _post_upstream(body: bytes, content_type: str) -> httpx.Response:
    if not UPSTREAM_URL:
        raise HTTPException(503, "UPSTREAM_ML_URL is not set")
    return await _upstream(
        "POST", "/predict", content=body, headers={"content-type": content_type}
    )


async def _upstream(method: str, path: str, **kwargs) -> httpx.Response:
    # httpx drops pooled connections that upstream closed cleanly, but one that
    # died silently (upstream killed or restarted) fails the request before any
    # answer and leaves the pool; several can die at once. Predictions are
    # idempotent, so try again, each time on the next connection or a new one.
    url = f"{UPSTREAM_URL}{path}"
    for _ in range(UPSTREAM_ATTEMPTS - 1):
        try:
            return await _client.request(method, url, **kwargs)
        except (httpx.NetworkError, httpx.RemoteProtocolError):
            pass
    return await _client.request(method, url, **kwargs)
