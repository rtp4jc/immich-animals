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


# Only the openvino image ships this provider; there the models run on the Intel GPU
# (OPENVINO_DEVICE=CPU lets CI smoke-test the image on a runner without one).
PROVIDERS = (
    [
        (
            "OpenVINOExecutionProvider",
            {"device_type": os.environ.get("OPENVINO_DEVICE", "GPU")},
        )
    ]
    if "OpenVINOExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)


def _session(name: str) -> tuple[ort.InferenceSession, str, tuple[int, int]]:
    session = ort.InferenceSession(str(MODEL_DIR / name), providers=PROVIDERS)
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


def _blob(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    resized = cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]


def _letterbox(
    image: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, float, int, int]:
    """The blob plus the scale and left/top padding that map boxes back to source pixels.

    YOLO trains on aspect-preserved images padded to square; stretched phone
    photos cost the detector ~7pp recall on owner photos.
    """
    height, width = image.shape[:2]
    scale = min(size[0] / height, size[1] / width)
    h, w = round(height * scale), round(width * scale)
    top, left = (size[0] - h) // 2, (size[1] - w) // 2
    canvas = np.full((*size, 3), 114, np.uint8)  # Ultralytics' pad colour
    canvas[top : top + h, left : left + w] = cv2.resize(
        image, (w, h), interpolation=cv2.INTER_LINEAR
    )
    return (
        np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None],
        scale,
        left,
        top,
    )


def _detect(rgb: np.ndarray) -> list[tuple[float, list[int], _Species]]:
    """Known-species detections above that species' min score, bbox in source pixels."""
    height, width = rgb.shape[:2]
    blob, scale, left, top = _letterbox(rgb, _det_size)
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


def _embed(crop: np.ndarray, kind: _Species) -> np.ndarray:
    embedder = embedders[kind.embedder_stem]
    blob = _blob(crop, embedder.size, cv2.INTER_AREA)
    blob = (blob - embedder.mean) / embedder.std
    vector = embedder.session.run(None, {embedder.input_name: blob})[0][0]
    return _rescale(vector, kind.shift) if 0 < kind.shift < 1 else vector


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
    "animal-ml: %s, people=%s, upstream=%s, %s",
    detector.get_providers()[0],
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

    buffer = np.frombuffer(await form["image"].read(), np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "Could not decode image")  # same status as upstream
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    faces = []
    for score, bbox, kind in _detect(rgb):
        x1, y1, x2, y2 = _pad(bbox, width, height)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        faces.append(
            {
                "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                # Immich expects the vector as a JSON string, not an array.
                "embedding": orjson.dumps(
                    _embed(crop, kind), option=orjson.OPT_SERIALIZE_NUMPY
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
