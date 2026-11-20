"""Behaviour of serve.py against tiny constant-output ONNX models.

Usage: uv run --project sidecar pytest sidecar
"""

import asyncio
import importlib.util
import json
import threading
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnx
import orjson
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

ENTRIES = {"facial-recognition": {"detection": {"options": {"minScore": 0.7}}}}


def _constant_model(path: Path, input_shape: list[int], output: np.ndarray, **meta):
    """A graph that ignores its input and always returns `output`."""
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["out"], value=numpy_helper.from_array(output)
            )
        ],
        "fake",
        [helper.make_tensor_value_info("in", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(output.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    helper.set_model_props(model, meta)
    onnx.save(model, path)


def _embedder(model_dir: Path, stem: str, vector: np.ndarray):
    _constant_model(model_dir / f"{stem}.onnx", [1, 3, 8, 8], vector[None])
    prep = {"mean": [0.5] * 3, "std": [0.5] * 3}
    (model_dir / f"{stem}.json").write_text(json.dumps({"preprocessing": prep}))


def _load_serve(model_dir: Path, monkeypatch, **env):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "serve", Path(__file__).with_name("serve.py")
    )
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    return serve


def _image() -> bytes:
    return cv2.imencode(".png", np.zeros((640, 640, 3), np.uint8))[1].tobytes()


def test_each_species_gets_its_own_threshold_embedder_and_mapping(
    tmp_path, monkeypatch
):
    rows = [
        [10, 10, 100, 100, 0.50, 0],  # dog
        [200, 200, 300, 300, 0.25, 1],  # cat, kept by CAT_MIN_SCORE=0.2
        [400, 400, 500, 500, 0.10, 1],  # cat below its threshold
        [10, 400, 100, 500, 0.90, 2],  # unknown species
    ]
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.array([rows], np.float32),
        names="{0: 'dog', 1: 'cat', 2: 'bird'}",
    )
    rng = np.random.default_rng(0)
    shared, cat = rng.normal(size=(2, 512)).astype(np.float32)
    _embedder(tmp_path, "embedding", shared)
    _embedder(tmp_path, "embedding_cat", cat)
    serve = _load_serve(
        tmp_path,
        monkeypatch,
        KEEP_HUMAN_FACES="false",
        CAT_MIN_SCORE="0.2",
        DOG_MAX_DISTANCE="0.5",  # equals Immich's, so dog vectors pass through
        CAT_MAX_DISTANCE="0.3",
    )

    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )
    faces = {
        round(f["score"], 2): np.array(json.loads(f["embedding"]))
        for f in response.json()["facial-recognition"]
    }

    assert set(faces) == {0.5, 0.25}
    assert np.allclose(faces[0.5], shared)
    cosine = faces[0.25] @ cat / np.linalg.norm(cat)
    assert abs(cosine - np.sqrt(0.5 / 0.7)) < 0.1  # cat's own shift was applied


def _slow_detector_serve(tmp_path, monkeypatch, **env):
    """serve with a detector that blocks until released; returns (serve, started, release, done)."""
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.zeros((1, 1, 6), np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    serve = _load_serve(tmp_path, monkeypatch, **env)
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def detect(rgb):
        started.set()
        # Bounded, so a blocked event loop fails the test instead of hanging it.
        release.wait(timeout=5)
        done.set()
        return []

    monkeypatch.setattr(serve, "_detect", detect)
    return serve, started, release, done


async def _predict(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )


def test_server_answers_while_animals_are_inferred(tmp_path, monkeypatch):
    serve, started, release, done = _slow_detector_serve(
        tmp_path, monkeypatch, KEEP_HUMAN_FACES="false"
    )

    async def scenario():
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            predict = asyncio.create_task(_predict(client))
            await asyncio.to_thread(started.wait, 5)
            ping = await client.get("/ping")
            answered_during_inference = not done.is_set()
            release.set()
            return ping, answered_during_inference, await predict

    ping, answered_during_inference, predict = asyncio.run(scenario())
    assert ping.text == "pong"
    assert answered_during_inference
    assert predict.status_code == 200


def test_human_faces_are_requested_while_animals_are_inferred(tmp_path, monkeypatch):
    serve, started, release, done = _slow_detector_serve(
        tmp_path, monkeypatch, UPSTREAM_ML_URL="http://upstream"
    )
    human = {"boundingBox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "score": 0.9}
    upstream_during_inference = []

    async def post_upstream(body, content_type):
        upstream_during_inference.append(not done.is_set())
        release.set()
        payload = orjson.dumps({"facial-recognition": [human]})
        return httpx.Response(200, content=payload, request=httpx.Request("POST", "/"))

    monkeypatch.setattr(serve, "_post_upstream", post_upstream)

    async def scenario():
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            return await _predict(client)

    response = asyncio.run(scenario())
    assert upstream_during_inference == [True]
    assert response.json()["facial-recognition"] == [human]
