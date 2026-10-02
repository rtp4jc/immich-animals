"""Behaviour of serve.py against tiny constant-output ONNX models.

Usage: uv run --project sidecar pytest sidecar
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
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
    for key, value in {
        "MODEL_DIR": str(tmp_path),
        "KEEP_HUMAN_FACES": "false",
        "CAT_MIN_SCORE": "0.2",
        "DOG_MAX_DISTANCE": "0.5",  # equals Immich's, so dog vectors pass through
        "CAT_MAX_DISTANCE": "0.3",
    }.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "serve", Path(__file__).with_name("serve.py")
    )
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)

    image = cv2.imencode(".png", np.zeros((640, 640, 3), np.uint8))[1].tobytes()
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": image}
    )
    faces = {
        round(f["score"], 2): np.array(json.loads(f["embedding"]))
        for f in response.json()["facial-recognition"]
    }

    assert set(faces) == {0.5, 0.25}
    assert np.allclose(faces[0.5], shared)
    cosine = faces[0.25] @ cat / np.linalg.norm(cat)
    assert abs(cosine - np.sqrt(0.5 / 0.7)) < 0.1  # cat's own shift was applied
