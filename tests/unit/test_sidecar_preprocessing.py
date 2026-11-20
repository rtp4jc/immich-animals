"""The sidecar must feed its models the tensors animal_id's pipeline does."""

import importlib.util
from pathlib import Path

import numpy as np

from animal_id.pipeline.onnx_models import ONNXDetector, ONNXEmbedding

_spec = importlib.util.spec_from_file_location(
    "sidecar_preprocess",
    Path(__file__).parents[2] / "sidecar" / "preprocess.py",
)
sidecar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sidecar)

# Odd, non-square sizes so a swapped width/height or rounding shows up.
IMAGE = np.random.default_rng(0).integers(0, 256, (481, 333, 3), np.uint8)


def _wrapper(cls, size):
    """The pipeline wrapper without an ONNX session, for its _preprocess only."""
    wrapper = cls.__new__(cls)
    wrapper.input_size = size
    return wrapper


def test_detector_letterbox_matches_the_pipeline():
    expected, (scale, left, top) = _wrapper(ONNXDetector, (640, 640))._preprocess(IMAGE)
    blob, *geometry = sidecar.letterbox(IMAGE, (640, 640))

    np.testing.assert_array_equal(blob, expected)
    assert geometry == [scale, left, top]


def test_embedder_input_matches_the_pipeline():
    expected, _ = _wrapper(ONNXEmbedding, (224, 224))._preprocess(IMAGE)
    blob = sidecar.embedder_blob(
        IMAGE, (224, 224), ONNXEmbedding.mean, ONNXEmbedding.std
    )

    np.testing.assert_array_equal(blob, expected)
