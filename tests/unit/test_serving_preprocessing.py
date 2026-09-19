"""The embedder must be served the tensor it was trained on."""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from animal_id.pipeline.onnx_models import ONNXEmbedding, _ONNXModel

# IdentityDataset's eval transform.
TRAINING_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class _FakeSession:
    """Stands in for onnxruntime so the check needs no .onnx file."""

    def get_inputs(self):
        class _Input:
            name = "input"
            shape = [1, 3, 224, 224]

        return [_Input()]


def _wrapper():
    wrapper = ONNXEmbedding.__new__(ONNXEmbedding)
    wrapper.session = _FakeSession()
    wrapper.input_size = (224, 224)
    return wrapper


def test_serving_preprocessing_matches_training(tmp_path):
    """Regression: serving raw [0,1] cost ~1.3pp MRR and raised nothing."""
    array = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    image_path = tmp_path / "dog.png"
    Image.fromarray(array).save(image_path)

    expected = TRAINING_TRANSFORM(Image.open(image_path).convert("RGB")).numpy()
    served, _ = _wrapper()._preprocess(
        cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    )

    assert served.shape == (1, 3, 224, 224)
    # PIL and cv2 resize differently, so compare the channel statistics that
    # normalisation governs rather than pixel-exact values.
    assert np.allclose(
        served[0].mean(axis=(1, 2)), expected.mean(axis=(1, 2)), atol=0.15
    )
    assert np.allclose(served[0].std(axis=(1, 2)), expected.std(axis=(1, 2)), atol=0.15)
    assert served.min() < -0.5, "embedder input is not ImageNet-normalised"


def test_yolo_stages_stay_unnormalised():
    """Detector and keypoint are YOLO: they expect [0, 1], not ImageNet stats."""
    assert not hasattr(_ONNXModel, "mean")
