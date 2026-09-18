"""ONNX runtime wrappers for the three pipeline stages."""

from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .models import AnimalClass, DetectionModel, EmbeddingModel, KeypointModel

DETECTION_CONF_THRESHOLD = 0.1


class _ONNXModel:
    """Shared session plumbing: single input, single output, NCHW float32 in [0, 1]."""

    interpolation = cv2.INTER_LINEAR

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize to the model's input size, returning the batch and the source shape."""
        source_shape = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=self.interpolation)
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        return np.expand_dims(chw, axis=0), source_shape

    def _run(self, model_input: np.ndarray) -> np.ndarray:
        name = self.session.get_inputs()[0].name
        return self.session.run(None, {name: model_input})[0][0]


class ONNXDetector(_ONNXModel, DetectionModel):
    """ONNX detection model wrapper."""

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect animals in image."""
        detector_input, (h, w) = self._preprocess(image)

        results = []
        for x1, y1, x2, y2, conf, _ in self._run(detector_input):
            if conf < DETECTION_CONF_THRESHOLD:
                continue

            # Scale to original image size.
            results.append(
                {
                    "bbox": [
                        int(x1 * w / self.input_size[1]),
                        int(y1 * h / self.input_size[0]),
                        int(x2 * w / self.input_size[1]),
                        int(y2 * h / self.input_size[0]),
                    ],
                    "confidence": float(conf),
                    "class": AnimalClass.DOG,
                }
            )

        return results


class ONNXKeypoint(_ONNXModel, KeypointModel):
    """ONNX keypoint model wrapper."""

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect keypoints in image."""
        keypoint_input, (ch, cw) = self._preprocess(image)

        results = []
        for det in self._run(keypoint_input):
            if len(det) < 7:
                continue
            keypoints = det[6:].reshape((4, 3))

            # Scale keypoints to crop size.
            keypoints[:, 0] = keypoints[:, 0] * cw / self.input_size[1]
            keypoints[:, 1] = keypoints[:, 1] * ch / self.input_size[0]

            results.append(
                {"keypoints": keypoints.tolist(), "confidence": float(det[4])}
            )

        return results


class ONNXEmbedding(_ONNXModel, EmbeddingModel):
    """ONNX embedding model wrapper."""

    interpolation = cv2.INTER_AREA

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for image."""
        embedding_input, _ = self._preprocess(image)
        return self._run(embedding_input)
