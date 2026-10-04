"""ONNX runtime wrappers for the three pipeline stages."""

import ast
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
        return self.session.run(None, {name: model_input})[0]


class ONNXDetector(_ONNXModel, DetectionModel):
    """ONNX detection model wrapper."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Ultralytics records the class names as a dict literal, e.g. "{0: 'dog'}".
        names = self.session.get_modelmeta().custom_metadata_map["names"]
        self.classes = {i: AnimalClass(n) for i, n in ast.literal_eval(names).items()}

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, int, int]]:
        """Letterbox, returning the scale and left/top padding that map boxes back.

        YOLO trains on aspect-preserved images padded to square; stretched phone
        photos cost the detector ~7pp recall on owner photos.
        """
        height, width = image.shape[:2]
        scale = min(self.input_size[0] / height, self.input_size[1] / width)
        h, w = round(height * scale), round(width * scale)
        top, left = (self.input_size[0] - h) // 2, (self.input_size[1] - w) // 2
        canvas = np.full((*self.input_size, 3), 114, np.uint8)  # Ultralytics' pad
        canvas[top : top + h, left : left + w] = cv2.resize(
            image, (w, h), interpolation=self.interpolation
        )
        chw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))
        return chw[None], (scale, left, top)

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect animals in image."""
        h, w = image.shape[:2]
        detector_input, (scale, left, top) = self._preprocess(image)

        results = []
        for x1, y1, x2, y2, conf, class_id in self._run(detector_input)[0]:
            if conf < DETECTION_CONF_THRESHOLD:
                continue

            results.append(
                {
                    "bbox": [
                        int(np.clip((x1 - left) / scale, 0, w)),
                        int(np.clip((y1 - top) / scale, 0, h)),
                        int(np.clip((x2 - left) / scale, 0, w)),
                        int(np.clip((y2 - top) / scale, 0, h)),
                    ],
                    "confidence": float(conf),
                    "class": self.classes[int(class_id)],
                }
            )

        return results


class ONNXKeypoint(_ONNXModel, KeypointModel):
    """ONNX keypoint model wrapper."""

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect keypoints in image."""
        keypoint_input, (ch, cw) = self._preprocess(image)

        results = []
        for det in self._run(keypoint_input)[0]:
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
    """ONNX embedding model wrapper.

    Unlike the YOLO stages, the embedder is trained on ImageNet-normalized
    input (``IdentityDataset``), so serving it raw [0, 1] is a train/serve skew
    that costs real accuracy without any error.
    """

    interpolation = cv2.INTER_AREA
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        batch, source_shape = super()._preprocess(image)
        return (batch - self.mean) / self.std, source_shape

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for image."""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Embeds several crops in one session run."""
        return self._run(np.concatenate([self._preprocess(i)[0] for i in images]))
