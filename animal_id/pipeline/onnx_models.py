import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Tuple, Optional

from .models import DetectionModel, KeypointModel, EmbeddingModel, AnimalClass

class ONNXDetector(DetectionModel):
    """ONNX detection model wrapper."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect animals in image."""
        detector_input, original_shape = self._preprocess(image)
        detections = self.session.run(
            None, {self.session.get_inputs()[0].name: detector_input}
        )[0][0]

        results = []
        h, w = original_shape
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            if conf < 0.1:
                continue

            # Scale to original image size
            x1 = int(x1 * w / self.input_size[1])
            x2 = int(x2 * w / self.input_size[1])
            y1 = int(y1 * h / self.input_size[0])
            y2 = int(y2 * h / self.input_size[0])

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": AnimalClass.DOG,
                }
            )

        return results

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for detection."""
        original_shape = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_fp = resized.astype(np.float32) / 255.0
        img_fp = np.transpose(img_fp, (2, 0, 1))
        return np.expand_dims(img_fp, axis=0), original_shape


class ONNXKeypoint(KeypointModel):
    """ONNX keypoint model wrapper."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect keypoints in image."""
        keypoint_input, crop_shape = self._preprocess(image)
        detections = self.session.run(
            None, {self.session.get_inputs()[0].name: keypoint_input}
        )[0][0]

        results = []
        ch, cw = crop_shape
        for det in detections:
            if len(det) < 7:
                continue
            conf = det[4]
            keypoints = det[6:].reshape((4, 3))

            # Scale keypoints to crop size
            keypoints[:, 0] = keypoints[:, 0] * cw / self.input_size[1]
            keypoints[:, 1] = keypoints[:, 1] * ch / self.input_size[0]

            results.append({"keypoints": keypoints.tolist(), "confidence": float(conf)})

        return results

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for keypoint detection."""
        crop_shape = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_fp = resized.astype(np.float32) / 255.0
        img_fp = np.transpose(img_fp, (2, 0, 1))
        return np.expand_dims(img_fp, axis=0), crop_shape


class ONNXEmbedding(EmbeddingModel):
    """ONNX embedding model wrapper."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = self.session.get_inputs()[0].shape[2:]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for image."""
        embedding_input = self._preprocess(image)
        embedding = self.session.run(
            None, {self.session.get_inputs()[0].name: embedding_input}
        )[0][0]
        return embedding

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for embedding."""
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        img_fp = resized.astype(np.float32) / 255.0
        img_fp = np.transpose(img_fp, (2, 0, 1))
        return np.expand_dims(img_fp, axis=0)
