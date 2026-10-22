"""Tensors the sidecar feeds its ONNX models; numpy and OpenCV only.

Kept apart from serve.py so the training project's tests can check it against
animal_id.pipeline.onnx_models without the sidecar's web dependencies.
"""

import cv2
import numpy as np


def letterbox(
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


def embedder_blob(
    crop: np.ndarray, size: tuple[int, int], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    resized = cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    blob = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return (blob - mean) / std
