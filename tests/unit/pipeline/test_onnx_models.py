from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from animal_id.pipeline.models import AnimalClass
from animal_id.pipeline.onnx_models import ONNXDetector, ONNXEmbedding, ONNXKeypoint


@pytest.fixture
def sample_image():
    """A sample 800x1200 image."""
    return np.random.randint(0, 256, (800, 1200, 3), dtype=np.uint8)


@patch("animal_id.pipeline.onnx_models.ort.InferenceSession")
class TestONNXDetector:
    def test_preprocess(self, mock_session_class, sample_image):
        """Test that preprocessing resizes, normalizes, and transposes the image correctly."""
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 640, 640]
        mock_session_class.return_value.get_inputs.return_value = [mock_input]

        detector = ONNXDetector("dummy_path.onnx")
        processed_image, original_shape = detector._preprocess(sample_image)

        assert original_shape == (800, 1200)
        assert processed_image.shape == (1, 3, 640, 640)  # Batch, C, H, W
        assert processed_image.dtype == np.float32
        assert np.max(processed_image) <= 1.0
        assert np.min(processed_image) >= 0.0

    def test_predict_parsing(self, mock_session_class, sample_image):
        """Test that the raw ONNX output is parsed and scaled correctly."""
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 640, 640]
        mock_session = mock_session_class.return_value
        mock_session.get_inputs.return_value = [mock_input]

        # Mock the ONNX session to return a predefined detection
        # The real output is a list containing the tensor.
        # The tensor shape is (num_detections, 6) where 6 = x1, y1, x2, y2, conf, class_id
        mock_output = np.array(
            [
                [100, 150, 200, 250, 0.9, 0],
                [300, 350, 400, 450, 0.05, 0],  # Should be filtered out
            ]
        )
        # The source code uses result[0][0], so we nest the array
        mock_session.run.return_value = [[mock_output]]

        detector = ONNXDetector("dummy_path.onnx")
        results = detector.predict(sample_image)

        # Original image is 800 (H) x 1200 (W)
        # Input size is 640 (H) x 640 (W)
        # Scale W: 1200 / 640 = 1.875
        # Scale H: 800 / 640 = 1.25

        assert len(results) == 1  # Low confidence detection filtered out
        det = results[0]

        expected_x1 = int(100 * 1200 / 640)  # 187
        expected_y1 = int(150 * 800 / 640)  # 187
        expected_x2 = int(200 * 1200 / 640)  # 375
        expected_y2 = int(250 * 800 / 640)  # 312

        assert det["class"] == AnimalClass.DOG
        assert det["confidence"] == pytest.approx(0.9)
        assert det["bbox"] == [expected_x1, expected_y1, expected_x2, expected_y2]


@patch("animal_id.pipeline.onnx_models.ort.InferenceSession")
class TestONNXKeypoint:
    def test_predict_parsing(self, mock_session_class, sample_image):
        """Test that raw keypoint output is parsed and scaled correctly."""
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 640, 640]
        mock_session = mock_session_class.return_value
        mock_session.get_inputs.return_value = [mock_input]

        # Mock output shape is (num_instances, 18)
        # where 18 = x1,y1,x2,y2, conf, cls, kpt1_x,kpt1_y,kpt1_conf,...
        mock_output = np.array(
            [
                [
                    10,
                    20,
                    30,
                    40,
                    0.95,
                    0,  # Bbox info
                    50,
                    60,
                    0.99,  # Kpt 1
                    70,
                    80,
                    0.98,  # Kpt 2
                    90,
                    100,
                    0.97,  # Kpt 3
                    110,
                    120,
                    0.96,  # Kpt 4
                ]
            ]
        )
        # The source code uses result[0][0], so we nest the array
        mock_session.run.return_value = [[mock_output]]

        # Simulate a crop passed to the model
        crop_image = cv2.resize(sample_image, (400, 300))  # Crop is 300H x 400W

        keypointer = ONNXKeypoint("dummy_path.onnx")
        results = keypointer.predict(crop_image)

        # Scale W: 400 / 640
        # Scale H: 300 / 640
        assert len(results) == 1
        kpt_result = results[0]
        keypoints = np.array(kpt_result["keypoints"])

        assert kpt_result["confidence"] == pytest.approx(0.95)
        assert keypoints.shape == (4, 3)

        # Check first keypoint
        expected_kpt1_x = 50 * 400 / 640
        expected_kpt1_y = 60 * 300 / 640
        assert keypoints[0, 0] == pytest.approx(expected_kpt1_x)
        assert keypoints[0, 1] == pytest.approx(expected_kpt1_y)
        assert keypoints[0, 2] == pytest.approx(0.99)


@patch("animal_id.pipeline.onnx_models.ort.InferenceSession")
class TestONNXEmbedding:
    def test_predict_returns_correct_shape(self, mock_session_class, sample_image):
        """Test that the embedding model returns a vector of the correct shape."""
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 224, 224]  # Embedding models have different sizes
        mock_session = mock_session_class.return_value
        mock_session.get_inputs.return_value = [mock_input]

        # Mock the session to return an embedding vector
        mock_embedding = np.random.rand(1, 512).astype(np.float32)
        mock_session.run.return_value = [mock_embedding]

        embedder = ONNXEmbedding("dummy_path.onnx")
        result = embedder.predict(sample_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)
