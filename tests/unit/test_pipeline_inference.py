from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from animal_id.pipeline.ambidextrous_axolotl import AmbidextrousAxolotl
from animal_id.pipeline.models import AnimalClass


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    return detector


@pytest.fixture
def mock_keypoint():
    kp = MagicMock()
    return kp


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    return embedder


@pytest.fixture
def pipeline(mock_detector, mock_keypoint, mock_embedder):
    return AmbidextrousAxolotl(
        detector=mock_detector,
        keypoint_model=mock_keypoint,
        embedding_model=mock_embedder,
        target_class=AnimalClass.DOG,
        use_keypoints=True,
    )


@pytest.fixture
def dummy_image_path(tmp_path):
    p = tmp_path / "dummy.jpg"
    # Create a valid 100x100 dummy image using cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return str(p)


def test_generate_embedding_no_detection(pipeline, dummy_image_path):
    """Test returns None if no detection found."""
    pipeline.detector.predict.return_value = []

    emb = pipeline.generate_embedding(dummy_image_path)
    assert emb is None
    pipeline.detector.predict.assert_called_once()


def test_generate_embedding_wrong_class(pipeline, dummy_image_path):
    """Test filters out non-target class."""
    pipeline.detector.predict.return_value = [
        {"bbox": [0, 0, 50, 50], "confidence": 0.9, "class": AnimalClass.CAT}
    ]
    emb = pipeline.generate_embedding(dummy_image_path)
    assert emb is None


def test_generate_embedding_basic_flow(pipeline, dummy_image_path):
    """Test successful flow: Detect -> Embed."""
    pipeline.detector.predict.return_value = [
        {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class": AnimalClass.DOG}
    ]
    # Mock keypoint to return nothing (fallback to detection crop)
    pipeline.keypoint_model.predict.return_value = []

    expected_embedding = np.random.rand(128)
    pipeline.embedding_model.predict.return_value = expected_embedding

    emb = pipeline.generate_embedding(dummy_image_path)

    assert emb is not None
    assert np.array_equal(emb, expected_embedding)

    # Verify calls
    pipeline.detector.predict.assert_called()
    pipeline.keypoint_model.predict.assert_called()
    pipeline.embedding_model.predict.assert_called()


def test_generate_embedding_with_keypoint_refinement(pipeline, dummy_image_path):
    """Test keypoint refinement logic."""
    # Detector says [10, 10, 90, 90]
    pipeline.detector.predict.return_value = [
        {"bbox": [10, 10, 90, 90], "confidence": 0.9, "class": AnimalClass.DOG}
    ]

    # Keypoints found inside: Tight crop around [40, 40, 60, 60]
    pipeline.keypoint_model.predict.return_value = [
        {
            "confidence": 0.9,
            "keypoints": [[40, 40, 1.0], [60, 60, 1.0]],  # High visibility
        }
    ]

    pipeline.embedding_model.predict.return_value = np.zeros(128)

    pipeline.generate_embedding(dummy_image_path)

    # Verify embedder was called with a crop.
    # We can check the shape of the input to predict()
    args, _ = pipeline.embedding_model.predict.call_args
    input_crop = args[0]

    # Original crop was 100x100 image, detection 10-90 (80x80).
    # Keypoints 40-60 (20x20).
    # Refinement adds padding 20% of width (4px).
    # So crop should be roughly 28x28 (40-4 to 60+4).
    # It won't be exactly 80x80 (detector crop).

    assert input_crop.shape[0] < 80
    assert input_crop.shape[1] < 80
    assert input_crop.shape[0] > 10


def test_predict_similarity(pipeline, dummy_image_path):
    """Test matching against gallery."""
    # Setup gallery
    pipeline.gallery_embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1]]  # Orthogonal  # Similar
    )
    pipeline.gallery_paths = ["path1", "path2"]

    # Mock generation to return [1.0, 0.0] (match first item)
    # We patch the METHOD on the instance, not the class, for this test
    with patch.object(
        pipeline, "generate_embedding", return_value=np.array([1.0, 0.0])
    ):
        found, results = pipeline.predict(dummy_image_path)

        assert found is True
        assert len(results) == 2
        # First match should be path1 with score 1.0
        assert results[0][0] == "path1"
        assert pytest.approx(results[0][1]) == 1.0
        assert results[1][0] == "path2"
        assert results[1][1] < 1.0
