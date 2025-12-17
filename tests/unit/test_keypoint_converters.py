from unittest.mock import patch

import numpy as np
import pytest

from animal_id.keypoint.dataset_converter import CocoKeypointDatasetConverter


@pytest.fixture
def mock_kp_config(tmp_path):
    return {
        "source_json": str(tmp_path / "source.json"),
        "cropped_image_dir": str(tmp_path / "crops"),
        "output_coco_dir": str(tmp_path / "coco"),
        "data_root": str(tmp_path),
        "padding_factor": 0.1,
        "train_val_split": 0.5,
    }


def test_keypoint_mapping_and_transform(mock_kp_config):
    converter = CocoKeypointDatasetConverter(mock_kp_config)

    # StanfordExtra has 24 joints. We map:
    # nose: 16, chin: 17, left_ear_base: 14, right_ear_base: 15

    # Create dummy joints: 24 points of [x, y, v]
    joints = [[0, 0, 0]] * 24

    # Set target joints
    joints[16] = [100, 100, 1]  # Nose
    joints[17] = [100, 120, 1]  # Chin
    joints[14] = [80, 80, 1]  # L Ear
    joints[15] = [120, 80, 0]  # R Ear (not visible)

    # Apply offset (simulate crop)
    x_offset, y_offset = 50, 50

    # Expected: (x-50, y-50, v)
    # Nose: 50, 50, 1
    # Chin: 50, 70, 1
    # L Ear: 30, 30, 1
    # R Ear: 70, 30, 0

    mapped = converter._map_and_transform_keypoints(joints, x_offset, y_offset)

    # 4 keypoints * 3 values = 12
    assert len(mapped) == 12

    # Check values
    # Nose
    assert mapped[0:3] == [50, 50, 1]
    # Chin
    assert mapped[3:6] == [50, 70, 1]
    # L Ear
    assert mapped[6:9] == [30, 30, 1]
    # R Ear
    # Visibility is 0, so converter forces it to [0, 0, 0]
    assert mapped[9:12] == [0, 0, 0]


def test_validate_keypoints(mock_kp_config):
    converter = CocoKeypointDatasetConverter(mock_kp_config)

    # Width 100, Height 100
    # [x, y, v]
    kpts = [
        50,
        50,
        1,  # Valid
        150,
        50,
        1,  # Invalid X
        50,
        150,
        1,  # Invalid Y
        -10,
        50,
        1,  # Invalid X negative
        50,
        50,
        0,  # Already invisible
    ]

    valid = converter._validate_keypoints(kpts, 100, 100)

    # Valid
    assert valid[0:3] == [50, 50, 1]
    # Invalid X -> 0,0,0
    assert valid[3:6] == [0, 0, 0]
    # Invalid Y -> 0,0,0
    assert valid[6:9] == [0, 0, 0]
    # Invalid X neg -> 0,0,0
    assert valid[9:12] == [0, 0, 0]
    # Invisible -> Unchanged
    assert valid[12:15] == [50, 50, 0]


@patch("cv2.imread")
@patch("cv2.imwrite")
def test_process_image(mock_imwrite, mock_imread, mock_kp_config, tmp_path):
    converter = CocoKeypointDatasetConverter(mock_kp_config)

    # Mock image: 200x200
    mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img

    # Setup directories
    img_dir = tmp_path / "stanford_dogs" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "test.jpg").touch()

    entry = {
        "img_path": "test.jpg",
        "img_bbox": [50, 50, 100, 100],  # Center crop
        "joints": [[0, 0, 0]] * 24,  # Dummy joints
        "is_multiple_dogs": False,
    }

    img_entry, ann_entry = converter._process_image(entry, 0, 0)

    assert img_entry is not None
    assert ann_entry is not None

    # Check crop size with padding factor 0.1
    # Bbox: 100x100
    # Pad: 10 each side
    # Result: 120x120
    assert img_entry["width"] == 120
    assert img_entry["height"] == 120
    assert ann_entry["bbox"] == [10, 10, 100, 100]  # Relative to crop
