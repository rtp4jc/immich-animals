import json

import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.fixture
def mock_image_dataset(tmp_path):
    """
    Creates a temporary dataset with dummy images and a JSON annotation file.
    Returns the path to the dataset.json file.
    """
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir()

    images_dir = dataset_dir / "images"
    images_dir.mkdir()

    annotations = []

    # Create 5 dummy images
    for i in range(5):
        # Create random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"img_{i}.jpg"
        img.save(img_path)

        # Add to annotations
        annotations.append(
            {
                "file_path": str(img_path),
                "identity_label": i % 2,  # 2 identities
                "bbox": [0, 0, 224, 224],
            }
        )

    # Save annotations
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f)

    return str(json_path)


@pytest.fixture
def mock_yolo_dataset(tmp_path):
    """
    Creates a temporary YOLO dataset for detection.
    Returns the path to the data.yaml file.
    """
    dataset_dir = tmp_path / "mock_yolo_det"
    dataset_dir.mkdir()

    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)

    # Create 5 dummy images and labels
    for i in range(5):
        # Image
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = dataset_dir / "images" / "train" / f"img_{i}.jpg"
        img.save(img_path)

        # Label (class x y w h)
        label_path = dataset_dir / "labels" / "train" / f"img_{i}.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    # Create data.yaml
    yaml_content = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/train",  # Use same for val
        "nc": 1,
        "names": ["dog"],
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    return str(yaml_path)


@pytest.fixture
def mock_keypoint_dataset(tmp_path):
    """
    Creates a temporary YOLO dataset for keypoint estimation.
    Returns the path to the data.yaml file.
    """
    dataset_dir = tmp_path / "mock_yolo_pose"
    dataset_dir.mkdir()

    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)

    # Create 5 dummy images and labels
    for i in range(5):
        # Image
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = dataset_dir / "images" / "train" / f"img_{i}.jpg"
        img.save(img_path)

        # Label (class x y w h + 17 keypoints * 3)
        # 17 keypoints is standard for COCO pose.
        # Format: x y visibility.
        # Let's create dummy keypoints: 0.5 0.5 2 (visible) repeated 17 times
        kpts = " ".join(["0.5 0.5 2"] * 17)
        label_content = f"0 0.5 0.5 0.2 0.2 {kpts}\n"

        label_path = dataset_dir / "labels" / "train" / f"img_{i}.txt"
        with open(label_path, "w") as f:
            f.write(label_content)

    # Create data.yaml
    yaml_content = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": ["dog"],
        "kpt_shape": [17, 3],
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    return str(yaml_path)
