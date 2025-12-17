import json
from pathlib import Path
from unittest.mock import patch

import pytest

from animal_id.common.identity_loader import IdentityLoader


@pytest.fixture
def mock_data_env(tmp_path):
    """
    Creates a temporary directory structure mimicking the project root
    with data/identity_val.json and data/additional_identities/.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 1. Create identity_val.json (Base Data)
    val_data = [
        {"file_path": "path/to/img1.jpg", "identity_label": "id_A"},
        {"file_path": "path/to/img2.jpg", "identity_label": "id_A"},
        {"file_path": "path/to/img3.jpg", "identity_label": "id_B"},
    ]
    with open(data_dir / "identity_val.json", "w") as f:
        json.dump(val_data, f)

    # 2. Create additional identities
    add_dir = data_dir / "additional_identities"
    add_dir.mkdir()

    # Identity C (2 images)
    (add_dir / "id_C").mkdir()
    (add_dir / "id_C" / "extra1.jpg").touch()
    (add_dir / "id_C" / "extra2.jpg").touch()

    # Identity D (1 image)
    (add_dir / "id_D").mkdir()
    (add_dir / "id_D" / "extra3.jpg").touch()

    return tmp_path


def test_load_base_validation(mock_data_env):
    """Test loading base validation data works and maps fields correctly."""
    with patch("animal_id.common.identity_loader.DATA_DIR", mock_data_env / "data"):
        loader = IdentityLoader()
        data = loader.load_validation_data(include_additional=False)

        assert len(data) == 3
        # Check field mapping
        assert data[0]["image_path"] == "path/to/img1.jpg"
        assert data[0]["identity_label"] == "id_A"


def test_limit_per_identity(mock_data_env):
    """Test that max_per_identity correctly limits samples."""
    with patch("animal_id.common.identity_loader.DATA_DIR", mock_data_env / "data"):
        loader = IdentityLoader()
        # id_A has 2 images, limit to 1
        data = loader.load_validation_data(include_additional=False, max_per_identity=1)

        id_A_count = sum(1 for item in data if item["identity_label"] == "id_A")
        id_B_count = sum(1 for item in data if item["identity_label"] == "id_B")

        assert id_A_count == 1
        assert id_B_count == 1
        assert len(data) == 2


def test_scan_additional_identities(mock_data_env):
    """Test scanning the additional_identities directory."""
    with (
        patch("animal_id.common.identity_loader.DATA_DIR", mock_data_env / "data"),
        patch("animal_id.common.identity_loader.PROJECT_ROOT", mock_data_env),
    ):
        loader = IdentityLoader()
        identities = loader._scan_additional_identities()

        assert "id_C" in identities
        assert "id_D" in identities
        assert len(identities["id_C"]) == 2
        assert len(identities["id_D"]) == 1

        # Verify relative path storage
        # Should be something like 'data/additional_identities/id_C/extra1.jpg'
        path_str = identities["id_C"][0]
        assert str(Path("data/additional_identities/id_C")) in str(Path(path_str))


def test_create_augmented_dataset(mock_data_env):
    """Test combining base and additional data."""
    with (
        patch("animal_id.common.identity_loader.DATA_DIR", mock_data_env / "data"),
        patch("animal_id.common.identity_loader.PROJECT_ROOT", mock_data_env),
    ):
        loader = IdentityLoader()
        data = loader.load_validation_data(include_additional=True)

        # Base: 3 images (2 id_A, 1 id_B)
        # Add: 3 images (2 id_C, 1 id_D)
        # Total: 6
        assert len(data) == 6

        all_labels = [item["identity_label"] for item in data]
        assert "id_A" in all_labels
        assert "id_C" in all_labels


def test_augmented_dataset_limits(mock_data_env):
    """Test limits apply to both base and additional data."""
    with (
        patch("animal_id.common.identity_loader.DATA_DIR", mock_data_env / "data"),
        patch("animal_id.common.identity_loader.PROJECT_ROOT", mock_data_env),
    ):
        loader = IdentityLoader()
        # Limit 1 per identity
        data = loader.load_validation_data(include_additional=True, max_per_identity=1)

        # id_A: 2->1, id_B: 1->1, id_C: 2->1, id_D: 1->1
        # Total: 4
        assert len(data) == 4

        # Check counts
        counts = {}
        for item in data:
            lbl = item["identity_label"]
            counts[lbl] = counts.get(lbl, 0) + 1

        assert counts["id_A"] == 1
        assert counts["id_C"] == 1
