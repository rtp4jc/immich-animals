"""
Tests for EmbeddingDatasetConverter's train/val/test identity split.

These verify the open-set (no-leakage) protocol: the three splits must contain
pairwise-disjoint identity sets and all three must be non-empty. The converter
only reads filenames to build the split, so dummy (empty) image files are enough
and the real DogFaceNet dataset is not required.
"""

import json

from animal_id.embedding.dataset_converter import EmbeddingDatasetConverter


def _make_fake_dataset(root, num_identities=20, images_per_identity=6):
    """Create a tmp directory tree of fake identity folders with dummy images."""
    for i in range(num_identities):
        identity_dir = root / f"identity_{i:03d}"
        identity_dir.mkdir(parents=True)
        for j in range(images_per_identity):
            (identity_dir / f"img_{j}.jpg").touch()


def _identity_set(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {item["identity_label"] for item in data}


def test_train_val_test_splits_are_disjoint_and_non_empty(tmp_path):
    source = tmp_path / "dogfacenet"
    source.mkdir()
    _make_fake_dataset(source, num_identities=20, images_per_identity=6)

    train_json = tmp_path / "out" / "identity_train.json"
    val_json = tmp_path / "out" / "identity_val.json"
    test_json = tmp_path / "out" / "identity_test.json"

    converter = EmbeddingDatasetConverter(
        source_path=str(source),
        output_train_json=str(train_json),
        output_val_json=str(val_json),
        output_test_json=str(test_json),
        min_images_per_identity=5,
        val_split_ratio=0.15,
        test_split_ratio=0.15,
    )
    converter.convert()

    train_ids = _identity_set(train_json)
    val_ids = _identity_set(val_json)
    test_ids = _identity_set(test_json)

    # All three splits are non-empty.
    assert train_ids, "train split is empty"
    assert val_ids, "val split is empty"
    assert test_ids, "test split is empty"

    # Pairwise disjoint identity sets (open-set, no identity leakage).
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Every filtered identity ends up in exactly one split.
    assert len(train_ids) + len(val_ids) + len(test_ids) == 20


def test_split_is_reproducible(tmp_path):
    """The fixed random.seed(42) must produce identical splits across runs."""
    source = tmp_path / "dogfacenet"
    source.mkdir()
    _make_fake_dataset(source, num_identities=20, images_per_identity=6)

    def run(out_name):
        out = tmp_path / out_name
        converter = EmbeddingDatasetConverter(
            source_path=str(source),
            output_train_json=str(out / "identity_train.json"),
            output_val_json=str(out / "identity_val.json"),
            output_test_json=str(out / "identity_test.json"),
        )
        converter.convert()
        return (
            _identity_set(out / "identity_train.json"),
            _identity_set(out / "identity_val.json"),
            _identity_set(out / "identity_test.json"),
        )

    assert run("run_a") == run("run_b")
