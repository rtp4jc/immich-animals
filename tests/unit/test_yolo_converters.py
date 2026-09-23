import json

import pytest
import yaml

from animal_id.keypoint.yolo_converter import CocoToYoloKeypointConverter

IMG_W, IMG_H = 640, 480


def _write_coco(ann_dir, file_names, annotations):
    ann_dir.mkdir(parents=True, exist_ok=True)
    images = [
        {"id": i, "file_name": name, "width": IMG_W, "height": IMG_H}
        for i, name in enumerate(file_names)
    ]
    for split in ("train", "val"):
        (ann_dir / f"annotations_{split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations})
        )


@pytest.fixture
def keypoint_converter(tmp_path):
    ann_dir = tmp_path / "coco"
    _write_coco(
        ann_dir,
        ["a.jpg", "b.jpg"],
        [
            {
                "id": 1,
                "image_id": 0,
                "bbox": [100, 100, 50, 60],
                # Third keypoint is flagged not-visible.
                "keypoints": [64, 48, 1, 128, 96, 1, 192, 144, 0, 256, 192, 1],
                "num_keypoints": 3,
            },
            {"id": 2, "image_id": 1, "bbox": [0, 0, 64, 48], "num_keypoints": 0},
        ],
    )
    (tmp_path / "keypoints").mkdir()
    return CocoToYoloKeypointConverter(
        coco_annotations_dir=str(ann_dir),
        labels_output_dir=str(tmp_path / "keypoints" / "labels"),
        data_root=str(tmp_path),
        yaml_output_path=str(tmp_path / "keypoints" / "k.yaml"),
    )


def test_keypoint_labels_are_flat_and_normalized(keypoint_converter, tmp_path):
    keypoint_converter.convert()
    fields = (tmp_path / "keypoints" / "labels" / "a.txt").read_text().split()
    assert fields[0] == "0"
    # bbox, then 4 keypoints of (x, y, visibility)
    assert len(fields) == 1 + 4 + 4 * 3
    assert fields[5:8] == ["0.100000", "0.100000", "2"]


def test_keypoint_visible_flag_promoted_to_two(keypoint_converter, tmp_path):
    """StanfordExtra only records 0/1; 1 must become YOLO's 2 (visible), otherwise
    every keypoint reads as occluded."""
    keypoint_converter.convert()
    fields = (tmp_path / "keypoints" / "labels" / "a.txt").read_text().split()
    visibilities = fields[7::3][:4]
    assert visibilities == ["2", "2", "0", "2"]


def test_keypoint_pads_when_annotation_has_none(keypoint_converter, tmp_path):
    keypoint_converter.convert()
    fields = (tmp_path / "keypoints" / "labels" / "b.txt").read_text().split()
    assert fields[5:] == ["0"] * 12


def test_keypoint_yaml_declares_pose_shape(keypoint_converter, tmp_path):
    keypoint_converter.convert()
    config = yaml.safe_load((tmp_path / "keypoints" / "k.yaml").read_text())
    assert config["kpt_shape"] == [4, 3]
    assert config["flip_idx"] == [0, 1, 3, 2]
