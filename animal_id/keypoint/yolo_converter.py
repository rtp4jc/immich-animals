"""Converts COCO keypoint datasets to YOLO pose format for training with Ultralytics."""

from pathlib import Path

from animal_id.common.yolo_converter import CocoToYoloConverter

NUM_KEYPOINTS = 4


class CocoToYoloKeypointConverter(CocoToYoloConverter):
    """Convert COCO keypoint format to YOLO pose format."""

    split_subdir = "keypoints"
    description = "Cropped COCO Keypoint Dataset to YOLOv8 Pose Format"
    clear_labels_dir = True
    extra_yaml = {
        "kpt_shape": [NUM_KEYPOINTS, 3],  # 4 keypoints, 3 dims (x, y, visibility)
        # Keypoints: ['nose', 'chin', 'left_ear_base', 'right_ear_base']
        # Indices:      0,      1,           2,               3
        "flip_idx": [0, 1, 3, 2],  # Swap left and right ear base
    }

    def _label_path(self, relative_img_path):
        # Cropped images all live in one folder, so labels are flat too.
        return self.labels_output_dir / Path(relative_img_path).with_suffix(".txt").name

    def _label_lines(self, annotations, img_width, img_height, relative_img_path):
        lines = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]

            keypoints = ann.get("keypoints", [])
            if keypoints and ann.get("num_keypoints", 0) > 0:
                kpts_str = ""
                for i in range(0, len(keypoints), 3):
                    # YOLO visibility is 0 (absent), 1 (present, occluded) or 2
                    # (visible). StanfordExtra only has 0/1, so map 1 -> 2: leaving it
                    # at 1 would tell YOLO everything is occluded, which is wrong.
                    visibility = 2 if keypoints[i + 2] > 0 else 0
                    kpts_str += (
                        f" {keypoints[i] / img_width:.6f}"
                        f" {keypoints[i + 1] / img_height:.6f} {visibility}"
                    )
            else:
                kpts_str = " 0" * (NUM_KEYPOINTS * 3)

            lines.append(
                f"0 {(x + w / 2) / img_width:.6f} {(y + h / 2) / img_height:.6f} "
                f"{w / img_width:.6f} {h / img_height:.6f}{kpts_str}\n"
            )
        return lines


def create_default_converter() -> CocoToYoloKeypointConverter:
    """Create converter with default paths."""
    return CocoToYoloKeypointConverter(
        coco_annotations_dir="data/keypoints/coco",
        labels_output_dir="data/keypoints/labels",
        data_root="data",
        yaml_output_path="data/keypoints/dogs_keypoints_only.yaml",
    )
