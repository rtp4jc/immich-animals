"""Converts COCO detection datasets to YOLO format for training with Ultralytics."""

import logging

from animal_id.common.yolo_converter import CocoToYoloConverter

logger = logging.getLogger(__name__)


class CocoToYoloDetectionConverter(CocoToYoloConverter):
    """Convert COCO detection format to YOLO format."""

    split_subdir = "detector"
    description = "COCO Detector Dataset to YOLOv8 Detection Format"

    def _label_lines(self, annotations, img_width, img_height, relative_img_path):
        lines = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]

            # Clamp bbox to image boundaries.
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width, x + w)
            y2 = min(img_height, y + h)

            if x1 != x or y1 != y or x2 != (x + w) or y2 != (y + h):
                logger.warning(
                    f"Clamped bbox for {relative_img_path}. "
                    f"Original: {[x, y, w, h]}, Clamped: {[x1, y1, x2 - x1, y2 - y1]}"
                )

            final_w = x2 - x1
            final_h = y2 - y1
            if final_w <= 0 or final_h <= 0:
                continue

            # YOLO format: normalized center coordinates.
            lines.append(
                f"0 {(x1 + final_w / 2) / img_width:.6f} "
                f"{(y1 + final_h / 2) / img_height:.6f} "
                f"{final_w / img_width:.6f} {final_h / img_height:.6f}\n"
            )
        return lines


def create_default_converter() -> CocoToYoloDetectionConverter:
    """Create converter with default paths."""
    return CocoToYoloDetectionConverter(
        coco_annotations_dir="data/detector/coco",
        labels_output_dir="data",
        data_root="data",
        yaml_output_path="data/detector/dogs_detection.yaml",
    )
