"""Converts COCO keypoint datasets to YOLO pose format for training with Ultralytics."""

import json
import logging
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 4


class CocoToYoloKeypointConverter:
    """Convert COCO keypoint format to YOLO pose format."""

    def __init__(
        self,
        coco_annotations_dir: str,
        labels_output_dir: str,
        data_root: str,
        yaml_output_path: str,
    ):
        """Initialize converter with paths."""
        self.coco_annotations_dir = Path(coco_annotations_dir)
        self.labels_output_dir = Path(labels_output_dir)
        self.data_root = Path(data_root)
        self.yaml_output_path = Path(yaml_output_path)

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

    def convert_split(self, split_name: str) -> list[str]:
        """Processes a single split (e.g., 'train' or 'val')."""
        coco_json_path = self.coco_annotations_dir / f"annotations_{split_name}.json"

        if not coco_json_path.exists():
            logger.warning(
                f"Annotation file not found, skipping split '{split_name}': {coco_json_path}"
            )
            return []

        logger.info(f"Processing {split_name} split from {coco_json_path}...")
        with open(coco_json_path) as f:
            coco_data = json.load(f)

        images_map = {img["id"]: img for img in coco_data["images"]}
        annotations_by_image: dict[int, list[dict]] = {}
        for ann in coco_data.get("annotations", []):
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        image_paths = []
        written_labels_count = 0

        for img_id, image_info in tqdm(
            images_map.items(), desc=f"Generating {split_name} labels"
        ):
            relative_img_path = image_info["file_name"]
            image_paths.append(str(self.data_root / relative_img_path))

            lines = self._label_lines(
                annotations_by_image.get(img_id, []),
                image_info["width"],
                image_info["height"],
                relative_img_path,
            )
            if lines:
                written_labels_count += 1

            # Images without annotations still get an empty label file: that is how
            # YOLO marks a negative sample.
            label_path = self._label_path(relative_img_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("".join(lines))

        logger.info(
            f"Split {split_name}: {len(image_paths)} images, "
            f"wrote labels for {written_labels_count}"
        )
        return image_paths

    def create_yaml_config(
        self, train_image_paths: list[str], val_image_paths: list[str]
    ) -> None:
        """Writes the train/val path lists and the Ultralytics dataset YAML."""
        txt_paths = {}
        for split, paths in (("train", train_image_paths), ("val", val_image_paths)):
            txt_path = self.data_root / "keypoints" / f"{split}.txt"
            txt_path.write_text(
                "".join(f"{Path(p).as_posix()}\n" for p in sorted(paths))
            )
            logger.info(f"Created {txt_path.name} with {len(paths)} image paths.")
            txt_paths[split] = txt_path

        yaml_content = {
            "path": Path(self.data_root.resolve()).as_posix(),
            "train": Path(txt_paths["train"].resolve()).as_posix(),
            "val": Path(txt_paths["val"].resolve()).as_posix(),
            "nc": 1,
            "names": ["dog"],
            "kpt_shape": [NUM_KEYPOINTS, 3],  # 4 keypoints, 3 dims (x, y, visibility)
            # Keypoints: ['nose', 'chin', 'left_ear_base', 'right_ear_base']
            # Indices:      0,      1,           2,               3
            "flip_idx": [0, 1, 3, 2],  # Swap left and right ear base
        }

        with open(self.yaml_output_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Successfully created YAML config at: {self.yaml_output_path}")

    def convert(self) -> None:
        """Main conversion function."""
        logger.info("Converting cropped COCO keypoint dataset to YOLO pose format")

        if self.labels_output_dir.exists():
            shutil.rmtree(self.labels_output_dir)
        self.labels_output_dir.mkdir(parents=True)

        train_paths = self.convert_split("train")
        val_paths = self.convert_split("val")

        if not train_paths and not val_paths:
            logger.error(
                f"No data was processed. Check that your COCO JSON files exist in {self.coco_annotations_dir}"
            )
            return

        self.create_yaml_config(train_paths, val_paths)
        logger.info("Conversion complete!")


def create_default_converter() -> CocoToYoloKeypointConverter:
    """Create converter with default paths."""
    return CocoToYoloKeypointConverter(
        coco_annotations_dir="data/keypoints/coco",
        labels_output_dir="data/keypoints/labels",
        data_root="data",
        yaml_output_path="data/keypoints/dogs_keypoints_only.yaml",
    )
