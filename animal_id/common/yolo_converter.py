"""Shared COCO -> YOLO dataset conversion.

The detection and pose variants differ only in where label files go, what each label
line contains, and a couple of extra dataset-YAML keys. Everything else — reading the
COCO split, grouping annotations, writing one label file per image, emitting the
train/val path lists and the dataset YAML — lives here.
"""

import json
import logging
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CocoToYoloConverter:
    """Base converter. Subclasses define the label format; see the module docstring."""

    # Subdirectory of data_root holding train.txt / val.txt.
    split_subdir: str = ""
    # Human-readable name used in the conversion log.
    description: str = "COCO Dataset to YOLO Format"
    # Extra keys merged into the dataset YAML (e.g. kpt_shape, flip_idx).
    extra_yaml: dict = {}
    # Whether convert() wipes the label directory before writing.
    clear_labels_dir: bool = False

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

    def _label_path(self, relative_img_path: str) -> Path:
        """Label file for an image, mirroring the image tree under labels/."""
        mirrored = str(Path(relative_img_path)).replace("images", "labels", 1)
        return (self.labels_output_dir / Path(mirrored)).with_suffix(".txt")

    def _label_lines(
        self,
        annotations: list[dict],
        img_width: int,
        img_height: int,
        relative_img_path: str,
    ) -> list[str]:
        """YOLO label lines for one image's annotations."""
        raise NotImplementedError

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
            txt_path = self.data_root / self.split_subdir / f"{split}.txt"
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
            **self.extra_yaml,
        }

        with open(self.yaml_output_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Successfully created YAML config at: {self.yaml_output_path}")

    def convert(self) -> None:
        """Main conversion function."""
        logger.info(f"Converting {self.description}")

        if self.clear_labels_dir:
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
