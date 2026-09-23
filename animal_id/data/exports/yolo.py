"""Ultralytics YOLO detection labels, split lists and dataset YAML.

yolo.write(samples, ("dog",), {Source.COCO: 17000}, DATA_DIR / "detector/dogs_detection.yaml")
"""

import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

from animal_id.common.constants import DATA_DIR
from animal_id.data.sample import Sample

logger = logging.getLogger(__name__)


def write(
    samples: list[Sample],
    classes: tuple[str, ...],
    max_negatives: dict[str, int],
    yaml_path: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    """Writes a YOLO label beside each image, and train/val lists beside ``yaml_path``."""
    rng = random.Random(seed)
    kept, negatives = [], defaultdict(list)
    for sample in samples:
        boxes = [b for b in sample.boxes if b.label in classes]
        # An unlocated target can't be labelled; an empty label would call it background.
        if any(b.xyxy is None for b in boxes):
            continue
        (kept if boxes else negatives[sample.source]).append((sample, boxes))
    for source, group in sorted(negatives.items()):
        kept += rng.sample(
            group, min(max_negatives.get(source, len(group)), len(group))
        )

    lists = {"train": [], "val": []}
    for sample, boxes in kept:
        image = (DATA_DIR / sample.path).as_posix()
        # Ultralytics finds a label by swapping the last /images/ for /labels/.
        head, sep, tail = image.rpartition("/images/")
        if not sep:
            raise ValueError(f"YOLO needs images under an images/ dir: {sample.path}")
        label = Path(f"{head}/labels/{tail}").with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "".join(
                f"{classes.index(b.label)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                f"{x2 - x1:.6f} {y2 - y1:.6f}\n"
                for b in boxes
                for x1, y1, x2, y2 in [b.xyxy]
            )
        )
        # Hashing the path keeps a split stable when sources are added or removed.
        bucket = int(hashlib.sha1(sample.path.encode()).hexdigest()[:8], 16) / 16**8
        lists["val" if bucket < val_fraction else "train"].append(image)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    for split, images in lists.items():
        (yaml_path.parent / f"{split}.txt").write_text(
            "".join(f"{p}\n" for p in sorted(images))
        )
        logger.info(f"YOLO {split}: {len(images)} images")
    yaml_path.write_text(
        yaml.dump(
            {
                "path": DATA_DIR.as_posix(),
                "train": (yaml_path.parent / "train.txt").as_posix(),
                "val": (yaml_path.parent / "val.txt").as_posix(),
                "names": list(classes),
            },
            sort_keys=False,
        )
    )
