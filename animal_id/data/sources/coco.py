"""COCO 2017 train and val: every image, with a box for each animal."""

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from animal_id.data.sample import Box, Sample, Source, normalised_xyxy

ANIMALS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def _license(url: str) -> str:
    """ "http://creativecommons.org/licenses/by-nc-sa/2.0/" -> "CC BY-NC-SA 2.0"."""
    match = re.search(r"licenses/([a-z-]+)/([\d.]+)", url)
    return f"CC {match[1].upper()} {match[2]}" if match else url


def load(data_dir: Path) -> Iterator[Sample]:
    for split in ("train2017", "val2017"):
        coco = json.loads(
            (data_dir / f"coco/annotations/instances_{split}.json").read_text()
        )
        names = {c["id"]: c["name"] for c in coco["categories"]}
        licenses = {lic["id"]: _license(lic["url"]) for lic in coco["licenses"]}
        animals = defaultdict(list)
        for ann in coco["annotations"]:
            if names[ann["category_id"]] in ANIMALS:
                animals[ann["image_id"]].append(ann)

        for image in coco["images"]:
            boxes = []
            for ann in animals[image["id"]]:
                x, y, w, h = ann["bbox"]
                xyxy = normalised_xyxy(
                    x, y, x + w, y + h, image["width"], image["height"]
                )
                if xyxy:
                    boxes.append(Box(names[ann["category_id"]], xyxy))
            yield Sample(
                path=f"coco/images/{split}/{image['file_name']}",
                source=Source.COCO,
                license=licenses[image["license"]],
                boxes=tuple(boxes),
            )
