"""DogReID-1553: owner phone-video frames, one boxed dog per frame, identity from the dataset."""

import csv
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from animal_id.data.sample import Box, Sample, Source, normalised_xyxy

ROOT = "dogreid"
LICENSE = "CC0 1.0"  # doi:10.7910/DVN/LVTRLG


def load(data_dir: Path) -> Iterator[Sample]:
    root = data_dir / ROOT
    boxes = {
        r["VIDEO_ID"]: r for r in csv.DictReader(open(root / "bounding_boxes.tab"))
    }
    for row in csv.DictReader(open(root / "splits.tab")):
        # Query/gallery frames are the published open-set benchmark, scene-disjoint
        # by construction; they stay out of every training export.
        if row["SPLIT_OPEN_SET"] != "train":
            continue
        dog, video = row["DOG_ID"], row["VIDEO_ID"]
        path = f"{ROOT}/images/{dog}/{dog}-{video}.jpg"
        width, height = Image.open(data_dir / path).size  # reads the header only
        b = boxes[video]
        x, y = float(b["x_top_left"]), float(b["y_top_left"])
        xyxy = normalised_xyxy(
            x, y, x + float(b["width"]), y + float(b["height"]), width, height
        )
        yield Sample(
            path=path,
            source=Source.DOGREID,
            license=LICENSE,
            boxes=(Box("dog", xyxy, identity=dog),),
        )
