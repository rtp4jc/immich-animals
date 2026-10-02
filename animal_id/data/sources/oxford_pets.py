"""Oxford-IIIT Pets: one cat or dog per photo, boxed from its segmentation trimap."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from animal_id.data.sample import Box, Sample, Source, normalised_xyxy

LICENSE = "CC BY-SA 4.0"
SPECIES = {"1": "cat", "2": "dog"}
BACKGROUND = 2  # Trimap values: 1 pet, 2 background, 3 pet boundary.


def load(data_dir: Path) -> Iterator[Sample]:
    root = data_dir / "oxford_pets/annotations"
    for line in (root / "list.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        name, _, species, _ = line.split()
        # The dataset's XML boxes are heads; the trimap covers the whole body.
        trimap = np.asarray(Image.open(root / f"trimaps/{name}.png"))
        # The largest blob: ~11% of trimaps carry stray specks that would
        # stretch a box over the whole frame.
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            (trimap != BACKGROUND).astype(np.uint8)
        )
        xyxy = None  # 14 trimaps are all background; those pets stay unlocated.
        if n > 1:
            x, y, w, h = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), :4]
            xyxy = normalised_xyxy(x, y, x + w, y + h, *trimap.shape[::-1])
        yield Sample(
            path=f"oxford_pets/images/{name}.jpg",
            source=Source.OXFORD_PETS,
            license=LICENSE,
            boxes=(Box(SPECIES[species], xyxy),),
        )
