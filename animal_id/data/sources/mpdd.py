"""Multi-pose Dog Dataset: whole-body street-dog crops, identity in the filename."""

from collections.abc import Iterator
from pathlib import Path

from animal_id.data.sample import Box, Sample, Source

ROOT = "mpdd/MPDD/pytorch"
LICENSE = "CC BY 4.0"  # doi:10.17632/v5j6m8dzhv.1


def load(data_dir: Path) -> Iterator[Sample]:
    # The reID train/val/query/gallery folders are ignored: identity splits are
    # made at export, the same way for every source.
    for image in sorted((data_dir / ROOT).glob("*/*.jpg")):
        yield Sample(
            path=image.relative_to(data_dir).as_posix(),
            source=Source.MPDD,
            license=LICENSE,
            # Already cropped to the dog, and the crop is its only location.
            boxes=(Box("dog", identity=image.name.split("_")[0]),),
        )
