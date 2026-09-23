"""DogFaceNet: aligned dog-face crops, one folder per dog."""

from collections.abc import Iterator
from pathlib import Path

from animal_id.data.sample import Box, Sample, Source

ROOT = "dogfacenet/DogFaceNet_224resized/after_4_bis"
# As declared on the dataset's Zenodo record (12578449); the repo's MIT covers code.
LICENSE = "CC BY 4.0"


def load(data_dir: Path) -> Iterator[Sample]:
    for image in sorted((data_dir / ROOT).glob("*/*.jpg")):
        yield Sample(
            path=image.relative_to(data_dir).as_posix(),
            source=Source.DOGFACENET,
            license=LICENSE,
            # The crop is a face, not a body box, so no xyxy: using the whole
            # image is the embedder's job, and it must not train the detector.
            boxes=(Box("dog", identity=image.parent.name),),
        )
