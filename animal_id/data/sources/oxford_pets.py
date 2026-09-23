"""Oxford-IIIT Pets: one cat or dog per photo, labelled by species."""

from collections.abc import Iterator
from pathlib import Path

from animal_id.data.sample import Box, Sample, Source

LICENSE = "CC BY-SA 4.0"
SPECIES = {"1": "cat", "2": "dog"}


def load(data_dir: Path) -> Iterator[Sample]:
    lines = (data_dir / "oxford_pets/annotations/list.txt").read_text().splitlines()
    for line in lines:
        if line.startswith("#"):
            continue
        name, _, species, _ = line.split()
        yield Sample(
            path=f"oxford_pets/images/{name}.jpg",
            source=Source.OXFORD_PETS,
            license=LICENSE,
            # The dataset's boxes are heads, not bodies, so the animal is unlocated.
            boxes=(Box(SPECIES[species]),),
        )
