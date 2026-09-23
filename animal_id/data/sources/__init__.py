"""Source adapters: each ``load(data_dir)`` yields the Samples of one dataset."""

import logging

from animal_id.common.constants import DATA_DIR, MANIFEST_DIR
from animal_id.data.sample import Sample, Source, read_manifest, write_manifest
from animal_id.data.sources import coco, dogfacenet, mpdd, oxford_pets, stanford_dogs

logger = logging.getLogger(__name__)

SOURCES = {
    Source.COCO: coco.load,
    Source.DOGFACENET: dogfacenet.load,
    Source.MPDD: mpdd.load,
    Source.OXFORD_PETS: oxford_pets.load,
    Source.STANFORD_DOGS: stanford_dogs.load,
}


def parse(name: Source) -> None:
    # A missing download globs to nothing; caching that would hide it on every load.
    samples = list(SOURCES[name](DATA_DIR))
    if not samples:
        raise FileNotFoundError(f"No samples for '{name}' under {DATA_DIR}")
    path = MANIFEST_DIR / f"{name}.jsonl"
    write_manifest(samples, path)
    logger.info(f"Parsed {len(samples)} {name} samples into {path}")


def load(name: Source) -> list[Sample]:
    """A source's Samples, parsing it first if it has no manifest yet."""
    path = MANIFEST_DIR / f"{name}.jsonl"
    if not path.exists():
        parse(name)
    return read_manifest(path)
