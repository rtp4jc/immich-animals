"""The dataset contract: every source is parsed into Samples, stored as JSON-lines manifests.

samples = read_manifest(MANIFEST_DIR / "dogfacenet.jsonl")
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class Source(StrEnum):
    """Every dataset with an adapter or a written manifest; the value names its manifest."""

    DOGFACENET = "dogfacenet"


@dataclass(frozen=True)
class Box:
    """One animal. If identity is None, the box has no ID. If xyxy is None,
    there may be and identity, but no bounding box."""

    label: str
    xyxy: tuple[float, float, float, float] | None = None
    identity: str | None = None

    def __post_init__(self):
        # Normalised so exports never need the image size; pixel coords are the
        # adapter bug this catches.
        if self.xyxy is not None:
            x1, y1, x2, y2 = self.xyxy
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError(f"xyxy must be normalised to [0, 1]: {self.xyxy}")


@dataclass(frozen=True)
class Sample:
    """One image, ``path`` relative to DATA_DIR; empty ``boxes`` means no animal."""

    path: str
    source: str
    license: str
    boxes: tuple[Box, ...] = ()

    def __post_init__(self):
        missing_xyxy = 0
        for box in self.boxes:
            if box.xyxy is None:
                if missing_xyxy > 0:
                    raise ValueError(
                        "Multiple identities in one sample without a bounding box. Identity is ambiguous."
                    )
                missing_xyxy += 1


def write_manifest(samples: Iterable[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample)) + "\n")


def read_manifest(path: Path) -> list[Sample]:
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            boxes = tuple(
                Box(b["label"], b["xyxy"] and tuple(b["xyxy"]), b["identity"])
                for b in row.pop("boxes")
            )
            samples.append(Sample(**row, boxes=boxes))
    return samples
