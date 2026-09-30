"""Identity splits as JSON rows for the PyTorch ``IdentityDataset`` (embedding training).

Boxed identities are cropped to files at export, so every row is one animal's image.

torch_identity.write(sources.load(Source.DOGFACENET), {"train": ..., "val": ..., "test": ...})
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from animal_id.common.constants import DATA_DIR, PROJECT_ROOT
from animal_id.data.sample import Sample

logger = logging.getLogger(__name__)

CROP_DIR = "identity_crops"
# Padding by this fraction of the box width on every side matches the sidecar's
# BBOX_PAD: train on the crops we serve.
CROP_PAD = 0.1


def splits(
    samples: list[Sample],
    min_images: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Identity-disjoint train/val/test rows in the format ``IdentityDataset`` reads."""
    paths_by_identity = defaultdict(list)
    for sample in samples:
        for i, box in enumerate(sample.boxes):
            if box.identity is None:
                continue
            path, source_box = sample.path, None
            if box.xyxy is not None:
                source_box = {"image": sample.path, "xyxy": list(box.xyxy)}
                path = f"{CROP_DIR}/{Path(sample.path).with_suffix('')}_{i}.jpg"
            paths_by_identity[(sample.source, box.identity)].append((path, source_box))

    # Labels are assigned before the split, over every identity, so a label is
    # stable for a given input regardless of which split it lands in.
    kept = sorted(
        identity
        for identity, paths in paths_by_identity.items()
        if len(paths) >= min_images
    )
    dropped = paths_by_identity.keys() - set(kept)
    logger.info(
        f"Kept {len(kept)} identities with >= {min_images} images; dropped "
        f"{len(dropped)} ({sum(len(paths_by_identity[k]) for k in dropped)} images)"
    )
    data_prefix = DATA_DIR.relative_to(PROJECT_ROOT)
    rows_by_label = [
        [
            {"file_path": (data_prefix / p).as_posix(), "identity_label": label}
            | ({"crop": source_box} if source_box else {})
            for p, source_box in paths_by_identity[identity]
        ]
        for label, identity in enumerate(kept)
    ]

    # Each source is split on its own, so adding a source never moves another
    # source's identities between splits (and never changes its test set).
    splits = {"train": [], "val": [], "test": []}
    for source in sorted({source for source, _ in kept}):
        order = [label for label, (s, _) in enumerate(kept) if s == source]
        total = sum(len(rows_by_label[label]) for label in order)
        random.Random(seed).shuffle(order)

        # Whole identities fill test, then val, then train (open-set protocol).
        part = {"train": [], "val": [], "test": []}
        for label in order:
            if len(part["test"]) < int(total * test_ratio):
                part["test"].extend(rows_by_label[label])
            elif len(part["val"]) < int(total * val_ratio):
                part["val"].extend(rows_by_label[label])
            else:
                part["train"].extend(rows_by_label[label])
        for split, rows in part.items():
            splits[split].extend(rows)
    return splits


def write(samples: list[Sample], paths: dict[str, Path]) -> None:
    """Writes each split's rows to ``paths[split]``, cropping boxed identities first."""
    for split, rows in splits(samples).items():
        for row in rows:
            if "crop" not in row:
                continue
            out = PROJECT_ROOT / row["file_path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(DATA_DIR / row["crop"]["image"]) as image:
                w, h = image.size
                x1, y1, x2, y2 = (
                    v * s
                    for v, s in zip(row["crop"]["xyxy"], (w, h, w, h), strict=True)
                )
                pad = (x2 - x1) * CROP_PAD
                box = (
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(w, x2 + pad),
                    min(h, y2 + pad),
                )
                image.convert("RGB").crop(tuple(map(int, box))).save(out, quality=95)
        paths[split].write_text(json.dumps(rows, indent=2))
        logger.info(f"Wrote {len(rows)} {split} rows to {paths[split]}")
