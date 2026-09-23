"""Identity splits as JSON rows for the PyTorch ``IdentityDataset`` (embedding training).

torch_identity.write(sources.load(Source.DOGFACENET), {"train": ..., "val": ..., "test": ...})
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from animal_id.common.constants import DATA_DIR, PROJECT_ROOT
from animal_id.data.sample import Sample

logger = logging.getLogger(__name__)


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
        for box in sample.boxes:
            if box.identity is None:
                continue
            if box.xyxy is not None:
                raise ValueError(f"IdentityDataset cannot crop boxes yet: {sample}")
            paths_by_identity[(sample.source, box.identity)].append(sample.path)

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
            for p in paths_by_identity[identity]
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
    """Writes each split's rows to ``paths[split]``."""
    for split, rows in splits(samples).items():
        paths[split].write_text(json.dumps(rows, indent=2))
        logger.info(f"Wrote {len(rows)} {split} rows to {paths[split]}")
