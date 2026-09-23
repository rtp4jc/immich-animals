"""Turns Samples into what each trainer reads.

splits = identity_splits(sources.load(Source.DOGFACENET))
"""

import logging
import random
from collections import defaultdict

from animal_id.common.constants import DATA_DIR, PROJECT_ROOT
from animal_id.data.sample import Sample

logger = logging.getLogger(__name__)


def identity_splits(
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
            paths_by_identity[f"{sample.source}/{box.identity}"].append(sample.path)

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

    total = sum(len(rows) for rows in rows_by_label)
    order = list(range(len(rows_by_label)))
    random.Random(seed).shuffle(order)

    # Whole identities fill test, then val, then train (open-set protocol).
    splits = {"train": [], "val": [], "test": []}
    for label in order:
        if len(splits["test"]) < int(total * test_ratio):
            splits["test"].extend(rows_by_label[label])
        elif len(splits["val"]) < int(total * val_ratio):
            splits["val"].extend(rows_by_label[label])
        else:
            splits["train"].extend(rows_by_label[label])
    return splits
