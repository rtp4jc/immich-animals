"""Drop near-duplicate bursts within each identity before identity splits are made.

samples = dedupe.drop_bursts(samples, DATA_CONFIG.dedupe)
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image

from animal_id.common.constants import DATA_DIR, MANIFEST_DIR
from animal_id.common.datasets import eval_transform
from animal_id.common.embeddings import normalize_embeddings
from animal_id.data.images import crop
from animal_id.data.sample import Sample
from animal_id.embedding.backbones import BackboneType, get_backbone

logger = logging.getLogger(__name__)

# Frozen DINOv2-B cosine. Same-dog pairs sit at a median of 0.48; burst shots of
# one cat at 0.85, and keeping them made cat scores near-perfect and meaningless.
THRESHOLD = 0.85


def drop_bursts(samples: list[Sample], sources: tuple[str, ...]) -> list[Sample]:
    """Keeps the first photo of each burst of one identity in ``sources``."""
    by_source = defaultdict(list)
    for sample in samples:
        if sample.source in sources:
            by_source[sample.source].append(sample)
    dropped = set()
    for source, group in by_source.items():
        # Keyed by everything the decision depends on, so it can never go stale.
        key = json.dumps([THRESHOLD, *map(asdict, group)], sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        cache = MANIFEST_DIR / f"{source}.bursts-{digest}.json"
        if cache.exists():
            bursts = set(json.loads(cache.read_text()))
        else:
            bursts = _bursts(group)
            cache.write_text(json.dumps(sorted(bursts)))
        logger.info(
            f"{source}: dropping {len(bursts)} of {len(group)} as burst duplicates"
        )
        dropped |= bursts
    return [s for s in samples if s.path not in dropped]


def _bursts(samples: list[Sample]) -> set[str]:
    boxed = [s for s in samples if any(b.identity for b in s.boxes)]
    embeddings = _embed(boxed)
    by_identity = defaultdict(list)
    for i, sample in enumerate(boxed):
        identity = next(b.identity for b in sample.boxes if b.identity)
        by_identity[identity].append(i)
    dropped = set()
    for members in by_identity.values():
        similar = embeddings[members] @ embeddings[members].T
        left = list(range(len(members)))
        while left:
            keep = left.pop(0)
            burst = [j for j in left if similar[keep, j] >= THRESHOLD]
            dropped |= {boxed[members[j]].path for j in burst}
            left = [j for j in left if j not in burst]
    return dropped


@torch.no_grad()
def _embed(samples: list[Sample], batch_size: int = 64) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_backbone(BackboneType.DINOV2_B)[0].to(device).eval()
    transform = eval_transform()
    out = []
    for start in range(0, len(samples), batch_size):
        batch = []
        for sample in samples[start : start + batch_size]:
            box = next(b for b in sample.boxes if b.identity)
            with Image.open(DATA_DIR / sample.path) as image:
                batch.append(transform(crop(image, box.xyxy)))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            features = model(torch.stack(batch).to(device)).float()
        out.append(normalize_embeddings(features.cpu().numpy()))
    return np.concatenate(out)
