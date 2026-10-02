"""Cat Individual Images: Taipei shelter cats on phones, one folder per cat, boxed by YOLO11x.

uv run python scripts/data.py prepare cat_individuals   # after unzipping the download
"""

import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from animal_id.data.sample import Box, Sample, Source

ROOT = "cat_individuals"
LICENSE = "CC BY 4.0"  # kaggle.com/datasets/timost1234/cat-individuals
ORIGINALS = "cat_individuals_dataset"
# The dataset has no boxes; these are COCO YOLO11x's largest cat per photo.
BOXES = "boxes.json"
COCO_CAT = 15
MAX_SIDE = 1280
BENCHMARK_FRACTION = 0.25


def is_benchmark(cat: str) -> bool:
    """Held-out cats, like DogReID's query/gallery: out of every training export."""
    digest = int(hashlib.sha1(cat.encode()).hexdigest()[:8], 16)
    return digest / 16**8 < BENCHMARK_FRACTION


def load(data_dir: Path, benchmark: bool = False) -> Iterator[Sample]:
    boxes = json.loads((data_dir / ROOT / BOXES).read_text())
    for rel, box in sorted(boxes.items()):
        # An unboxed photo is no confirmed negative, and two cats leave the
        # identity ambiguous.
        cat = Path(rel).parent.name
        if box is None or box["cats"] > 1 or is_benchmark(cat) != benchmark:
            continue
        yield Sample(
            path=f"{ROOT}/{rel}",
            source=Source.CAT_INDIVIDUALS,
            license=LICENSE,
            boxes=(Box("cat", tuple(box["xyxy"]), identity=cat),),
        )


def prepare(data_dir: Path, workers: int = 8) -> None:
    """Writes 1280px copies under images/ and YOLO11x boxes for them."""
    from ultralytics import YOLO

    root = data_dir / ROOT
    originals = sorted(
        p
        for p in (root / ORIGINALS).glob("*/*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    # Decoding the 16MP originals every epoch starved detector training of CPU.
    with ProcessPoolExecutor(workers) as pool:
        resized = list(pool.map(_resize, originals, [root] * len(originals)))
    model = YOLO("yolo11x.pt")
    boxes = {}
    for start in range(0, len(resized), 32):
        batch = resized[start : start + 32]
        results = model.predict(
            [str(root / p) for p in batch], classes=[COCO_CAT], conf=0.25, verbose=False
        )
        for rel, result in zip(batch, results, strict=True):
            found = result.boxes
            if not len(found):
                boxes[rel] = None
                continue
            xyxyn = found.xyxyn.cpu().numpy()
            area = (xyxyn[:, 2] - xyxyn[:, 0]) * (xyxyn[:, 3] - xyxyn[:, 1])
            boxes[rel] = {
                "xyxy": [round(float(v), 6) for v in xyxyn[area.argmax()]],
                "cats": int((found.conf >= 0.5).sum()),
            }
    (root / BOXES).write_text(json.dumps(boxes))


def _resize(original: Path, root: Path) -> str:
    rel = (Path("images") / original.relative_to(root / ORIGINALS)).with_suffix(".jpg")
    out = root / rel
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original) as image:
            image.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG decode at reduced size
            image = image.convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            image.save(out, quality=92)
    return rel.as_posix()
