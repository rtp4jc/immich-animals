"""Uniform check that any source parsed correctly: a contact sheet and a per-source summary.

contact_sheet(sources.load("dogfacenet")).save("outputs/data_inspect/dogfacenet.png")
"""

import random
from collections import defaultdict

from PIL import Image, ImageDraw

from animal_id.common.constants import DATA_DIR
from animal_id.common.license import LicenseTier, license_tier
from animal_id.data.sample import Sample

TILE = 160


def summary(samples: list[Sample]) -> list[dict]:
    by_source = defaultdict(list)
    for s in samples:
        by_source[s.source].append(s)
    rows = []
    for source, group in sorted(by_source.items()):
        boxes = [b for s in group for b in s.boxes]
        rows.append(
            {
                "source": source,
                "images": len(group),
                "negatives": sum(not s.boxes for s in group),
                "boxes": sum(b.xyxy is not None for b in boxes),
                "identities": len({b.identity for b in boxes} - {None}),
                "permissive": sum(
                    license_tier(s.license) is LicenseTier.PERMISSIVE for s in group
                ),
            }
        )
    return rows


def _rows(samples: list[Sample], rows: int, cols: int, rng: random.Random):
    """``rows`` rows of ``cols`` samples; one identity per row when the source has them."""
    by_identity = defaultdict(list)
    for s in samples:
        by_identity[next((b.identity for b in s.boxes if b.identity), None)].append(s)
    by_identity.pop(None, None)
    if by_identity:
        picked = rng.sample(sorted(by_identity), min(rows, len(by_identity)))
        return [
            rng.sample(by_identity[k], min(cols, len(by_identity[k]))) for k in picked
        ]
    flat = rng.sample(samples, min(rows * cols, len(samples)))
    return [flat[i : i + cols] for i in range(0, len(flat), cols)]


def _tile(sample: Sample) -> Image.Image:
    image = Image.open(DATA_DIR / sample.path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    width = max(2, w // 150)
    for box in sample.boxes:
        if box.xyxy is not None:
            x1, y1, x2, y2 = box.xyxy
            draw.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline="red", width=width)
    image.thumbnail((TILE, TILE))
    tile = Image.new("RGB", (TILE, TILE), "white")
    tile.paste(image, ((TILE - image.width) // 2, (TILE - image.height) // 2))
    caption = " ".join(b.identity or b.label for b in sample.boxes) or "negative"
    ImageDraw.Draw(tile).text(
        (3, 3), caption, fill="yellow", stroke_width=1, stroke_fill="black"
    )
    return tile


def contact_sheet(
    samples: list[Sample], rows_per_source: int = 4, cols: int = 6, seed: int = 0
) -> Image.Image:
    """One band of rows per source, boxes drawn and each tile captioned with its label."""
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for s in samples:
        by_source[s.source].append(s)
    bands = [
        (source, _rows(group, rows_per_source, cols, rng))
        for source, group in sorted(by_source.items())
    ]
    header = 20
    height = sum(header + len(rows) * TILE for _, rows in bands)
    sheet = Image.new("RGB", (cols * TILE, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for source, rows in bands:
        draw.text((4, y + 4), source, fill="black")
        y += header
        for row in rows:
            for x, sample in enumerate(row):
                sheet.paste(_tile(sample), (x * TILE, y))
            y += TILE
    return sheet
