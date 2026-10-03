"""Image helpers shared by source adapters and exports."""

from PIL import Image

# Padding by this fraction of the box width on every side matches the sidecar's
# BBOX_PAD: train on the crops we serve.
CROP_PAD = 0.1
# Decoding multi-megapixel JPEGs every epoch starved detector training of CPU.
MAX_SIDE = 1280


def shrink(image: Image.Image) -> Image.Image:
    """RGB, longest side at most MAX_SIDE; the JPEG is decoded at reduced size."""
    image.draft("RGB", (MAX_SIDE, MAX_SIDE))
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def crop(image: Image.Image, xyxy: list[float] | None) -> Image.Image:
    """The animal in RGB, padded like the sidecar's crop; the whole image if unlocated."""
    image = image.convert("RGB")
    if xyxy is None:
        return image
    w, h = image.size
    x1, y1, x2, y2 = (v * s for v, s in zip(xyxy, (w, h, w, h), strict=True))
    pad = (x2 - x1) * CROP_PAD
    box = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
    return image.crop(tuple(map(int, box)))
