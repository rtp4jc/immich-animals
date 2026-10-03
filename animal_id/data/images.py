"""Image helpers shared by source adapters and exports."""

from PIL import Image

# Decoding multi-megapixel JPEGs every epoch starved detector training of CPU.
MAX_SIDE = 1280


def shrink(image: Image.Image) -> Image.Image:
    """RGB, longest side at most MAX_SIDE; the JPEG is decoded at reduced size."""
    image.draft("RGB", (MAX_SIDE, MAX_SIDE))
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image
