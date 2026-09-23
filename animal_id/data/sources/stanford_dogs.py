"""Stanford Dogs: ImageNet dog photos with a box per dog."""

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from animal_id.data.sample import Box, Sample, Source, normalised_xyxy

LICENSE = "ImageNet terms of access"


def load(data_dir: Path) -> Iterator[Sample]:
    for xml_path in sorted((data_dir / "stanford_dogs/annotation").glob("*/*")):
        root = ET.parse(xml_path).getroot()
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        boxes = []
        for obj in root.iter("object"):
            corners = (
                float(obj.findtext(f"bndbox/{k}"))
                for k in ("xmin", "ymin", "xmax", "ymax")
            )
            xyxy = normalised_xyxy(*corners, width, height)
            if xyxy:
                boxes.append(Box("dog", xyxy))
        yield Sample(
            path=f"stanford_dogs/images/{xml_path.parent.name}/{xml_path.name}.jpg",
            source=Source.STANFORD_DOGS,
            license=LICENSE,
            boxes=tuple(boxes),
        )
