"""Open Images V7: Flickr photos with cat and dog boxes, fetched from the public bucket.

open_images.fetch(DATA_DIR, max_per_class=13000, workers=32)
"""

import csv
import io
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from animal_id.data.images import MAX_SIDE, shrink
from animal_id.data.sample import Box, Sample, Source, normalised_xyxy

logger = logging.getLogger(__name__)

ROOT = "open_images"
CLASSES = {"/m/01yrx": "cat", "/m/0bt9lr": "dog"}
CSV_URL = "https://storage.googleapis.com/openimages"
OFFICIAL_CSVS = {
    "train": (
        "v6/oidv6-train-annotations-bbox.csv",
        "2018_04/train/train-images-boxable-with-rotation.csv",
    ),
    "validation": (
        "v5/validation-annotations-bbox.csv",
        "2018_04/validation/validation-images-with-rotation.csv",
    ),
    "test": (
        "v5/test-annotations-bbox.csv",
        "2018_04/test/test-images-with-rotation.csv",
    ),
}
BOX_COLUMNS = "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf,IsDepiction".split(",")
IMAGE_COLUMNS = "ImageID,Subset,OriginalLandingURL,License,Author,Rotation".split(",")
XYXY = ("XMin", "YMin", "XMax", "YMax")
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"


def _license(url: str) -> str:
    """ "https://creativecommons.org/licenses/by/2.0/" -> "CC BY 2.0"."""
    match = re.search(r"licenses/([a-z-]+)/([\d.]+)", url)
    return f"CC {match[1].upper()} {match[2]}" if match else url


def _usable(root: Path) -> Iterator[tuple[dict, tuple[Box, ...]]]:
    """Each listed image whose boxes can train a photo detector, with those boxes."""
    rows = defaultdict(list)
    with open(root / "boxes.csv") as f:
        for row in csv.DictReader(f):
            rows[row["ImageID"]].append(row)
    with open(root / "images.csv") as f:
        images = list(csv.DictReader(f))
    for image in images:
        image_rows = rows[image["ImageID"]]
        # A group-of box covers several animals with one box, and a depiction is a
        # drawing or toy; neither should be learnt, and dropping just the box would
        # leave a real animal labelled as background.
        if any(r["IsGroupOf"] != "0" or r["IsDepiction"] != "0" for r in image_rows):
            continue
        # Boxes match the stored pixels, which for these are sideways; ~1% of images.
        if image["Rotation"] not in ("", "0.0"):
            continue
        corners = (
            (r["LabelName"], normalised_xyxy(*(float(r[k]) for k in XYXY), 1, 1))
            for r in image_rows
        )
        yield image, tuple(Box(CLASSES[label], xyxy) for label, xyxy in corners if xyxy)


def _path(image: dict) -> str:
    return f"{ROOT}/images/{image['Subset']}/{image['ImageID']}.jpg"


def load(data_dir: Path) -> Iterator[Sample]:
    for image, boxes in _usable(data_dir / ROOT):
        if (data_dir / _path(image)).exists():
            yield Sample(
                _path(image), Source.OPEN_IMAGES, _license(image["License"]), boxes
            )


def _download_csvs(csv_dir: Path) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name in (n for pair in OFFICIAL_CSVS.values() for n in pair):
        path = csv_dir / Path(name).name
        if not path.exists():
            logger.info(f"Downloading {name}")
            with requests.get(f"{CSV_URL}/{name}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path.with_suffix(".part"), "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            path.with_suffix(".part").rename(path)


def _filter_csvs(csv_dir: Path, root: Path) -> None:
    """Writes boxes.csv (our classes' rows) and images.csv (their images' metadata)."""
    ids = set()
    with open(root / "boxes.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, BOX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for boxes_csv, _ in OFFICIAL_CSVS.values():
            with open(csv_dir / Path(boxes_csv).name) as f:
                for row in csv.DictReader(f):
                    if row["LabelName"] in CLASSES:
                        writer.writerow(row)
                        ids.add(row["ImageID"])
    # images.csv marks the filter done, so an interrupted run must not leave it.
    partial = root / "images.csv.part"
    with open(partial, "w", newline="") as out:
        writer = csv.DictWriter(out, IMAGE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for _, images_csv in OFFICIAL_CSVS.values():
            with open(csv_dir / Path(images_csv).name) as f:
                for row in csv.DictReader(f):
                    if row["ImageID"] in ids:
                        writer.writerow(row)
    partial.rename(root / "images.csv")


def _fetch_image(session: requests.Session, url: str, path: Path) -> None:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    if max(image.size) > MAX_SIDE:
        shrink(image).save(path.with_suffix(".part"), "JPEG", quality=90)
    else:
        path.with_suffix(".part").write_bytes(response.content)
    path.with_suffix(".part").rename(path)


def fetch(data_dir: Path, max_per_class: int, workers: int) -> None:
    """Downloads up to ``max_per_class`` usable images per class; reruns resume."""
    root = data_dir / ROOT
    _download_csvs(root / "csv")
    if not (root / "images.csv").exists():
        _filter_csvs(root / "csv", root)
    # ImageIDs are random hex, so the lowest IDs are a fixed, unbiased subset.
    by_class = defaultdict(list)
    for image, boxes in sorted(_usable(root), key=lambda u: u[0]["ImageID"]):
        for label in {b.label for b in boxes}:
            by_class[label].append(image)
    chosen = {
        r["ImageID"]: r for group in by_class.values() for r in group[:max_per_class]
    }
    todo = [r for r in chosen.values() if not (data_dir / _path(r)).exists()]
    logger.info(f"{len(chosen)} images chosen, {len(todo)} to download")
    for split in OFFICIAL_CSVS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(
                _fetch_image,
                session,
                IMAGE_URL.format(split=r["Subset"], id=r["ImageID"]),
                data_dir / _path(r),
            )
            for r in todo
        ]
        failed = sum(f.exception() is not None for f in tqdm(futures))
    logger.info(f"{failed} downloads failed; rerun to retry")
