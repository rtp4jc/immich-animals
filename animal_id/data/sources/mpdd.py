"""Multi-pose Dog Dataset: whole-body street-dog crops, identity in the filename."""

import hashlib
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import requests

from animal_id.data.sample import Box, Sample, Source

ROOT = "mpdd/MPDD/pytorch"
LICENSE = "CC BY 4.0"  # doi:10.17632/v5j6m8dzhv.1
URL = "https://data.mendeley.com/public-files/datasets/v5j6m8dzhv/files/05d1d583-faf6-410d-89a5-a6b1134b6e5e/file_downloaded"
SHA256 = "6c800c1b4aa67629544dec7444dee85bc57781abde1ae1d077ea9ef1804284cd"


def fetch(data_dir: Path) -> None:
    """Download and extract MPDD.zip (30 MB) unless it is already there."""
    if (data_dir / ROOT).exists():
        return
    body = requests.get(URL, timeout=300).content
    if hashlib.sha256(body).hexdigest() != SHA256:
        raise ValueError(f"MPDD.zip from {URL} does not match its published SHA-256")
    zipfile.ZipFile(io.BytesIO(body)).extractall(data_dir / "mpdd")


def load(data_dir: Path) -> Iterator[Sample]:
    # The reID train/val/query/gallery folders are ignored: identity splits are
    # made at export, the same way for every source.
    for image in sorted((data_dir / ROOT).glob("*/*.jpg")):
        yield Sample(
            path=image.relative_to(data_dir).as_posix(),
            source=Source.MPDD,
            license=LICENSE,
            # Already cropped to the dog, and the crop is its only location.
            boxes=(Box("dog", identity=image.name.split("_")[0]),),
        )
