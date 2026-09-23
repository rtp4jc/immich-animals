import pytest
from PIL import Image

from animal_id.common.license import LicenseTier, license_tier
from animal_id.data import sources, visualize
from animal_id.data.export import identity_splits
from animal_id.data.sample import Box, Sample, read_manifest, write_manifest
from animal_id.data.sources import dogfacenet


def _identity_samples(source, num_identities=20, per_identity=6):
    return [
        Sample(
            f"{source}/{i}/{j}.jpg", source, "CC BY 4.0", (Box("dog", identity=str(i)),)
        )
        for i in range(num_identities)
        for j in range(per_identity)
    ]


def test_manifest_round_trips(tmp_path):
    samples = [
        Sample("a.jpg", "src", "CC BY 4.0", (Box("dog", (0.1, 0.2, 0.5, 0.9), "rex"),)),
        Sample("b.jpg", "src", "unknown", (Box("cat"),)),
        Sample("c.jpg", "src", "CC0"),
    ]
    write_manifest(samples, tmp_path / "m.jsonl")
    assert read_manifest(tmp_path / "m.jsonl") == samples


def test_box_rejects_pixel_coordinates():
    with pytest.raises(ValueError):
        Box("dog", (10, 10, 50, 50))


def test_dogfacenet_identity_is_the_folder(tmp_path):
    root = tmp_path / dogfacenet.ROOT
    for dog in ("7", "8"):
        (root / dog).mkdir(parents=True)
        (root / dog / f"{dog}.0.jpg").touch()
    samples = list(dogfacenet.load(tmp_path))
    assert [(s.path, s.boxes[0].identity) for s in samples] == [
        (f"{dogfacenet.ROOT}/7/7.0.jpg", "7"),
        (f"{dogfacenet.ROOT}/8/8.0.jpg", "8"),
    ]
    assert samples[0].boxes[0].xyxy is None


def test_identity_splits_are_disjoint_and_complete():
    splits = identity_splits(_identity_samples("a"))
    ids = {k: {r["identity_label"] for r in rows} for k, rows in splits.items()}
    assert all(ids.values())
    assert ids["train"].isdisjoint(ids["val"] | ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert len(ids["train"] | ids["val"] | ids["test"]) == 20


def test_identity_splits_are_reproducible():
    samples = _identity_samples("a")
    assert identity_splits(samples) == identity_splits(samples)


def test_same_identity_name_in_two_sources_stays_distinct():
    splits = identity_splits(_identity_samples("a") + _identity_samples("b"))
    labels = {r["identity_label"] for rows in splits.values() for r in rows}
    assert len(labels) == 40


def test_identity_splits_drop_identities_below_min_images():
    splits = identity_splits(_identity_samples("a", per_identity=4))
    assert not any(splits.values())


@pytest.mark.parametrize(
    ("license", "tier"),
    [
        ("CC BY 4.0", LicenseTier.PERMISSIVE),
        ("CC-BY-2.0", LicenseTier.PERMISSIVE),
        ("CC0", LicenseTier.PERMISSIVE),
        ("CC BY-NC 4.0", LicenseTier.NONCOMMERCIAL),
        ("CC BY-SA 4.0", LicenseTier.ENCUMBERED),
        ("unknown", LicenseTier.ENCUMBERED),
    ],
)
def test_license_tier(license, tier):
    assert license_tier(license) is tier


def test_contact_sheet_has_one_band_per_source(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "DATA_DIR", tmp_path)
    Image.new("RGB", (300, 200)).save(tmp_path / "x.jpg")
    samples = [
        Sample("x.jpg", "boxes", "CC0", (Box("dog", (0.1, 0.1, 0.9, 0.9)),)),
        Sample("x.jpg", "boxes", "CC0"),
        Sample("x.jpg", "ids", "unknown", (Box("dog", identity="rex"),)),
        Sample("x.jpg", "ids", "unknown", (Box("dog", identity="fido"),)),
    ]
    sheet = visualize.contact_sheet(samples, rows_per_source=2, cols=3)
    # "boxes": 2 samples fill one row; "ids": one row per identity.
    assert sheet.size == (3 * visualize.TILE, 2 * 20 + 3 * visualize.TILE)
    assert [r["permissive"] for r in visualize.summary(samples)] == [2, 0]


def test_sample_rejects_two_unlocated_boxes():
    with pytest.raises(ValueError):
        Sample(
            "a.jpg", "src", "CC0", (Box("dog", identity="a"), Box("dog", identity="b"))
        )


def test_parse_refuses_to_cache_an_empty_source(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sources, "MANIFEST_DIR", tmp_path / "manifests")
    with pytest.raises(FileNotFoundError):
        sources.parse("dogfacenet")
    assert not (tmp_path / "manifests").exists()
