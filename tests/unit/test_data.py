import json

import numpy as np
import pytest
import yaml
from PIL import Image

from animal_id.common.license import LicenseTier, license_tier
from animal_id.data import dedupe, sources, visualize
from animal_id.data.exports import torch_identity, yolo
from animal_id.data.sample import Box, Sample, read_manifest, write_manifest
from animal_id.data.sources import (
    cat_individuals,
    coco,
    dogfacenet,
    dogreid,
    mpdd,
    open_images,
    oxford_pets,
    stanford_dogs,
)


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


def test_mpdd_identity_is_the_filename_prefix(tmp_path):
    for split, name in (("train", "12_c1s1_0.jpg"), ("gallery", "3_c2s4_1.jpg")):
        (tmp_path / mpdd.ROOT / split).mkdir(parents=True)
        (tmp_path / mpdd.ROOT / split / name).touch()
    samples = list(mpdd.load(tmp_path))
    assert sorted(s.boxes[0].identity for s in samples) == ["12", "3"]
    assert all(s.boxes[0].xyxy is None for s in samples)


def test_dogreid_loads_only_open_set_train_frames(tmp_path):
    root = tmp_path / dogreid.ROOT
    (root / "images/rex").mkdir(parents=True)
    (root / "splits.tab").write_text(
        "DOG_ID,VIDEO_ID,GROUP,SPLIT_CLOSED_SET,SPLIT_OPEN_SET\n"
        "rex,v1,0,train,train\nrex,v2,0,gallery,query\nfido,v3,1,query,gallery\n"
    )
    (root / "bounding_boxes.tab").write_text(
        "DOG_ID,VIDEO_ID,x_top_left,y_top_left,width,height\n"
        "rex,v1,50,20,100,60\nrex,v2,0,0,10,10\nfido,v3,0,0,10,10\n"
    )
    Image.new("RGB", (200, 100)).save(root / "images/rex/rex-v1.jpg")
    (sample,) = dogreid.load(tmp_path)
    assert sample.path == "dogreid/images/rex/rex-v1.jpg"
    assert sample.boxes == (Box("dog", (0.25, 0.2, 0.75, 0.8), "rex"),)


def test_open_images_keeps_only_whole_photo_labelled_upright_images(tmp_path):
    root = tmp_path / open_images.ROOT
    (root / "images/train").mkdir(parents=True)
    (root / "boxes.csv").write_text(
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf,IsDepiction\n"
        "a,/m/01yrx,0.1,0.5,0.2,0.6,0,0\n"
        "a,/m/0bt9lr,0.5,1.0,0.0,1.0,0,0\n"
        "b,/m/0bt9lr,0.1,0.5,0.2,0.6,0,0\n"
        "b,/m/01yrx,0.5,1.0,0.0,1.0,0,1\n"
        "c,/m/0bt9lr,0.0,1.0,0.0,1.0,1,0\n"
        "d,/m/0bt9lr,0.0,1.0,0.0,1.0,0,0\n"
        "e,/m/0bt9lr,0.0,1.0,0.0,1.0,0,0\n"
    )
    cc_by = "https://creativecommons.org/licenses/by/2.0/"
    (root / "images.csv").write_text(
        "ImageID,Subset,License,Rotation\n"
        + "".join(f"{i},train,{cc_by},0.0\n" for i in "abce")
        + f"d,train,{cc_by},90.0\n"
    )
    for i in "abcd":  # e was never downloaded
        (root / f"images/train/{i}.jpg").touch()
    (sample,) = open_images.load(tmp_path)
    assert sample.path == "open_images/images/train/a.jpg"
    assert sample.license == "CC BY 2.0"
    assert sample.boxes == (
        Box("cat", (0.1, 0.2, 0.5, 0.6)),
        Box("dog", (0.5, 0, 1, 1)),
    )


def test_boxed_identities_are_exported_as_padded_crops(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_identity, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(torch_identity, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data/src").mkdir(parents=True)
    samples = []
    for i in range(20):
        Image.new("RGB", (200, 100)).save(tmp_path / f"data/src/{i}.jpg")
        box = Box("dog", (0.25, 0.2, 0.75, 0.8), identity=str(i // 5))
        samples.append(Sample(f"src/{i}.jpg", "src", "CC0", (box,)))
    paths = {s: tmp_path / f"{s}.json" for s in ("train", "val", "test")}
    torch_identity.write(samples, paths)
    rows = [r for p in paths.values() for r in json.loads(p.read_text())]
    assert len(rows) == 20
    for row in rows:
        assert row["file_path"].startswith(f"data/{torch_identity.CROP_DIR}/")
        # 100px-wide box padded by 10% of its width on every side.
        assert Image.open(tmp_path / row["file_path"]).size == (120, 80)


def test_identity_splits_are_disjoint_and_complete():
    splits = torch_identity.splits(_identity_samples("a"))
    ids = {k: {r["identity_label"] for r in rows} for k, rows in splits.items()}
    assert all(ids.values())
    assert ids["train"].isdisjoint(ids["val"] | ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert len(ids["train"] | ids["val"] | ids["test"]) == 20


def test_identity_splits_are_reproducible():
    samples = _identity_samples("a")
    assert torch_identity.splits(samples) == torch_identity.splits(samples)


def test_same_identity_name_in_two_sources_stays_distinct():
    splits = torch_identity.splits(_identity_samples("a") + _identity_samples("b"))
    labels = {r["identity_label"] for rows in splits.values() for r in rows}
    assert len(labels) == 40


def test_adding_a_source_leaves_each_sources_split_unchanged():
    by_source = {"a": _identity_samples("a"), "b": _identity_samples("b", 30)}
    both = torch_identity.splits(by_source["a"] + by_source["b"])
    for source, samples in by_source.items():
        for split, rows in torch_identity.splits(samples).items():
            assert {r["file_path"] for r in rows} == {
                r["file_path"] for r in both[split] if f"/{source}/" in r["file_path"]
            }


def test_identity_splits_drop_identities_below_min_images():
    splits = torch_identity.splits(_identity_samples("a", per_identity=4))
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


def test_coco_keeps_animals_clamps_boxes_and_maps_licenses(tmp_path):
    ann = tmp_path / "coco/annotations"
    ann.mkdir(parents=True)
    for split in ("train2017", "val2017"):
        (ann / f"instances_{split}.json").write_text(
            json.dumps(
                {
                    "categories": [
                        {"id": 18, "name": "dog"},
                        {"id": 1, "name": "person"},
                    ],
                    "licenses": [
                        {
                            "id": 2,
                            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                        },
                        {"id": 4, "url": "http://creativecommons.org/licenses/by/2.0/"},
                    ],
                    "images": [
                        {
                            "id": 1,
                            "file_name": "a.jpg",
                            "width": 100,
                            "height": 50,
                            "license": 4,
                        },
                        {
                            "id": 2,
                            "file_name": "b.jpg",
                            "width": 100,
                            "height": 50,
                            "license": 2,
                        },
                    ],
                    "annotations": [
                        {"image_id": 1, "category_id": 18, "bbox": [-10, 5, 60, 100]},
                        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
                    ],
                }
            )
        )
    a, b = list(coco.load(tmp_path))[:2]
    assert a.path == "coco/images/train2017/a.jpg" and a.license == "CC BY 2.0"
    assert a.boxes == (Box("dog", (0.0, 0.1, 0.5, 1.0)),)
    assert b.boxes == () and b.license == "CC BY-NC 2.0"


def test_stanford_dogs_yields_every_dog_box(tmp_path):
    breed = tmp_path / "stanford_dogs/annotation/n1-Pug"
    breed.mkdir(parents=True)
    obj = "<object><bndbox><xmin>{}</xmin><ymin>0</ymin><xmax>{}</xmax><ymax>50</ymax></bndbox></object>"
    (breed / "n1_7").write_text(
        f"<annotation><size><width>100</width><height>50</height></size>"
        f"{obj.format(0, 50)}{obj.format(50, 100)}</annotation>"
    )
    (sample,) = stanford_dogs.load(tmp_path)
    assert sample.path == "stanford_dogs/images/n1-Pug/n1_7.jpg"
    assert [b.xyxy for b in sample.boxes] == [(0, 0, 0.5, 1), (0.5, 0, 1, 1)]


def test_cat_individuals_holds_out_benchmark_cats_and_ambiguous_photos(tmp_path):
    box = {"xyxy": [0.1, 0.2, 0.5, 0.6], "cats": 1}
    boxes = {f"images/{c}/{c}_000.jpg": box for c in ("0001", "0005")}
    boxes["images/0001/0001_001.jpg"] = None  # no cat found
    boxes["images/0001/0001_002.jpg"] = {**box, "cats": 2}  # which one is 0001?
    (tmp_path / "cat_individuals").mkdir()
    (tmp_path / "cat_individuals/boxes.json").write_text(json.dumps(boxes))
    assert cat_individuals.is_benchmark("0005")
    assert [s.boxes for s in cat_individuals.load(tmp_path)] == [
        (Box("cat", (0.1, 0.2, 0.5, 0.6), identity="0001"),)
    ]


def test_oxford_pets_boxes_the_largest_trimap_blob(tmp_path):
    root = tmp_path / "oxford_pets/annotations"
    (root / "trimaps").mkdir(parents=True)
    (root / "list.txt").write_text(
        "#Image CLASS-ID SPECIES BREED ID\nAbyssinian_1 1 1 1\nbeagle_1 13 2 2\n"
    )
    cat = np.full((10, 20), 2, np.uint8)  # 2 is background
    cat[2:6, 4:12] = 1
    cat[9, 19] = 3  # a stray speck must not stretch the box
    Image.fromarray(cat).save(root / "trimaps/Abyssinian_1.png")
    Image.fromarray(np.full((10, 20), 2, np.uint8)).save(root / "trimaps/beagle_1.png")
    assert [s.boxes for s in oxford_pets.load(tmp_path)] == [
        (Box("cat", (0.2, 0.2, 0.6, 0.6)),),
        (Box("dog"),),  # an all-background trimap leaves the pet unlocated
    ]


def _yolo_labels(tmp_path, samples, **kwargs):
    for s in samples:
        (tmp_path / s.path).parent.mkdir(parents=True, exist_ok=True)
    yolo.write(samples, ("dog", "cat"), tmp_path / "det/d.yaml", **kwargs)
    listed = (tmp_path / "det/train.txt").read_text().split() + (
        tmp_path / "det/val.txt"
    ).read_text().split()
    labels = {}
    for image in listed:
        label = image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        labels[image.removeprefix(f"{tmp_path}/")] = open(label).read()
    return labels


def test_yolo_labels_are_centre_format_per_class(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo, "DATA_DIR", tmp_path)
    samples = [
        Sample(
            "s/images/a.jpg",
            "s",
            "CC0",
            (Box("cat", (0.1, 0.2, 0.3, 0.6)), Box("bird", (0, 0, 1, 1))),
        ),
        Sample("s/images/b.jpg", "s", "CC0"),
    ]
    assert _yolo_labels(tmp_path, samples) == {
        "s/images/a.jpg": "1 0.200000 0.400000 0.200000 0.400000\n",
        "s/images/b.jpg": "",
    }
    config = yaml.safe_load((tmp_path / "det/d.yaml").read_text())
    assert config["names"] == ["dog", "cat"]


def test_yolo_skips_images_with_an_unlocated_target(tmp_path, monkeypatch):
    """Oxford dogs have no body box: labelling them empty would teach "no dog"."""
    monkeypatch.setattr(yolo, "DATA_DIR", tmp_path)
    samples = [Sample("s/images/a.jpg", "s", "CC0", (Box("dog"),))]
    assert _yolo_labels(tmp_path, samples) == {}


def test_yolo_caps_negatives_per_source(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo, "DATA_DIR", tmp_path)
    samples = [
        Sample(f"{src}/images/{i}.jpg", src, "CC0")
        for src in ("a", "b")
        for i in range(5)
    ]
    labels = _yolo_labels(tmp_path, samples, max_negatives={"a": 2})
    assert sum(p.startswith("a/") for p in labels) == 2
    assert sum(p.startswith("b/") for p in labels) == 5


def test_yolo_repeats_a_source_in_train_only(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo, "DATA_DIR", tmp_path)
    box = (Box("dog", (0.1, 0.1, 0.9, 0.9)),)
    samples = [
        Sample(f"{src}/images/{i}.jpg", src, "CC0", box)
        for src in ("a", "b")
        for i in range(40)
    ]
    _yolo_labels(tmp_path, samples, repeats={"a": 3})
    train = (tmp_path / "det/train.txt").read_text().split()
    val = (tmp_path / "det/val.txt").read_text().split()
    in_train = {s: {p for p in train if f"/{s}/" in p} for s in ("a", "b")}
    assert sum("/a/" in p for p in train) == 3 * len(in_train["a"])
    assert sum("/b/" in p for p in train) == len(in_train["b"])
    assert len(val) == len(set(val))


def test_drop_bursts_keeps_one_photo_per_burst_of_each_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(dedupe, "MANIFEST_DIR", tmp_path)
    samples = [
        Sample(f"{name}.jpg", "s", "CC0", (Box("cat", identity=cat),))
        for name, cat in [("a1", "a"), ("a2", "a"), ("a3", "a"), ("b1", "b")]
    ]
    # a1/a2 are one burst; a3 is a different shot of a; b1 looks like a1 but is b.
    vectors = {"a1": [1, 0], "a2": [0.99, 0.14], "a3": [0, 1], "b1": [1, 0]}
    monkeypatch.setattr(
        dedupe,
        "_embed",
        lambda group: np.array([vectors[s.path[:2]] for s in group], np.float32),
    )
    kept = [s.path for s in dedupe.drop_bursts(samples, ("s",))]
    assert kept == ["a1.jpg", "a3.jpg", "b1.jpg"]
    assert (tmp_path / "s.bursts.json").exists()
