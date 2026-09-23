# Data contract: one Sample shape for every source (2026-09-22)

Goal: grow the training data from permissively licensed public sources without
each new dataset adding a bespoke converter. Today every source is parsed inside
a trainer-specific converter (`CocoDetectorDatasetConverter`,
`EmbeddingDatasetConverter`), so adding one means touching training code.

## The contract

```
source files ──adapter──▶ Sample stream ──manifest──▶ data/manifests/<source>.jsonl
                                                          │
                     ┌────────────────────┬───────────────┴────────┐
                 export: identity     export: YOLO           inspect: contact
                 splits (embedding)   (detection, phase 2)   sheet + summary
```

- `Box(label, xyxy | None, identity | None)`: one animal. `xyxy` is normalised
  to [0, 1] so exports never need the image size and resized copies of a source
  keep valid boxes. `None` means the source names the animal but not where it
  is (DogFaceNet face crops; later, identity folders the detector will box).
- `Sample(path, source, license, boxes)`: one image. `path` is relative to
  `DATA_DIR` (portable across machines). Empty `boxes` = confirmed no animal.
  `license` is the string the source states; `license_tier()` maps it onto the
  `LicenseTier` enum the backbones already use.
- Identities are namespaced at export as `<source>/<identity>`, so two sources
  can both have a dog called `12`.
- No pixels in the record. Images are decoded in `Dataset.__getitem__` inside
  DataLoader workers (pytorch#13246: Python objects in forked workers get
  copied page by page).
- A manifest (JSON lines, one Sample each) is the unit of data. Third-party
  datasets get one from an adapter; data we harvest ourselves (Commons, Flickr)
  is *written* as a manifest by its fetch script and needs no adapter.

Why not FiftyOne / HF `datasets` as the contract: both model exactly this
(filepath + labels, lazy media), but FiftyOne needs mongod to prepare training
data and `datasets` adds Arrow for ~10K rows. Plain frozen dataclasses map 1:1
onto `fo.Sample` if we want the App later.

## Phases

1. **Contract + embedding.** `animal_id/data/` with `Sample`/`Box`, manifests,
   DogFaceNet adapter, identity-split export (replaces
   `EmbeddingDatasetConverter`, same split for the same input), contact sheet +
   per-source summary (replaces script 07). `scripts/data.py parse|inspect`.
2. **Detection.** Adapters for COCO (per-image Flickr licence from the JSON),
   Stanford Dogs, Oxford Pets; boxes for every animal class, with the class list
   chosen at export. YOLO export straight from normalised boxes; delete
   `CocoDetectorDatasetConverter` and the COCO→YOLO detection converter.
3. **First permissive sources.** Open Images V7 (detection, multi-class),
   Commons identity categories and MPDD (identity, whole-body). Needs crop
   support in `IdentityDataset` matching the sidecar's 10% padding.
4. **Harvested identity data.** YFCC100M CC-BY by owner + pet-name tag;
   dedupe (perceptual hash) across sources before any split.

Held-out evaluation sources are chosen per export, never mixed into train.
Sub-center ArcFace is re-evaluated once phase 3–4 data lands (no gain on
DogFaceNet alone).

## Licence status of current sources

| Source | Stated licence | Tier |
| --- | --- | --- |
| DogFaceNet | CC BY 4.0 (Zenodo record; images originally from the web) | permissive |
| Stanford Dogs | ImageNet terms, non-commercial | not permissive |
| COCO 2017 | per image: Flickr CC variants incl. NC; annotations CC BY 4.0 | filter per image |
| Oxford-IIIT Pets | CC BY-SA 4.0 | policy decision |
| MPDD | CC BY 4.0 | permissive |
