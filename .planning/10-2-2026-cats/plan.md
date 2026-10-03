# Cats (2026-10-02)

Goal: serve cats alongside dogs without regressing dogs (AGENTS.md tenet), and
decide between one shared embedder and one per species.

## Data

- **Cat Individual Images** (Kaggle, CC BY 4.0): 518 Taipei shelter cats on
  phones, no boxes. `data.py prepare` writes 1280px copies (decoding the 16MP
  originals made a detector epoch take 40 minutes) and boxes them with COCO
  YOLO11x; photos with no cat or with two confident cats are dropped. A quarter
  of the cats, chosen by id hash, are the held-out benchmark (123 cats, 2,550
  photos), like DogReID's query/gallery.
- The photos are shot in **bursts**: same-cat frozen DINOv2 cosine has a median
  of 0.85 (DogReID dogs: 0.48), so 2,550 benchmark photos are ~660 distinct
  shots and every model scored near-perfect. Cat benchmarks use one photo per
  burst; training drops bursts too (`dedupe.drop_bursts`, threshold 0.85).
- **Oxford Pets** is boxed from its trimaps (largest connected blob; ~11% of
  trimaps carry specks that would stretch the box over the frame).
- **Commons**: 50 named cats (1,040 photos) in `sidecar-validation`, every
  record tagged with `species`.
- **Open Images V7**: 12.6k cat + 12.7k dog photos (CC BY 2.0). Adding them for
  15 more epochs moved every detection number by less than noise and raised
  false positives 23.4% → 25.8%, so they are not in the default mix.

## Evaluation

Household simulation from the 0.2.0 work: Immich's face assignment
(`identification.cluster`), 300 homes per half of the identities, 1-4 own pets
with 3-15 photos plus 20-200 strangers. Plain DBSCAN chained dense cat
embeddings into one cluster (purity 0.08 at 0.4) that Immich's
nearest-person assignment keeps apart (0.81), so the clusterer now replays
Immich. Two dog-only retrains with different seeds differ by 1.5pp clean,
6pp never grouped and 2pp merged at 0.4: smaller gaps are noise.

## Results (DINOv2-B + ArcFace, dogs at 0.4)

| Embedder | Dogs clean / never grouped / merged | Dog F | Cats clean / merged (at 0.35) | Commons cat purity (0.35) |
|---|---|---|---|---|
| Shipped 0.2.0 | 28.2 / 27.8 / 9.7% | 0.662 | 21 / 10% (best: 37 / 2% at 0.25) | 0.75 |
| Dog-only retrain, 2 seeds | 27.8-29.3 / 21.8-28.2 / 8.9-11.2% | 0.661-0.679 | - | - |
| Cat-only | 15.8 / 13.4 / 12.1% | 0.648 | 54 / 0.5% | 0.87 |
| Joint, cat bursts dropped, 2 seeds | 28.7-29.7 / 22.5-23.7 / 10.8-11.3% | 0.675-0.678 | 49-60 / 1% | 0.84-0.87 |
| Joint, every source deduped, 2 seeds | 25.4-26.1 / 26.5-29.2 / 9.7-9.9% | 0.649-0.658 | 58-59 / 1% | 0.87-0.89 |

- One joint embedder matches the cat-only model on cats and stays inside the
  dog-only seed spread on dogs. Per-species embedders would add ~344 MB to the
  image for no measurable gain.
- Deduping DogFaceNet and MPDD (31% and 35% "bursts") cost dogs ~2pp F: their
  similar-looking crops are signal. Only cats are deduped.
- Sub-center ArcFace (k=3) was within noise of ArcFace.
- DogReID top-1 across seeds (scratch PyTorch harness, before
  `scripts/evaluate_release.py`): two dog-only retrains of the 0.2.0 recipe
  scored 0.539 and 0.488; four 0.3.0 seeds scored 0.515-0.529. The release seed
  (13) was picked on validation mAP (0.570 vs 0.567 for the others).
- Release Max Distance for dogs: 0.375. At 0.4 the 0.3.0 embedder merges like
  0.2.0 did (~10% of pets merged with a sibling); 0.375 cuts that ~2 points with
  the same share in one correct person, for ~9 points more dogs never grouped.
  0.35 would reach 0.1.1's merge rate but leave ~41% never grouped.
- Cats cluster tighter than dogs: `CAT_MAX_DISTANCE` 0.35 is the knee (1%
  merged in households, Commons purity ~0.87; 0.4 doubles merges).
- Dogs and cats almost never share a cluster (≤0.3% of clusters).

## Detector (YOLO11n, dog + cat, fine-tuned 30 epochs from 0.2.0, conf 0.3)

| | 0.2.0 | Dog + cat |
|---|---|---|
| DogReID owner dogs | 94.7% | 95.3% |
| Commons dogs (right class) | 85.2% | 84.4% (81.4%) |
| Owner cats (right class) | 8% (as dog) | 99.0% (98.8%) |
| Commons cats (right class) | 19% (as dog) | 91.2% (87.9%) |
| Dog- and cat-free photos with a detection | 25.5% | 23.4% |

Exported with class-agnostic NMS so one animal is never both a dog and a cat
face. About 3% of Commons dogs come back only as "cat"; with a shared embedder
the label only selects the min score and Max Distance mapping.

## Caveats

- One cat identity source, from one city's shelters, with machine-made boxes;
  Commons gives only 50 cats in the wild.
- Not yet run through Immich itself.
