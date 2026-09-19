# Embedding Backbone Ablation — Findings

_Run 2026-09-18/19. Companion to [plan.md](plan.md). All four questions answered.
Live tables regenerate into `outputs/ablation/STATUS.md` (gitignored); this file
records the conclusions._

## Headline

Three things decided this, and only one of them was accuracy.

1. **The plan's candidate matrix was wrong about licensing.** Every ConvNeXt-V2
   weight tag is CC-BY-NC — including the one the repo was *already shipping* as
   `models/onnx/embedding.onnx`. The intended "shippable contender" could never
   have shipped.
2. **ONNX Runtime ranks these architectures differently than PyTorch does**, and
   ORT is what Immich runs. That moves the cost gate enough to change the answer.
3. **A train/serve preprocessing bug** was costing 1.3pp MRR in production,
   invisible to every offline metric in this ablation.

## Corrections to the plan

| Plan said | Actually |
|---|---|
| ConvNeXt-V2-Tiny/Nano are "Apache-2.0, shippable contender" | Every `convnextv2_*` tag is **cc-by-nc-4.0**. Reference ceiling only. |
| — (V1 absent from the matrix) | `convnext_tiny.fb_in22k_ft_in1k` is Apache-2.0 and scores higher here. |
| 849 train identities | 705 train / 147 val / 149 test (splits regenerated since). |
| Cost axis = "CPU latency" | Torch and ORT disagree sharply; only ORT reflects Immich. |

`BackboneSpec.license_tier` is now a **required** field, so no backbone can be
added without answering the question, and weight *tags* are pinned because the
licence rides on the tag rather than the architecture.

## Q1 — Baseline gap: how much does a modern backbone beat ResNet50?

**+7.3pp MRR, +9.8pp Top-1, three seeds each.**

| backbone | MRR (3 seeds) | Top-1 | licence |
|---|---|---|---|
| convnext_tiny | **0.9661 ± 0.0007** | 0.9511 | Apache-2.0 |
| resnet50 (baseline) | 0.8935 ± 0.0038 | 0.8528 | Apache-2.0 |

The seed spread is ~15x smaller than the gap, so this is a real effect rather
than a lucky seed. The linear probe predicted the fine-tuned ranking (probe
.9605 vs .8752), which is the assumption Stage A's design rests on.

ResNet50 early-stopped on all three seeds (best at full-phase epoch 30/20/31 of
45) while ConvNeXt used the entire budget and was still gaining +0.0003 mAP over
its final ten epochs — the baseline has saturated on this data; the replacement
has not.

## Q2 — Best shippable (Apache) backbone

**ConvNeXt-V1-Tiny on accuracy, subject to the latency decision below.**
`convnext_small` was eliminated at the Stage A cost gate (−0.2pp MRR for +75%
ORT latency); both EfficientNetV2 variants lost on accuracy and cost.

### The open decision: the latency gate

ONNX Runtime's CPU provider fuses conv+BN aggressively — essentially all of
ResNet50 — and gives ConvNeXt nothing, because LayerNorm, GELU, 7x7 depthwise
convs and tensor permutes do not fuse. Measured on an idle machine
(`outputs/ablation/latency.csv`):

| backbone | torch ms | ORT ms | change |
|---|---|---|---|
| resnet50 | 37.8 | **16.2** | −57% |
| convnext_tiny | 38.2 | **29.1** | −24% |

The two are within 0.4 ms of each other in PyTorch and 12.9 ms apart under ORT.
Because the agreed budget is a *multiple* of the baseline (≤1.5x), the ORT
budget is **24.3 ms** and convnext_tiny fails it at **1.80x**.

**Strictly applied, the locked decision rule selects ResNet50.** The substantive
question is whether **+7.3pp MRR is worth +12.9 ms per crop on CPU**. That is a
product call, not a measurement, and it is the one thing this ablation cannot
decide for you. Both artifacts are built so it can be decided without re-running
anything.

_Caveat on precision:_ this timing harness is noisy — repeated ResNet50 ORT
measurements on one box span 14.7–22.9 ms (30 iterations, threads unpinned), so
the ratio lands somewhere in 1.6–2.0x depending on the run. Every instrument
puts it above 1.5x, so the **direction** is solid; treat the exact figure as
approximate.

## Q3 — Ceiling gap vs encumbered backbones

**The ceiling came in below the floor.** Fine-tuned, MegaDescriptor-T scores
**0.883 MRR** — under the ImageNet ResNet50 baseline (0.893) and 8.3pp under the
Apache winner (0.966).

| tier | backbone | MRR | licence |
|---|---|---|---|
| shippable | convnext_tiny | **0.966 ± 0.001** (n=3) | Apache-2.0 |
| reference | convnextv2_tiny | 0.960 (n=1) | CC-BY-NC |
| baseline | resnet50 | 0.893 ± 0.004 (n=3) | Apache-2.0 |
| animal-pretrained | megadescriptor_t_224 | 0.883 (n=1) | CC-BY-NC |

So there is **no ceiling to chase**: per decision **D1**, MiewID and DINOv3 stay
deferred and their bespoke loaders were never built — now settled by measurement
at fine-tune, not merely at linear probe.

The Apache backbone also edges the CC-BY-NC one (0.966 vs 0.960), so the licence
constraint did not cost accuracy here. Read that loosely: convnextv2_tiny is a
single seed with no error bar, and plan.md warns against over-reading
single-seed gaps. The defensible claim is "not worse", not "better".

This reproduces PetFace rather than the wildlife-benchmark literature: the cited
20–70pp animal-pretraining wins did not transfer to ~7 images/identity open-set
dog faces. A reported delta is a measurement on someone else's distribution.

## Q4 — Head effect (ArcFace vs Sub-center)

**Neutral. Sub-center ArcFace neither helps nor hurts here.**

| backbone | arcface | sub-center (K=3) | delta |
|---|---|---|---|
| convnext_tiny | 0.9661 ± 0.0007 | 0.9670 ± 0.0005 | +0.09pp |
| resnet50 | 0.8935 ± 0.0038 | 0.8972 ± 0.0055 | +0.37pp |

Both deltas sit inside the combined seed spread, and sub-center *raised*
ResNet50's variance without raising its mean — usually the signature of capacity
the data cannot support. This is the plan's predicted sample-starvation outcome:
K=3 at ~7 images/identity leaves ~2 images per sub-centre, too few to learn
distinct intra-class modes.

Consequently `summarize_ablation.py --best-head` requires an improvement to
clear the combined spread before it displaces ArcFace, and returns **arcface**
for both backbones. A plain `max()` would have shipped sub-center on noise.

## Shipped model

Two artifacts, so selection keeps a choice the way Immich does for faces. Both
exported from their **seed-42 ArcFace** Stage B checkpoints — the canonical seed
rather than the best-scoring one, since max-over-seeds on the reported split is
an optimistic bias.

| artifact | backbone | test MRR | Top-1 | ORT CPU | licence |
|---|---|---|---|---|---|
| `models/onnx/embedding.onnx` | convnext_tiny | 0.9668 | 0.9527 | 29.1 ms | Apache-2.0 |
| `models/onnx/embedding_resnet50.onnx` | resnet50 | 0.8960 | 0.8571 | 16.2 ms | Apache-2.0 |

Headline numbers for the *method* are the 3-seed means above (0.966 / 0.893);
the per-artifact figures here are that one seed's.

Both verified: 512-d, L2-normalised, PyTorch↔ORT parity ≤1.6e-06, and loadable
through `ONNXEmbedding`. Each carries a `.json` sidecar recording backbone,
head, source run, parity, latency, test metrics, DBSCAN `eps` and the full
preprocessing recipe. The previous `embedding.onnx` was a CC-BY-NC ConvNeXt-V2,
preserved as `embedding.pre-ablation.onnx`.

**Clustering.** `eps = 0.35`, `min_samples = 3` (Immich's `maxDistance` /
`minFaces`). This is swept per model because it depends on embedding geometry
and cannot be inherited across a backbone swap. The sweep runs on the **test**
split, so the clustering scores are best-case; the retrieval metrics involve no
tuning and stay unbiased.

**Preprocessing (required).** RGB, resize to 224x224, `pixels / 255`, then
ImageNet `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, NCHW. Serving
raw `[0,1]` silently costs ~1.3pp MRR — see below.

Regenerate either artifact:

```bash
uv run python scripts/train_final.py --backbone convnext_tiny --head arcface \
    --checkpoint runs/<run>/best_model.pt --output models/onnx/embedding.onnx
```

Or retrain from scratch on train+val (see the negative result below):

```bash
uv run python scripts/train_final.py --backbone convnext_tiny --include-val
```

## Train/serve skew (found and fixed)

`IdentityDataset` trains on ImageNet-normalised input; `ONNXEmbedding` served
raw `[0, 1]`. Nothing errored — the model simply received a distribution it was
never trained on. Measured on the test split with the then-deployed model:

| preprocessing | MRR | Top-1 |
|---|---|---|
| as served (`[0,1]` only) | 0.9474 | 0.9249 |
| as trained (ImageNet mean/std) | **0.9602** | **0.9416** |

The normalised path reproduces that model's recorded benchmark (0.9602) exactly,
which is the proof: the model was fine, the serving code was wrong. That 1.3pp
is roughly 18% of the entire backbone-swap gain, recovered for zero compute.

The YOLO detector and keypoint stages genuinely do take `[0, 1]`, so the
normalisation is overridden on the embedder alone. Regression test:
`tests/unit/test_serving_preprocessing.py`.

**The general hazard:** every offline metric in this ablation ran through
`IdentityDataset`, so the benchmark was structurally incapable of seeing a bug
that existed only on the serving path. An accuracy ablation cannot detect a
deployment bug.

_Residual:_ training resizes with PIL bilinear+antialias, serving with
`cv2.INTER_AREA`. Far smaller than the normalisation gap, but not zero.

## Negative result: training on train+val did not help

Folding the 147 validation identities into training (+21% identities, +22%
images) was expected to help — more classes is the strongest lever in metric
learning. It did not:

| backbone | train-only (3 seeds) | train+val | delta |
|---|---|---|---|
| convnext_tiny | 0.9670 ± 0.0005 | 0.9676 | +0.06pp |
| resnet50 | 0.8972 ± 0.0055 | 0.9024 | +0.52pp |

Both inside the seed spread; ConvNeXt's Top-1 went the other way (0.9527 vs
0.9530 train-only), which is what "no real change" looks like across two metrics
rather than one noisy one. The likely reason is the trade the method embeds: folding val in buys
identities but **spends the early-stopping signal**, forcing a fixed epoch
budget. Data gained, model selection lost, roughly cancelling. A better version
would hold out a slice of *train* identities for selection instead.

The shipped artifacts are therefore the train-only ones — not because train+val
was worse, but because no measurable gain justifies the less-verifiable recipe.

## Method notes

- Stage A froze the trunk so every candidate saw identical hyperparameters,
  removing the per-backbone LR confound rather than trying to control for it.
- Model selection on val (mAP), reporting on the held-out test split. Val/test
  identities are disjoint from train (open-set), so the margin head is discarded
  at inference and only the 512-d L2-normalised embedding ships.
- Stage C cells differ from their ArcFace counterparts in `head_type` alone —
  same 25+45 epoch budget, LRs, batch size and image size — so the head
  comparison is not confounded.
- Cost is measured on untrained models: latency is a property of the
  architecture, not the weights, so the whole cost axis costs minutes.
- All candidates are timed on **one** instrument. A budget is a comparison, so
  mixing a torch number with an ORT one is not a comparison at all.

## Known issues found along the way

- `models/onnx/embedding.onnx` was, before this work, an exported
  **ConvNeXt-V2** — the CC-BY-NC problem was already shipped, not merely planned.
- `IdentityDataset.num_classes` is `max(identity_label) + 1`, not the unique
  count, so the margin head allocates 1001 prototypes for 705 identities. The
  296 phantom classes only ever act as negatives. Left alone deliberately:
  changing it would make the final model incomparable to every measured cell.
- `trainer.py` hardcoded `label_smoothing=0.1` while `HEAD_CONFIG` plumbed a
  value into the head, so configuring it did nothing. Fixed.
- `train_master.py` still uses `DATA_CONFIG.img_size` rather than the backbone's
  native size, so `--backbone megadescriptor_l_384` would train at the wrong
  resolution. Not hit here (every other backbone is 224).
- Results live in gitignored `outputs/`, so a `git clean` would destroy a
  multi-day sweep. Append-only is a convention with no audit trail.
- The latency harness is noisy (±35% across runs) for a gate expressed as a
  1.5x ratio. Pinning threads and raising iteration counts would tighten it.
