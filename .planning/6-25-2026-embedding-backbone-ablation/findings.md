# Embedding Backbone Ablation — Findings

_Run 2026-09-18. Companion to [plan.md](plan.md). Live numbers are in
`outputs/ablation/STATUS.md` (gitignored); this file records the conclusions._

> **Status: Stage B complete (3 seeds), Stage C in flight.** Q1 and Q3 are
> answered below; Q4 fills in when the head A/B finishes.

## Headline

The plan's candidate matrix was wrong about licensing, and that — not accuracy —
decided the outcome. ConvNeXt-**V2**, the intended shippable contender and the
Stage A winner, is **CC-BY-NC on every published weight tag**. ConvNeXt-**V1**
(Apache-2.0) was added mid-ablation, and scores slightly *better* on our data.

A second surprise landed at the cost gate: **ONNX Runtime does not treat these
architectures equally**, which changes the ranking relative to PyTorch timings.

## Corrections to the plan

| Plan said | Actually |
|---|---|
| ConvNeXt-V2-Tiny/Nano are "Apache-2.0, shippable contender" | Every `convnextv2_*` tag is **cc-by-nc-4.0**. Reference ceiling only. |
| — (V1 absent from the matrix) | `convnext_tiny.fb_in22k_ft_in1k` is Apache-2.0 and beats V2 here. |
| 849 train identities | 705 train / 147 val / 149 test (splits regenerated since). |
| Cost axis = "CPU latency" | Torch and ORT latency disagree sharply; only ORT reflects Immich. |

The registry now carries `BackboneSpec.license_tier` as a **required** field, so
no future backbone can be added without answering the question. Weight tags are
pinned, because the license rides on the tag and not on the architecture.

## Q1 — Baseline gap: how much does a modern backbone beat ResNet50?

**Answered. +7.3pp MRR, +9.8pp Top-1, on 3 seeds each.**

| backbone | MRR (3 seeds) | Top-1 | licence |
|---|---|---|---|
| convnext_tiny | **0.966 ± 0.001** | 0.951 | Apache-2.0 |
| resnet50 (baseline) | 0.893 ± 0.004 | 0.853 | Apache-2.0 |

The seed spread is tiny — σ = 0.001 and 0.004 — so the gap is roughly 15x the
combined noise. This is a real effect, not a lucky seed, and it is the one
number the whole ablation existed to produce.

Two method notes. First, the linear probe predicted the fine-tuned ranking
correctly (probe: .961 vs .875; fine-tuned: .966 vs .893), which is the
assumption Stage A's design rests on — it held. Second, ResNet50 early-stopped
on all three seeds (best at epoch 20/30/31 of 45) while ConvNeXt used the full
budget and was still nudging upward at +0.0003 mAP/10 epochs. That asymmetry is
evidence the baseline has saturated on this data and the replacement has not.

## Q2 — Best shippable (Apache) backbone

**ConvNeXt-V1-Tiny** on accuracy, subject to the latency decision below.
`convnext_small` was eliminated at the Stage A cost gate: −0.2pp MRR for +76%
ORT latency. `tf_efficientnetv2_m` and `efficientnetv2_rw_m` lost on both axes.

### The open decision: the latency gate

ONNX Runtime's CPU provider fuses conv+BN aggressively — which is essentially
all of ResNet50 — and gives ConvNeXt nothing, because LayerNorm, GELU, 7x7
depthwise convs and tensor permutes do not fuse:

| backbone | torch ms | ORT ms | change |
|---|---|---|---|
| resnet50 | 38.2 | **22.4** | −41% |
| convnext_tiny | 36.3 | **38.0** | +5% |

The two are within 2 ms in PyTorch and 15.6 ms apart in the runtime Immich
actually uses. Because the agreed budget is a *multiple* of the baseline
(≤1.5x), ORT tightened it from 55.8 ms to 33.6 ms, and ConvNeXt-V1-Tiny fails
at 1.70x.

**Strictly applied, the locked decision rule now selects ResNet50.** The
substantive question is whether **~+7pp MRR is worth +15.6 ms per crop on CPU**.
That is a product call, not a measurement, and it is the one thing this ablation
cannot decide for you.

## Q3 — Ceiling gap vs encumbered backbones

**Answered, and the ceiling is below the floor.** Fine-tuned on 1 seed,
MegaDescriptor-T scores **0.883 MRR** — under the ImageNet ResNet50 baseline
(0.893) and 8.3pp under the shippable Apache winner (0.966).

| tier | backbone | MRR | licence |
|---|---|---|---|
| shippable | convnext_tiny | **0.966** | Apache-2.0 |
| reference | convnextv2_tiny | 0.960 | CC-BY-NC |
| baseline | resnet50 | 0.893 | Apache-2.0 |
| animal-pretrained | megadescriptor_t_224 | 0.883 | CC-BY-NC |

So there is **no ceiling to chase**: per decision **D1**, MiewID and DINOv3 stay
deferred and their bespoke loaders were never built. That decision is now
settled by measurement at fine-tune, not merely at linear probe.

Note also that the best *permissive* backbone beats the best *encumbered* one
(0.966 vs 0.960). The licence constraint cost nothing here — it gained.

This reproduces PetFace rather than the wildlife-benchmark literature: the cited
20-70pp animal-pretraining wins did not transfer to ~7 images/identity open-set
dog faces. A reported delta is a measurement on someone else's distribution.

## Q4 — Head effect (ArcFace vs Sub-center)

**PENDING** — Stage C runs after Stage B on the winning backbone.

## Method notes

- Stage A used a frozen trunk so every candidate saw identical hyperparameters,
  removing the per-backbone LR confound rather than trying to control for it.
- Model selection on val (mAP), reporting on the held-out test split; val/test
  identities are disjoint from train (open-set), so the ArcFace head is
  discarded at inference and only the 512-d L2-normalised embedding ships.
- Cost is measured on untrained models: latency is a property of the
  architecture, not the weights, so the whole cost axis costs minutes.
- All candidates are timed on **one** instrument. A budget is a comparison, so
  mixing a torch number with an ORT one is not a comparison at all.

## Shipped model

Two artifacts, so selection keeps a choice the way Immich does for faces.
Both verified: 512-d, L2-normalised, PyTorch/ORT parity, and loadable through
`ONNXEmbedding`.

| artifact | backbone | test MRR | Top-1 | ORT CPU | licence |
|---|---|---|---|---|---|
| `models/onnx/embedding.onnx` | convnext_tiny | **0.9668** | 0.9527 | 39.3 ms | Apache-2.0 |
| `models/onnx/embedding_resnet50.onnx` | resnet50 | 0.8960 | 0.8571 | 19.9 ms | Apache-2.0 |

Each ships a `.json` sidecar with backbone, seed, source run, parity, test
metrics, the tuned DBSCAN `eps` (0.35, `min_samples` 3) and the full
preprocessing recipe. The previous `embedding.onnx` was a CC-BY-NC
ConvNeXt-V2; it is preserved as `embedding.pre-ablation.onnx`.

Regenerate either from its checkpoint:

```bash
uv run python scripts/train_final.py --backbone convnext_tiny --head arcface \
    --seed 42 --checkpoint runs/<run>/best_model.pt --output models/onnx/embedding.onnx
```

Or retrain on train+val (folds in 147 more identities; test stays held out):

```bash
uv run python scripts/train_final.py --backbone convnext_tiny --include-val
```

**Open decision.** Under the locked rule (maximise MRR subject to permissive
licence, ONNX-exportable, latency <= 1.5x the ResNet50 baseline) the ORT budget
is 33.6 ms and convnext_tiny's 39.3 ms fails it at 1.70x. The trade is
**+7.3pp MRR for +19.4 ms per crop on CPU**. Both artifacts are built so this
can be decided without re-running anything.

## Train/serve skew (found 2026-09-18, fixed)

`IdentityDataset` trains on ImageNet-normalised input; `ONNXEmbedding` served
raw `[0, 1]`. Nothing errored — the model simply received a distribution it was
never trained on. Measured on the test split with the deployed model:

| preprocessing | MRR | Top-1 |
|---|---|---|
| as served (`[0,1]` only) | 0.9474 | 0.9249 |
| as trained (ImageNet mean/std) | **0.9602** | **0.9416** |

The normalised path reproduces the recorded benchmark (0.9602) exactly, which
is the proof: the model was fine, the serving code was wrong. That 1.3pp is
roughly 18% of the entire backbone-swap gain, recovered for zero compute.

The YOLO detector and keypoint stages genuinely do take `[0, 1]`, so the
normalisation is overridden on the embedder alone rather than moved into the
shared preprocessing. Regression test:
`tests/unit/test_serving_preprocessing.py`. The exported sidecar now records
the full preprocessing recipe so no consumer has to guess it.

This is the general hazard worth remembering: every offline number in this
ablation was computed through `IdentityDataset`, so the benchmark was blind to
a bug that only existed on the serving path. An accuracy ablation cannot see a
deployment bug.

## Known issues found along the way

- `trainer.py` hardcodes `CrossEntropyLoss(label_smoothing=0.1)` while
  `HeadConfig.label_smoothing` is plumbed into the head — setting the config
  silently does nothing. A no-op at the 0.1 default; fix deferred until the
  sweep finishes so every cell runs identical code.
- `train_master.py` still uses `DATA_CONFIG.img_size` rather than the backbone's
  native size, so `--backbone megadescriptor_l_384` would train at the wrong
  resolution. Not hit here (every other backbone is 224).
- Results live in gitignored `outputs/`, so a `git clean` would destroy a
  multi-day sweep.
- `models/onnx/embedding.onnx` was, before this work, an exported
  **ConvNeXt-V2** — i.e. the CC-BY-NC licence problem was already shipped, not
  merely planned. Backed up to `embedding.pre-ablation.onnx`.
- `IdentityDataset.num_classes` is `max(identity_label) + 1`, not the unique
  count, so the margin head allocates 1001 prototypes for 705 identities. The
  296 phantom classes only ever act as negatives. Left alone deliberately:
  changing it would make the final model incomparable to every measured cell.
