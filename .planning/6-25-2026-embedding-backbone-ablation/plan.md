# Embedding Backbone Ablation — Plan

_Date: 2026-06-25. Goal: ground the backbone-swap decision (audit SOTA rec 1–3) in
measured numbers on our own data before merging the backbone-integration PR._

## Why this exists

The SOTA audit says ResNet50-ImageNet is the one stale component and a modern /
animal-pretrained backbone is the highest-ROI change. But the headline 20–70pp
wins are on **wildlife** benchmarks; PetFace (closest to our domain) found vanilla
ArcFace competitive. Our data is small + open-set (~6k images, 849 train
identities, ~7 img/identity, val/test identities disjoint from train). So the
transfer is *likely* but unverified. This ablation measures it before we commit a
backbone into the Immich fork.

## Questions to answer

- **Q1 — Baseline gap:** how much does any modern backbone beat ResNet50 on *our*
  dog-face, open-set data?
- **Q2 — Best shippable:** among permissively-licensed (Apache-2.0) backbones,
  which wins? This is the one we actually deploy.
- **Q3 — Ceiling gap:** how far behind is the best *shippable* backbone vs the best
  *encumbered* one (MegaDescriptor / MiewID / DINOv3)? This tells us whether it's
  worth chasing the MiewID license or accepting DINOv3's terms.
- **Q4 — Head effect:** does Sub-center ArcFace help at 7 img/identity, or is it
  sample-starved and neutral/harmful? (Free to test — one-line config flip.)

## What's already in place

- Held-out **test split** (PR #6, merged) — select on val, report on test.
- **Configurable head** (PR #7, open) — ArcFace / Sub-center / CosFace via config.
- **Backbones plumbed** (worktree, unmerged) — timm ConvNeXt-V2 / EfficientNetV2,
  MegaDescriptor-T/L; MiewID stubbed; DINOv3 not integrated.
- **Determinism** — `common/seed.py` wired into train_master (seeded RNG + loader).
- **Metrics** — `benchmark/evaluator.py`: MRR, Top-1/3/5, TAR@FAR on test.
- **Linear-probe = trainer Phase 1** (freeze backbone, train head only).

## Methodology

### Controls (held constant across all runs)
Same train/val/test JSON splits; same augmentation policy; embedding dim = 512,
L2-normalized, cosine; same batch size; same seed (per-seed where we replicate);
same early-stopping-on-val / report-on-test protocol.

### The confound we must manage: per-backbone LR
The current differential LRs (head 1e-4, backbone 1e-6) were tuned for ResNet50.
Different backbones have different optimal LRs, so a single full-finetune sweep at
fixed LR is *unfair* and could rank a good backbone last purely on optimization.
Solution = two stages:

- **Stage A — Linear probe (frozen backbone).** Train only the projection + head
  (existing Phase 1). The backbone never updates, so there is **no backbone-LR to
  tune** — every candidate sees identical hyperparameters. This isolates raw
  feature quality and is a fair, cheap ranking. Standard practice (DINO papers
  report linear-probe). Fast: only the small head trains.
- **Stage B — Full fine-tune.** Run the existing 2-phase (warmup → unfreeze) on the
  **top 3–4 from Stage A + ResNet50 baseline**, to get real deployable numbers. Only
  here do we spend the compute to tune/accept the differential LR.

### Reproducibility / variance
Determinism is handled. Use **1 seed** for the broad Stage A sweep; **3 seeds** for
the Stage B finalists so the headline comparison has error bars (a 1pp "win" inside
seed noise is not a win).

### Cost axis (gates shippability, not just accuracy)
Immich's ML worker is CPU-bound. For every Stage B finalist also measure:
**CPU inference latency**, **param count / ONNX size**, and **ONNX export success**
(Swin/ViT may not export cleanly under the current legacy exporter — a hard gate).
A backbone that wins accuracy but doubles CPU latency or won't export loses.

## Candidate matrix

| Backbone | License tier | Role | Stage |
|---|---|---|---|
| ResNet50 (current) | permissive | baseline / control | A + B |
| ConvNeXt-V2-Tiny | Apache-2.0 | shippable contender | A + B |
| ConvNeXt-V2-Nano | Apache-2.0 | shippable, lighter | A |
| EfficientNetV2-M | Apache-2.0 | shippable, MiewID-arch proxy | A + B |
| MegaDescriptor-T-224 | CC-BY-NC | animal-pretrained ceiling | A + B |
| MegaDescriptor-L-384 | CC-BY-NC | heavier animal ceiling | A (B if T wins big) |
| _MiewID-msv3_ | none (eval-only) | animal ceiling probe | **deferred** (D1) |
| _DINOv3-ConvNeXt-S_ | encumbered | future ceiling | **deferred** (D1) |

Head A/B (**Stage C**): on the winning *shippable* backbone, ArcFace vs Sub-center
ArcFace (K=3, optionally K=2). CosFace dropped (audit: marginal).

## Run plan (single local GPU, serial)

- **Phase 0 — build the harness** (see prerequisites). No training yet.
- **Phase A — linear probe**, all candidates, 1 seed. Rank by Top-1/MRR on test.
- **Phase B — full fine-tune**, top 3–4 + ResNet50 baseline, 3 seeds each.
- **Phase C — head A/B** on the Stage-B winner (ArcFace vs Sub-center).
- **Phase D — deploy validation** of the chosen backbone: ONNX export + CPU latency
  + confirm 512-d L2-normalized contract preserved (projection head).

Rough budget: Stage A ~6 short runs; Stage B ~4×3 = 12 longer runs; Stage C ~2–3.
~20–25 serial runs total — days of wall-clock on one GPU, not hours. Worth pruning
aggressively at the Stage A gate.

## Decision rule

Pick the backbone that **maximizes test MRR/Top-1 subject to: Apache-permissive
license AND ONNX-exportable AND CPU latency within budget.** Report the encumbered
backbones (MegaDescriptor / MiewID / DINOv3) as reference ceilings — if a ceiling
beats the best shippable by a margin that matters, that's the trigger to chase the
MiewID license or accept DINOv3's terms.

## Implementation prerequisites (code, before any training)

1. **Parametrize the entrypoint** — add `--backbone` and `--head` (+ relevant
   hyperparams) CLI overrides to `train_master.py` so a run is fully specified
   without editing `config.py`. (Currently hardcodes `DEFAULT_BACKBONE`.)
2. **Ablation runner + results aggregation** — `scripts/run_ablation.py` that
   iterates a config list, invokes training, and appends each run's test
   metrics + cost (latency, params, ONNX-ok) to a `results.csv` / markdown table
   under this planning dir. Idempotent / resumable (single GPU, long jobs).
3. **(Stage A support)** — a "linear-probe only" mode (warmup-phase-only / freeze
   for all epochs) so Stage A doesn't pay for fine-tuning.
4. **(Deferred per D1)** — MiewID loader / DINOv3 integration: only after Stage A
   shows the animal-pretrained tier winning.
5. **(Nice-to-have)** — wire W&B into the embedding trainer (audit #7) for run
   comparison; otherwise the CSV/markdown table is the system of record.

## Decisions (locked 2026-06-25)

- **D1 — Backbone scope: plumbed-only first pass.** Stage A uses the
  already-integrated timm Apache backbones + MegaDescriptor (T/L) as the
  animal-pretrained ceiling. **MiewID and DINOv3 are deferred** — their bespoke
  loaders get built only if the animal-pretrained tier clearly wins Stage A
  (so we never build a bespoke loader blind). MegaDescriptor serves as the
  animal-pretrained proxy ceiling in the meantime.
- **D2 — Rigor: two-stage + seeds.** Linear-probe broad sweep (1 seed) → full
  fine-tune of the top 3–4 + ResNet50 baseline (3 seeds each for error bars).
  Chosen over single fixed-LR finetune because it removes the LR confound and is
  cheaper.

## Risks

- **Residual LR confound** in Stage B even with linear-probe gating — mitigate with a
  short LR-range check per finalist.
- **ONNX export** for Swin/ViT backbones under the legacy exporter — may block
  MegaDescriptor/DINOv3 deployment regardless of accuracy.
- **Sub-center starvation** at ~7 img/identity (K=3 → ~2 img/sub-center) — Q4 may
  come back "neutral/worse," which is itself a useful answer.
- **Small val/test** (152 test identities) — Top-1 swings of ~1–2pp may be noise;
  lean on MRR + 3-seed error bars, don't over-read single-seed Stage A gaps.
