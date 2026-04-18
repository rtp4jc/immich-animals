# SOTA Audit: immich-animals vs 2026 State of the Art

_Date: 2026-04-16. Conceptual comparison only — no benchmarks were executed._

## Executive Summary

The current pipeline is a faithful port of Immich's face recognition stack (YOLO detector → keypoint alignment → ResNet50+ArcFace embedding, 512-dim, L2-normalized, cosine-matched). For **human** faces that contract is still reasonable. For **individual animal re-ID**, the field has moved decisively toward (a) domain-specific foundation backbones trained on aggregated wildlife corpora, and (b) sub-center / adaptive-margin losses that tolerate noisy multi-identity data. The two strongest wins would come from swapping the embedding backbone (ResNet50 → MegaDescriptor / MiewID / DINOv3-initialized Swin or ConvNeXt) and upgrading the loss (ArcFace → Sub-center ArcFace with dynamic margins, or AdaFace). The detector is fine; keypoints are probably dead weight at our current scale.

| Stage | Current choice | Verdict | Notes |
|---|---|---|---|
| Detection | YOLO11n | OK | Still near-SOTA for latency-bounded real-time detection. RT-DETR / RF-DETR / YOLOv12 offer marginal mAP gains, usually not worth the ONNX-export pain at n-scale. |
| Keypoints / alignment | YOLO11n-pose, 4 landmarks (disabled) | Stale / optional | Consistent with 2024+ animal re-ID literature — keypoint-free global-embedding approaches dominate. Keeping it disabled is correct. |
| Embedding backbone | ResNet50 (ImageNet-pretrained) | Stale | MegaDescriptor (Swin-L, WACV'24 best paper) and MiewID (EffNetV2-M, 2024) both dominate ResNet50 on animal re-ID by wide margins. BioCLIP 2 / DINOv3 are credible initialization sources for 2026. |
| Loss | ArcFace (m=0.5, s=30) | OK, upgradeable | Sub-center ArcFace + dynamic margins (MiewID recipe) or AdaFace (CVPR'22, quality-adaptive) both beat vanilla ArcFace on noisy / unbalanced data, which DogFaceNet has. |
| Embedding dim / contract | 512-d, L2-normalized, cosine | OK for Immich interop | Immich's buffalo_l is 512-d L2-normalized cosine. Matching it avoids needing to touch Immich's DBSCAN clustering or pgvector index. |

---

## Per-Stage Comparison

### 1. Detection

**Current:** YOLO11n (Ultralytics), trained on COCO dog subset, exported to ONNX. Produces dog bounding boxes at ~COCO-grade accuracy.

**2026 landscape:**
- **YOLO11 / YOLOv12** — YOLOv12 (sunsmarterjie, NeurIPS 2025) introduces area-attention (A²) and flash-attention. Marginally higher mAP than YOLO11 at equal scale, but flash-attention isn't well-supported on CPU / ONNXRuntime-Web, and the Roboflow / blog-measured throughput is ~25% lower than YOLO11 at n-scale. For an n-class detector where we already have headroom, there's little reason to chase this.
- **RT-DETR / RT-DETRv2 / RF-DETR** — RF-DETR (Roboflow, 2025) is the first real-time detector to clear 60 mAP on COCO (60.5 @ 25 FPS on T4). Better for highly-occluded / small-face scenarios. ONNX export is supported upstream, but the decoder attention layers are heavier than YOLO convolutions, so CPU inference inside Immich's ML service would regress.
- **Grounding DINO** (ECCV 2024) — open-vocabulary, prompt-conditioned. Useful if you want to detect species by name without retraining (e.g., "a cat face", "a rabbit face"). Too heavy to run as a first-stage real-time detector (~500 MB weights, no small variant); good as a labeling/pseudo-labeling assistant, not for serving.
- **Academic animal-face detectors** — The cattle muzzle work by Garcia et al. (Sep 2025) uses Grounding DINO as a label generator and fine-tunes a lightweight detector on top. That pattern (open-vocab model → pseudo-labels → small supervised detector) is the practical 2026 move for expanding to non-dog species cheaply.

**Verdict: OK / hold.** YOLO11n was chosen sensibly and isn't the bottleneck in this pipeline. If a detector change comes up, RF-DETR-nano is the most credible direct swap. Low ROI vs. embedding changes.

Sources: [Ultralytics YOLO11 vs RT-DETR docs](https://docs.ultralytics.com/compare/rtdetr-vs-yolo11/), [Roboflow 2026 best detectors](https://blog.roboflow.com/best-object-detection-models/), [YOLOv12 repo](https://github.com/sunsmarterjie/yolov12), [Grounding DINO cattle muzzle paper](https://arxiv.org/html/2509.06427v1).

### 2. Keypoints and Alignment

**Current:** YOLO11n-pose with 4 facial landmarks on Stanford Dogs. Disabled at inference because benchmarks showed degraded top-k accuracy.

**2026 landscape:**
- **InsightFace school (strong alignment):** Human face-recognition pipelines align with 5-point landmarks + similarity transform, then feed a 112×112 crop. ArcFace was trained assuming this exact alignment. In buffalo_l this matters — removing alignment drops LFW-style accuracy by several points.
- **Wildlife re-ID school (no alignment / global embedding):** MegaDescriptor (WACV 2024, Best Paper) and MiewID (arXiv:2412.05602) both use **crop-only** preprocessing — no keypoint alignment — and dominate the PetFace, WildlifeReID-10k, and 29-dataset WildlifeDatasets benchmarks. The rationale is that (a) non-rigid animal pose variation is far worse than the rigid rotations human face alignment corrects, and (b) strong augmentation plus a large ViT/ConvNeXt learns pose-invariant representations without explicit alignment.
- **ATRW (Amur Tiger) specifically did ship pose-part models** (PPbM, PPGNet, PGCFL), but the 2024+ survey-level results show keypoint-free methods catching up or surpassing them when backbones get larger.
- **STN / dense correspondence approaches** exist (STNReID for partial-person re-ID, SEAS CVPR'24) but target occluded-person settings, not animal faces. Not a good fit here.

**Why our keypoint stage hurts:** Two compounding issues —
  1. Only 4 landmarks trained on Stanford Dogs (low-variety dataset), so the landmarks themselves are noisy.
  2. A noisy landmark → bbox refinement will crop too tight or off-center some fraction of the time. That fraction times "catastrophic crop" outweighs the gain on well-aligned cases. This is the classic "feature that needs to be perfect or should be absent" problem.

**Verdict: OK to keep disabled.** Don't spend effort on this. If you ever want alignment back, the literature-supported path is _dense_ alignment (affine from a learned STN, not a hand-picked 4-point landmark set) or nothing. Nothing is cheaper and nearly as good for animal re-ID.

Sources: [MegaDescriptor / WildlifeDatasets WACV'24](https://openaccess.thecvf.com/content/WACV2024/papers/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.pdf), [MiewID / multispecies re-ID Dec 2024](https://arxiv.org/abs/2412.05602), [ATRW tiger pose baselines](https://arxiv.org/abs/1906.05586).

### 3. Embedding Backbone

**Current:** ResNet50 (ImageNet V2 weights) → GAP → Dropout → Linear(2048, 512) → L2-normalize. ArcFace head on top at train time. EfficientNet-B0 and MobileNetV3-Small also plumbed in `backbones.py` but unused in the shipping pipeline.

**2026 landscape (ordered by ROI for our use case):**

1. **MegaDescriptor-L-384** (`BVRA/MegaDescriptor-L-384` on HF) — Swin-L/p4-w12-384, 228.8M params, 1536-d L2-normalized output, trained with ArcFace on 29 wildlife datasets. WACV 2024 best paper. Beats CLIP and DINOv2 by 20–70 pp Top-1 on challenging animal datasets. Smaller variants (-T-224, -S, -M) exist. There's also now a `MegaDescriptor-DINOv2-518` variant that initializes from DINOv2 features. **ONNX export**: Swin transformers are exportable (some known opset friction around `Concat` ops — see microsoft/Swin-Transformer#89 — but solvable with opset 17+). 228M params is heavy for CPU; -T or -S variants are more realistic for Immich's ML worker.

2. **MiewID** — EfficientNetV2-M backbone + sub-center ArcFace with dynamic margins, 2048-d output after GeM pooling + BN, trained on 49 species / 225K images. Reportedly beats MegaDescriptor by ~19 pp Top-1 on _unseen_ species. Already production-deployed in Wild Me / WildBook for 60+ species. **ONNX export**: EfficientNetV2 exports cleanly. This is probably the strongest single swap available for our architecture today.

3. **DINOv3** (Meta, Aug 2025) — self-supervised ViT, scaled to 7B params with 1.7B-image training set. Distilled ViT-B / ViT-L / ConvNeXt variants released for deployment. Commercial license. Outperforms weakly-supervised baselines (CLIP-tier) on downstream probing. **For our use case:** use as a frozen or LoRA-fine-tuned backbone with an ArcFace head trained on DogFaceNet. DINOv2 already-shown on HF as DINOv2-small-ONNX exports; DINOv3 export is claimed supported via the `onnx_export_merge` branch lineage. The ConvNeXt distilled variant is the ONNX-friendliest.

4. **BioCLIP 2** (NeurIPS 2025, OSU Imageomics) — CLIP ViT-L/14 further-trained on 214M images / 925K taxonomic classes (TREEOFLIFE-200M). Learns intra-species variation in orthogonal subspaces. Great species prior, but trained for species / trait recognition, not individual identity — would still need an ArcFace head + fine-tuning on DogFaceNet.

5. **ConvNeXt-V2 / Swin-V2 / EfficientNetV2** as ImageNet-pretrained swaps — all legitimate upgrades from ResNet50, but the gap is small compared to jumping to a domain-specific pretrain.

6. **CLIP / OpenCLIP visual encoders** — Great zero-shot, poor identity discrimination without fine-tuning. MegaDescriptor and MiewID both report CLIP is not competitive for individual re-ID.

**Export-friendly shortlist for this project:**
- MiewID (EffNetV2-M) — highest expected gain, cleanest export.
- MegaDescriptor-T-224 — cheaper Swin variant, already-trained on animals, just need an ONNX export pass.
- DINOv3-ViT-B or DINOv3-ConvNeXt-S fine-tuned with ArcFace on DogFaceNet — more flexibility, more work.

**Verdict: Stale.** ResNet50 ImageNet → 512-d is 2019-era. Switching backbones is the single highest-ROI change on this project.

Sources: [MegaDescriptor-L-384 HF](https://huggingface.co/BVRA/MegaDescriptor-L-384), [MiewID multispecies paper](https://arxiv.org/html/2412.05602v1), [DINOv3 Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/), [DINOv3 paper 2508.10104](https://arxiv.org/abs/2508.10104), [BioCLIP 2](https://imageomics.github.io/bioclip-2/), [dinov2_onnx repo](https://github.com/sefaburakokcu/dinov2_onnx).

### 4. Loss Function

**Current:** ArcFace (`losses.py`), additive angular margin m=0.5, scale s=30, label smoothing 0.1. TripletLoss is plumbed but unused. CosFace is referenced in AGENTS.md but not obviously wired in `losses.py` (only ArcFace is implemented).

**2026 landscape:**

| Loss | Year | Key idea | Fit for animal re-ID |
|---|---|---|---|
| ArcFace | 2018 | Additive angular margin | Baseline. Fine if data is clean. |
| CosFace | 2018 | Additive cosine margin | Marginal vs ArcFace. Ignore. |
| Sub-center ArcFace | 2020 | K sub-centers per class, dominant sub-class soaks noisy images | **Strong fit.** DogFaceNet has label noise + intra-identity appearance variance (puppy→adult, seasonal coat). Used by MiewID. |
| CurricularFace | 2020 | Curriculum-adaptive margin | Small gains. |
| MagFace | 2021 | Magnitude ∝ quality, quality-aware margin | Good for heterogeneous image quality. |
| AdaFace | CVPR 2022 | Feature-norm-based quality-adaptive margin | Beats ArcFace/MagFace/CurricularFace on IJB-B/C. Good for pet-photo datasets with variable focus/lighting. |
| UniFace / ElasticFace | 2022–2023 | Unified binary CE / elastic margin | Small gains on top of ArcFace; not obviously better for animals. |
| **Triplet + ArcFace hybrid** | common in re-ID | ArcFace for class separability + Triplet for fine intra-class structure | Frequently used in wildlife re-ID papers. Can help when you have many identities with few samples each (DogFaceNet has this shape). |

**Concrete recommendations for this project:**
- **Sub-center ArcFace (K=3) with dynamic margins** is what MiewID uses and what we should benchmark first. It directly addresses label noise and uneven samples per identity.
- **AdaFace** is the best pure-replacement if we want to stay single-loss. It's public, PyTorch, easily ported.
- **Triplet + ArcFace hybrid** is worth a benchmark pass, specifically with batch-hard mining.

**Verdict: OK, upgradeable.** ArcFace is a solid baseline; Sub-center ArcFace is the directionally-correct upgrade.

Sources: [AdaFace CVPR'22](http://cvlab.cse.msu.edu/project-adaface.html), [Sub-center ArcFace ECCV'20](https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf), [MiewID (uses sub-center + dynamic margin)](https://arxiv.org/html/2412.05602v1).

---

## Animal-Specific Approaches

Distinct from "transplanted human face ID", these approaches treat animal re-ID as its own problem:

- **Wildbook / Wild Me / MiewID** — production system. Ingests photos, runs species detection → re-ID per species. Recently collapsed per-species networks into a single multispecies embedding (MiewID). Strategy: one big dataset, one EfficientNetV2 + sub-center ArcFace, cosine matching. ~60 species in production.
- **MegaDescriptor / WildlifeDatasets toolkit** — the academic sibling. Swin-L + ArcFace, trained on 29 aggregated public datasets, ~140K images, 10K individuals, 23 species. Species-agnostic.
- **WildFusion** (CEUR 2025 AnimalCLEF) — calibrated similarity fusion over multiple backbones (MegaDescriptor, DINOv2, etc.). Useful trick when you can afford multiple forward passes.
- **PetFace** (ECCV 2024 Oral) — largest pet-focused benchmark, 257K individuals, 13 families, 319 breeds. The paper's own finding: ArcFace still beats other losses on this dataset at their scale. Relevant because it's the closest to our deployment domain (pet photos).
- **DogFaceNet literature improvements** — GhostDogFaceNets (2024) got Rank-1 from 60% → 86% with a lightweight design; EfficientFormer-L1 approaches (2023–2024) hit 97.6% accuracy. These are worth reading for training recipes, not necessarily for architecture copying.

**Key divergence from human face ID:**
1. **No landmark alignment** — pose is too non-rigid, and animal heads vary in shape far more than human faces.
2. **Heavier augmentation** (color jitter, seasonal-coat synthesis, random crop) carries more of the invariance burden.
3. **Sub-center losses** — noisier labels, multi-identity frames, age-range-within-identity.
4. **Foundation-model init from domain-specific pretrains** — MegaDescriptor / BioCLIP 2 change the floor for small-dataset fine-tuning.

Sources: [WildlifeReID-10k CVPR'25](https://openaccess.thecvf.com/content/CVPR2025W/FGVC/html/Adam_WildlifeReID-10k_Wildlife_re-identification_dataset_with_10k_individual_animals_CVPRW_2025_paper.html), [WildFusion](https://ceur-ws.org/Vol-4038/paper_253.pdf), [PetFace](https://arxiv.org/abs/2407.13555), [Dog face recognition ViT (Springer 2023)](https://link.springer.com/chapter/10.1007/978-3-031-45389-2_3).

---

## Immich Interop Constraints

What the Immich fork sees from our ONNX model matters for the rest of Immich's pipeline (DBSCAN clustering, face-person assignment, search).

| Contract element | Immich buffalo_l | Our model | Flexibility |
|---|---|---|---|
| Embedding dimension | 512 | 512 | **Match required** without touching Immich's DB schema (pgvector column width). |
| Normalization | L2-normalized | L2-normalized | **Match.** Immich's DBSCAN distance thresholds are tuned for unit-sphere vectors. |
| Distance metric | Cosine | Cosine (our pipeline too) | **Match.** L2-on-unit-sphere ⇔ cosine, either works. |
| Detection output | Bbox + 5 landmarks + det score | Bbox + det score (+ optional 4 kpts) | **Divergent, probably OK.** Immich uses landmarks for alignment; we don't align. As long as the detector ONNX output shape matches what Immich's preprocessor expects, we're fine. May need a small adapter in the fork. |
| Input image size to recognizer | 112×112 (buffalo_l) | Our own (224 or 384) | Handled internally by our ONNX; Immich resizes to the model's declared input shape. |
| Clustering thresholds | Tuned for buffalo_l | — | **Divergent.** Our cosine distribution will differ; Immich's default DBSCAN eps/min_samples will need re-tuning. This is a deployment-config issue, not an architecture one. |
| Model pack layout | `detection/model.onnx`, `recognition/model.onnx` | `copy_models.sh` already mirrors this | Match. |

**Hard constraints (must match):**
- 512-d L2-normalized, cosine-compared.

**Soft constraints (easily adapted):**
- Detector output shape (wrap in Immich fork if needed).
- Clustering thresholds (re-tune once per model change).

**Safe divergences:**
- Input resolution, normalization stats, backbone architecture, training loss — Immich doesn't know and doesn't care.

If we go to a backbone that natively emits 1536-d (MegaDescriptor) or 2048-d (MiewID), we can either:
- Add a learned linear projection `W: R^native → R^512` trained jointly (standard practice; slight accuracy cost), or
- Bump Immich's pgvector column width in the fork (one DB migration; no downstream code change).

The projection approach is the safer, zero-migration path.

Sources: [Immich buffalo_l HF](https://huggingface.co/immich-app/buffalo_l), [Immich face recognition docs](https://docs.immich.app/features/facial-recognition/), [Immich DeepWiki people/face page](https://deepwiki.com/immich-app/immich/4.2-people-and-face-recognition).

---

## Self-Supervised / Foundation-Model Angle

**Should we start from DINOv3 / MegaDescriptor / BioCLIP 2 instead of ImageNet ResNet?**

Yes — with caveats. The ordered effort/ROI ranking:

1. **Swap to a domain-pretrained backbone as drop-in feature extractor** (MegaDescriptor or MiewID, frozen or LoRA). Lowest effort, highest expected gain. Fine-tune an ArcFace head on DogFaceNet. Expect large Top-1 gains based on MegaDescriptor's own published 20–70 pp advantage over general-purpose pretraining.
2. **DINOv3 backbone + ArcFace head, full fine-tune on DogFaceNet.** More effort (larger model, more compute), more flexible (not pre-biased to WildlifeDatasets' species mix). Gives us a path beyond dogs (DINOv3 is species-agnostic).
3. **Train-from-scratch or ImageNet-init ResNet50** (current). Cheapest, most compatible, but a 2020 recipe.

**ONNX export realities:**
- **MiewID EfficientNetV2**: straightforward export. This is the lowest-friction path.
- **MegaDescriptor Swin-L**: exportable with opset 17+; some `Concat`-shape-inference issues historically (microsoft/Swin-Transformer#89 style), usually resolvable. Size is the real concern — Swin-L is 228M params, heavier than what Immich's CPU ML worker typically handles.
- **DINOv3 ViT-B / ConvNeXt**: ConvNeXt variant is the easiest export target. ViT-B exports cleanly too; has been done for DINOv2 at `sefaburak/dinov2-small-onnx` on HF, and DINOv3 upstream uses the same export hooks.
- **BioCLIP 2 ViT-L/14**: standard CLIP-style export, known good.

**Verdict:** Foundation-model init is not optional in 2026 for a small-dataset re-ID task. Starting with MiewID weights (if license permits for pet use) or MegaDescriptor weights and fine-tuning is strictly better than ResNet50-ImageNet, and the effort is modest.

---

## Top Recommended Experiments (Ranked by Expected ROI)

Ranked conceptually. Not PR recommendations; just research directions.

### Tier 1 (probably worth your time)

1. **Swap ResNet50 → MiewID EfficientNetV2-M backbone, keep ArcFace head, keep 512-d projection.**
   - Why first: largest literature-supported gain (MiewID's own ablations show 12.5% avg Top-1 over per-species training; 19.2% over MegaDescriptor on unseen species). EffNetV2 exports cleanly to ONNX. Preserves Immich 512-d contract via a linear projection head.
   - Risk: license check (MiewID weights are under a wildlife-research-friendly license; verify it's usable for pet applications).

2. **Swap ArcFace → Sub-center ArcFace with dynamic margins (K=3).**
   - Why: directly addresses DogFaceNet's label noise + long-tail identity distribution. Nearly free implementation cost (small change to `losses.py`). Paired with experiment 1, matches the MiewID recipe exactly.

3. **Fine-tune MegaDescriptor-T-224 or -S-224 on DogFaceNet with ArcFace head.**
   - Why: the smaller MegaDescriptor variants are the ONNX-deployable animal-specific pretrains. Smaller than Swin-L, still domain-matched. Compare head-to-head with experiment 1 to choose a winner.

### Tier 2 (interesting if Tier 1 lands)

4. **AdaFace as the loss, backbone-held-constant.** Clean apples-to-apples test of whether quality-adaptive margin helps on pet-photo quality variance. Single-loss, easy A/B.
5. **DINOv3-ViT-B (distilled) as backbone, LoRA fine-tune with ArcFace head on DogFaceNet.** Future-proof path beyond dogs; more compute-expensive. Worth running once to establish upper bound.
6. **Re-enable keypoints _only_ as a dense-correspondence STN** (learned affine alignment, not a hand-picked 4-point crop). Likely still not worth it, but this is the one alignment approach the 2024+ literature hasn't declared dead for re-ID.

### Tier 3 (speculative / low-priority)

7. **Grounding DINO as a pseudo-label generator** to expand beyond dogs (cats first) without manual annotation.
8. **BioCLIP 2 as a species-prior branch** in a two-branch model (species features + individual features, concatenated). Probably overkill for a pet-only product.
9. **RF-DETR-nano detector swap.** Detection accuracy gains are marginal for our current workload; pursue only if detection recall is demonstrably limiting end-to-end metrics.

### Anti-recommendations (don't bother)

- Chasing YOLOv12 over YOLO11 — flash-attention path is CPU-hostile, gains are within noise for single-class dog detection.
- CosFace, CurricularFace, ElasticFace, UniFace as drop-ins — small gains vs the backbone-swap opportunity cost.
- CLIP / OpenCLIP as direct feature extractor — known to underperform animal-specific pretrains by 20+ pp Top-1 (per WildlifeDatasets benchmarks).
- Bringing back the current 4-landmark keypoint crop. The literature is consistent that this hurts in the animal-face regime.

---

## Uncertainty / Claims to Verify

- **MiewID license for non-wildlife / pet use** — the paper describes it as released for wildlife research; commercial pet-product usage terms should be checked before shipping.
- **Exact ONNX opset required for MegaDescriptor Swin-L clean export** — claimed opset 17+ based on general Swin export practice; haven't verified on MegaDescriptor weights specifically.
- **DINOv3 commercial license applicability** — released under "commercial license" per Meta blog (Aug 2025); exact terms for a self-hosted Immich integration should be read.
- **PetFace weights availability** — the dataset is released; checkpoint availability is less clear. Their paper reports ArcFace winning over alternatives at their scale (257K individuals), which is informative but may not generalize to DogFaceNet's ~1.4K-dog scale.
- **Actual delta size of projection-head regret** when compressing 1536-d / 2048-d → 512-d for Immich interop. Plausibly 1–3 pp Top-1 drop based on general compression-head literature, but haven't seen it measured on MegaDescriptor specifically.

---

## Key References

- MegaDescriptor / WildlifeDatasets: Čermák et al., WACV 2024 Best Paper — https://arxiv.org/abs/2311.09118
- MiewID multispecies re-ID: Otarashvili et al., Dec 2024 — https://arxiv.org/abs/2412.05602
- WildlifeReID-10k: Adam et al., CVPRW 2025 — https://arxiv.org/abs/2406.09211
- PetFace: ECCV 2024 Oral — https://arxiv.org/abs/2407.13555
- DINOv3: Meta AI, Aug 2025 — https://arxiv.org/abs/2508.10104
- BioCLIP 2: NeurIPS 2025 — https://imageomics.github.io/bioclip-2/
- AdaFace: Kim et al., CVPR 2022 — http://cvlab.cse.msu.edu/project-adaface.html
- Sub-center ArcFace: Deng et al., ECCV 2020 — https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
- Immich buffalo_l: https://huggingface.co/immich-app/buffalo_l
- Grounding DINO: Liu et al., ECCV 2024 — https://arxiv.org/abs/2303.05499
- RF-DETR / 2026 detector survey: Roboflow — https://blog.roboflow.com/best-object-detection-models/
- YOLOv12 (NeurIPS 2025): https://github.com/sunsmarterjie/yolov12
