# Dog detection & identification — consolidated implementation plan (on-device, Immich)

Clean, end-to-end plan that combines the design, dataset strategy, training & evaluation recipes, deployment (on-device), personalization, and fallback label-work. I’ve folded in the earlier options (dog-specific models vs. fine-tuning human face models) and an experiment plan so you can pick the evidence-based winner.

---

# 1 — Executive summary

* Goal: detect dogs in photos and identify *individual dogs* (label them across a library) on-device, mirroring Immich’s people pipeline (detect → crop/align → embed → cluster → user verify/name).
* Strategy: use dog-specific detectors and a dog-identity embedding model as the primary route (best final accuracy). Run a controlled experiment to measure whether *initializing* the embedding backbone from Immich’s human face weights speeds training or improves accuracy; do **not** expect human face detector/landmark pipelines to work reliably without heavy modification.
* Keep all inference and personalization on-device; use public datasets for pretraining and leverage in-app semi-supervised clustering + user confirmation for household fine-tuning.

---

# 2 — High-level architecture (flow)

1. **Detector stage** — body detector (dog present) + head/face sub-detector (if found, prefer head crop).
2. **Crop & normalization** — crop with context padding; if head landmarks available, do light alignment; otherwise use learned robustness (no strict human-style alignment).
3. **Embedding network** — metric-learning model that emits fixed-dim vector (128–256d).
4. **Clustering / ID assignment** — DBSCAN/HDBSCAN or nearest-prototype matching with per-user thresholds.
5. **UI loop** — suggested clusters offered to user for confirmation/merge/split (same UX pattern as people).
6. **Personalization** — on-device fine-tuning or last-layer adapters using user-confirmed samples; update thresholds per household.

---

# 3 — Models: recommendations & roles

Detector family (on-device friendly)

* **Body detector**: lightweight YOLOv8n / YOLOv5n / MobileNet-SSD or RT-DETR tiny. Trained on COCO/OpenImages/Stanford. Output: dog body boxes.
* **Head detector**: smaller detector specialized for head/face/head-region boxes trained on DogFaceNet/StanfordExtra/Oxford masks derived boxes. If a head detector is unavailable, detect body → heuristic crop top 40–60%.

Embedding (identity)

* Backbone: MobileNetV3-Large / EfficientNet-Lite / MobileFaceNet as on-device student.
* Teacher (offline): ResNet50-ArcFace or larger ArcFace variant for distillation.
* Loss: ArcFace (preferred) or CosFace + optional triplet loss auxiliary.
* Output dims: **128 or 256** (tradeoff: 128 smaller on device, 256 better separability).
* Inference format: TFLite (with NNAPI/Metal delegate) and ONNX for other acceleration stacks. Support int8 quantized models (QAT workflow).

Optional: lightweight breed classifier branch (multi-task) trained from Stanford Dogs for optional breed tags.

---

# 4 — Datasets & exact usage (how we will maximize available data)

| Dataset                                            |                                                  Purpose | Specific usage                                                                                   |
| -------------------------------------------------- | -------------------------------------------------------: | ------------------------------------------------------------------------------------------------ |
| **COCO (dog class)**                               |                                 Detector pretrain (body) | Large, diverse scenes for body detector.                                                         |
| **Open Images (dog, mammal)**                      |                       Detector pretrain + hard negatives | More varied contexts; mine false positives for hard negatives.                                   |
| **Oxford-IIIT Pets**                               |            Detector refinement, segmentation→tight boxes | Convert masks→boxes for tight crops/occlusion handling.                                          |
| **Stanford Dogs**                                  |               Detector refinement, breed-aware negatives | Many breeds, high variety—good for robustness and intra-breed negatives.                         |
| **StanfordExtra**                                  |                          Head keypoints & alignment data | Derive head boxes and small landmark supervision.                                                |
| **DogFaceNet (and similar identity datasets)**     |                         **Primary identity supervision** | Identity-labeled face/head crops for metric learning (embedding training & verification tuning). |
| **Flickr-dog / PetFace / other public re-ID sets** |                                 Supplement identity data | Add more unique dogs/poses.                                                                      |
| **In-app user photos (unlabeled)**                 | Self-supervised pretraining & semi-supervised clustering | On-device augmentations, pseudo-labels, personalization data (never leaves device).              |
| **In-app user photos (confirmed)**                 |                                    Household fine-tuning | Small, high-value labeled set per household for adapter fine-tuning and threshold tuning.        |

Data maximization strategies

* Convert masks to boxes (Oxford) to create more tight head crops.
* Use StanfordExtra keypoints to synthesize head crops and create alignment pseudo-labels.
* Self-supervised pretrain on large unlabeled dog images (SimCLR / MoCo) to learn invariances.
* Pseudo-labeling / high-confidence clustering on public unlabeled sets to expand identity data while filtering low-confidence clusters.
* Use copy-paste augmentation and synthetic viewpoint augmentation (rendered backgrounds, style transfer) to increase side/back/partial views.

---

# 5 — Training pipeline (stage-by-stage, no time guesses)

Detector training

1. **Stage D0 — Pretrain**: COCO + OpenImages; heavy augment (mosaic, blur, lighting) to learn robust dog/body boxes.
2. **Stage D1 — Head specialization**: train head detector using DogFaceNet crops, StanfordExtra-derived head boxes, Oxford converted masks. Multi-task outputs (body vs head) are recommended.
3. **Stage D2 — Hard negatives**: mine false positives (toys, statues) and retrain to improve precision while preserving recall.

Embedding training

1. **Stage E0 — Self-supervised warmup**: SimCLR / MoCo on Stanford + Oxford + in-app unlabeled dog crops (learn pose/lighting invariance).
2. **Stage E1 — Supervised metric learning**: use DogFaceNet (IDs) with ArcFace + strong augmentation (random crops, occlusion, motion blur). Use breed-aware sampling for hard negatives (sample negatives from same breed).
3. **Stage E2 — Distillation**: Train big teacher (e.g., ResNet50 ArcFace), then distill to mobile student (MobileNetV3/EfficientNet-Lite).
4. **Stage E3 — Semi-supervised expansion**: cluster unlabeled data (high confidence only) → create pseudo-identities → soft-label fine-tune. Reject noisy clusters.
5. **Stage E4 — On-device personalization**: adapt last block or add small adapter layers trained on user-confirmed images (never leaves device).

Model compression & export

* Quantization-Aware Training (QAT) followed by TFLite int8 conversion or ONNX quantization; run per-device tests with delegates (NNAPI/Metal/CUDA).
* Aim for student model size in the **\~4–20 MB** range (configurable). Keep embedding computation cost similar to current human face model.

---

# 6 — Experiment plan: human face model fine-tune vs dog-specific (A/B)

Purpose: verify whether initializing from Immich’s InsightFace/ArcFace weights helps.

**Setup**

* Data splits: same train/val/test across DogFaceNet + small household validation set. Ensure no identity leakage.
* Models to compare:

  * **Baseline 1 (Dog-specific)**: ImageNet init → train on dog datasets (recommended default).
  * **Baseline 2 (Human init)**: InsightFace/ArcFace weights from Immich → fine-tune on dog datasets (replace alignment head, re-train first layers as needed).
  * **Baseline 3 (Teacher→Student Distill)**: ResNet50 teacher (ArcFace) trained on dog IDs → distill to mobile student.
* Metrics: closed-set top-1 / top-5, verification TAR @ FAR={1e-2,1e-3,1e-4}, cluster purity, pairwise precision/recall.
* Decision rule: prefer the approach that yields higher verification TAR at low FAR (1e-3) *and* better cluster purity on household validation. If human init beats dog-specific by a sustained margin (e.g., +3–5% top-1 and better TAR\@FAR without heavy alignment hacks), adopt human init; otherwise choose dog-specific.

---

# 7 — Evaluation & acceptance metrics (targets & thresholds)

Detector targets

* **mAP (IoU=0.5) for body boxes**: target ≥ **0.85** on held-out public validation.
* **Head detector recall (head size >64px)**: target ≥ **0.90**.

Embedding / ID targets (initial public dataset)

* **Closed-set top-1** (DogFaceNet): target **≥ 80%** (realistic baseline).
* **Verification TAR @ FAR=1e-3**: target **≥ 0.85** on curated dataset.
* **Cluster purity (DBSCAN at chosen eps)**: target **≥ 0.90** on curated sets.

Household / in-the-wild targets (realism)

* After personalization: closed-set top-1 **≥ 85–90%** on household validation when there are reasonable photos (clear shots).
* Cluster precision prioritized over recall: prefer conservative linking to avoid mislabeling different dogs.

DBSCAN / clustering suggestions

* Embedding dim = 128 or 256.
* DBSCAN eps: tune per household but start with **eps ≈ 0.45–0.65** (cosine distance transformed as 1−cos), min\_samples = 3. Provide per-user threshold slider in UI.

Acceptance criteria for production rollout

* Detector recall and head detection meet targets on public val set.
* Embedding verification TAR\@FAR=1e-3 ≥ 0.85 on curated data and cluster purity ≥ 0.9 on a small set of household libraries.
* On-device inference latency and memory within device constraints (see next section).

---

# 8 — On-device deployment & optimization

Formats & delegates

* Export TFLite models (float16 or int8) and ONNX (for other stacks). Test with NNAPI, Metal, CUDA, and fallback CPU.
* Provide a small model (student) for phones & Raspberry/Immich server edge nodes; provide an optional larger model for dedicated ML nodes.

Optimizations

* **Quantization-Aware Training** for int8 accuracy preservation.
* **Knowledge distillation**: train mobile student supervised by a stronger teacher.
* **Pruning & weight clustering** as last step if needed.
* **Batching & caching**: cache embeddings for images already processed; do incremental processing for new photos only.

Resource budgets (guidance only — no time estimates)

* Student model target: **4–20 MB** TFLite.
* CPU inference: < \~50ms per crop on modern mobile CPU (target); accelerate with NPU when available.
* Memory: keep peak memory low; run detection first and only run embedding on crops exceeding detection confidence threshold.

Privacy

* All personalization/fine-tuning runs on device. Only metadata (if opted in) can be exported — but default is local only. Use the same privacy model Immich uses for people.

---

# 9 — UI / UX & personalization flow

Clustering UX (mirrors people)

* Run clustering in background (on device or local ML machine). Present suggested clusters in a “Pets” panel.
* Ask user to confirm a cluster name (dog name) — minimal friction (single tap merges).
* Active learning: show 6–12 candidate photos and ask “Are these the same dog?” for ambiguous clusters — one answer yields many pairwise labels.

Fine-tuning UX

* After user confirms a small seed set (5–10 high quality photos per dog), run a light adapter tune (last block or linear classifier) on device while charging. Update local thresholds to that household.

Search & filters

* Support text search (CLIP fallback) like “my golden retriever”. Combine breed classifier (optional) + embedding nearest neighbors.

Confidence & privacy indicators

* Show label confidence and allow easy split/merge. Make clear all models and fine-tuning are local.

---

# 10 — Minimal labeling fallback (only if accuracy stalls)

Start minimal, escalate only if needed:

1. **Head boxes**: annotate 1–3k head boxes covering hard cases (tiny heads, profile, backlit). This typically unlocks large gains.
2. **Household seeds**: ask users for 5–10 best photos per dog (cheap, high ROI).
3. **Viewpoint tags**: add 1–2k viewpoint annotations (frontal/¾/profile/back) for a targeted auxiliary head.
4. **Occlusion/quality flags**: tag 1–3k images for blur/occlusion to weight training.
5. **Landmarks (last resort)**: annotate \~1k landmarked heads to train a small dog-landmark network if alignment is still the bottleneck.

Active learning collection: prioritize labeling where model is uncertain (close nearest neighbors with distance in ambiguous band) to maximize label value.

---

# 11 — Risks & mitigations

* **Risk:** public datasets are cleaner than in-the-wild household photos → lower real performance.
  **Mitigation:** early household validation set, strong augmentation, semi-supervised clustering, and in-app seed confirmations.

* **Risk:** human face models mislead rather than help.
  **Mitigation:** run the AB experiment with clear decision rule; if human init doesn’t help, default to dog-specific.

* **Risk:** heavy false positives (toys, statues).
  **Mitigation:** hard-negative mining from OpenImages and in-app incorrect detections.

* **Risk:** privacy concerns.
  **Mitigation:** keep all personalization on device; explicit opt-in for any server analytics.

---

# 12 — Deliverables & concrete outputs I can produce for your team (pick now)

I can produce any or all of the following artifacts (no scheduling implied — I’ll produce them now in this message if you want):

* A runnable **experiment recipe** (exact training commands, dataset splits, augmentations, loss configs, batch sizes) for the 3-way comparison (dog-only, ImageNet init, human-init).
* **TFLite export checklist** with QAT and delegate test cases.
* **Evaluation notebook** (code + metrics) to compute TAR\@FAR, cluster metrics, and produce ROC/precision-recall curves.
* A compact **annotation spec** (label formats, CSV schemas, recommended annotation UI flow) for fallback labeling tasks (head boxes, viewpoints).
* A **UI mock** and user prompts for cluster confirmation & active learning.

Tell me which artifact(s) you want first and I’ll produce them immediately in full.
