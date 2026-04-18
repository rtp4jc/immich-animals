# immich-animals Codebase Audit — Consolidated Report

**Date:** 2026-04-16
**Audience:** Maintainer evaluating readiness for adoption into the Immich open-source organization
**Scope:** Training pipeline, benchmarking/evaluation, code structure & extensibility, SOTA alignment, Immich model-I/O contract. *Excludes* the temporary hijack integration (copy_models.sh, REPLICATION.md, docker-compose.dogs.yml).

**Integration model assumed:** A new `AnimalRecognizer` / `AnimalDetector` pair living as a **sibling module** alongside Immich's existing `FaceRecognizer` / `FaceDetector`, not a drop-in replacement for the `buffalo_l` slot. This is the framing that matches how Immich's ML service is structured and how a native animal-recognition feature would ship upstream.

Full findings live in the sub-reports:
- [`audit-training.md`](./audit-training.md) — training pipeline production-readiness
- [`audit-benchmarking.md`](./audit-benchmarking.md) — benchmark & evaluation correctness
- [`audit-structure.md`](./audit-structure.md) — structure, conventions, extensibility, testing
- [`audit-sota.md`](./audit-sota.md) — 2026 SOTA conceptual comparison
- [`audit-immich-interface.md`](./audit-immich-interface.md) — Immich ONNX I/O contract

---

## Overall Verdict

**Not yet production-grade, but every blocker is a small PR.** The project already looks like a real ML repo — Protocol-based abstractions, a three-stage package layout, pre-commit + ruff in CI, a reasonable test pyramid, a central constants module. What's missing is ML-discipline polish:

1. **Two numerically wrong benchmark metrics** that silently over-report the operating point Immich would care about (TAR@FAR threshold-range bug + pair duplication).
2. **Shipped baseline is broken** — default embedding LRs in `animal_id/embedding/config.py` are `1e-8` / `1e-9`, effectively a no-op. A new user running the defaults does not get the advertised model.
3. **No reproducibility** — no global seed, no dep pins, no git-SHA capture. Two identical invocations produce different models.
4. **Ablation ergonomics are poor** — swapping loss, backbone, or dataset touches 4–8 files because `ArcFaceLoss` is hardcoded at model-instantiation time and species assumptions leak into 8+ files.
5. **Public-API naming is off-brand for an Immich project** — `DogEmbeddingModel`, `DogIdentityDataset`, `AmbidextrousAxolotl` (the main pipeline class) are unserious names for a system claiming "animal identification."
6. **Backbone is the one genuinely stale component** — `audit-sota.md`: ResNet50-ImageNet is 2019-era. 2024–2025 animal re-ID (MegaDescriptor WACV'24 Best Paper, MiewID 2024, WildlifeReID-10k CVPR'25) has converged on animal-pretrained Swin / EfficientNetV2 with sub-center ArcFace + dynamic margins, beating ResNet50+ArcFace by 20–70 pp Top-1 head-to-head. This is the highest-ROI single change available.

None of this is architectural rot. An engineer from Immich can credibly land all of it in ~4 weeks.

---

## The Single Hard Interop Constraint

The Immich interface deep-dive (`audit-immich-interface.md`) enumerated seven mismatches (M1–M7) between our ONNX outputs and Immich's face pipeline. **In a sibling-module integration, only one of them is a real constraint:**

> **512-dimensional, L2-normalized, cosine-compared embeddings.**

That constraint exists because Immich's downstream clustering (DBSCAN) and storage (`pgvector` column) are shared across modalities. Diverging here would require schema changes in Immich, which is a much bigger upstream lift than wrapping a new ONNX model.

Everything else in M1–M7 — input resolution (112 vs 224), color order (BGR vs RGB), normalization constant (`(x−127.5)/127.5` vs `/255`), number of keypoints (5 vs 4 vs 0), detection output dict layout, model-registry name — is **module-internal** and negotiable with whoever reviews the upstream PR. A new sibling module brings its own preprocessing and own output-dict shape; Immich just needs to call it correctly.

Our current pipeline already satisfies the one hard constraint: 512-d, L2-normalized, cosine-matched. So the interface picture is actually friendlier than the raw M1–M7 list makes it look. **Don't burn priority on matching Immich's exact tensor shapes.**

### The keypoint question, settled

The SOTA audit independently confirms what your instinct already said: humans use 5-landmark `norm_crop` because ArcFace was trained for it and human head-pose variation is rigid. Animal re-ID has moved decisively to **crop-only global embeddings** (MegaDescriptor, MiewID, WildlifeReID-10k winners) because non-rigid animal pose variation exceeds what a small landmark set can correct. Our keypoint stage is correctly disabled and should stay that way. No 5-landmark retraining needed for interop, no keypoint work at all.

---

## Cross-Cutting Themes

The five deep-dives independently surfaced overlapping root issues. Overlap indicates they're real.

### Theme A — Reproducibility is absent, not incomplete
No `torch.manual_seed` / `np.random.seed` / `cudnn.deterministic` anywhere. DataLoader has `shuffle=True` with no `generator=` or `worker_init_fn` (`scripts/train_master.py:357-368`). Dataset converters use `random.seed(42)` but `os.listdir` ordering is filesystem-dependent, so the split itself is fragile across hosts (`animal_id/embedding/dataset_converter.py:76-91`). No git-SHA, `pip freeze`, or hardware info captured in run directories. Dependency floors are `>=` with no lock file; `ultralytics` ships breaking API changes in 8.3.x minors.

### Theme B — Two disagreeing metric implementations
`animal_id/benchmark/evaluator.py` uses a binary search over `[0, 1]` for TAR@FAR thresholds — but cosine similarities live in `[-1, 1]`. At FAR=0.001 the search can clip to ~0 and silently over-report TAR (`audit-benchmarking.md` §F1). A second TAR@FAR lives in `animal_id/benchmark/metrics.py` with a different operator (`>` vs `>=`) and a different pair-sampling policy. Training-time validation metrics and benchmark-time headline metrics are **not comparable** because they come from the two different implementations.

### Theme C — `ArcFaceLoss` hardcoded at the model seam, not the config seam
`animal_id/embedding/models.py:113` instantiates `ArcFaceLoss` directly. The outer loss (`nn.CrossEntropyLoss(label_smoothing=0.1)`) is also hardcoded at `animal_id/embedding/trainer.py:26`. Swapping to Sub-center ArcFace / AdaFace / CosFace requires editing 4 files. This becomes acute once SOTA recommendations land.

### Theme D — Species assumptions bleed through the public API
`DogEmbeddingModel`, `DogIdentityDataset`, `DogIdentificationSystem`, `detect_dogs()` are public classes/methods with generic implementations (`audit-structure.md` §3.1). Adding a new species touches 8+ files, mostly for renames and a hardcoded `"dog"` string match in the detection converter (`animal_id/detection/dataset_converter.py:125, 217, 329`). The main pipeline class is `AmbidextrousAxolotl` — unsearchable and hard to justify in a review.

### Theme E — W&B integration is asymmetric
`WandBLogger` exists and is well-structured, but it's wired only into the pipeline benchmark, **not into any trainer**. There's no per-epoch train/val loss, no LR/grad-norm trace, no best-model artifact. A reviewer cannot compare two embedding runs in W&B.

### Theme F — No held-out test set
`identity_val.json` is used both for per-epoch early-stopping (`animal_id/embedding/trainer.py:131`) and for the headline pipeline benchmark (`scripts/train_master.py:83-223`). Reported numbers are validation numbers. Standard practice is train / val / held-out test — we have two of three.

### Theme G — Regression detection does not exist
`evaluator.save_results()` is never called; `outputs/temp_ground_truth.json` is deleted at the end of the run. The only persisted artifacts are W&B run metadata and PNG visualizations. No on-disk baseline JSON for a future PR to diff against. CI runs ruff + pytest — no benchmark gate.

### Theme H — The embedding backbone is the one genuinely stale component
`audit-sota.md`: ResNet50-ImageNet was reasonable in 2019 but is clearly surpassed by animal-pretrained Swin/EffNetV2 (MegaDescriptor, MiewID) and by DINOv3/BioCLIP 2 initialization. Sub-center ArcFace with dynamic margins is the MiewID recipe and directly addresses DogFaceNet's long-tail / label-noise profile. This is the single highest-ROI experimental direction.

---

## Top 15 PR-Sized Recommendations (Prioritized)

Each item is ≤1 day of focused work. Ordered so earlier PRs unblock later ones.

### P0 — Correctness & adoption blockers

| # | PR title | Files | Source |
|---|----------|-------|--------|
| 1 | ✅ **Fix TAR@FAR threshold range bug** — replace binary search with `np.quantile`, unify the two implementations | `animal_id/benchmark/evaluator.py:305-329`, `animal_id/benchmark/metrics.py:108-157` | bench F1/F4 | b09e599 |
| 2 | ✅ **Fix TAR@FAR pair duplication** — stop counting both `(A,B)` and `(B,A)` | `animal_id/benchmark/evaluator.py:273-285` | bench F2 | b09e599 |
| 3 | ✅ **Restore a sane embedding baseline** — the `1e-8 / 1e-9` LRs in `animal_id/embedding/config.py:19-22` are a debugging remnant; replace with values from a known-good run and add a test asserting `HEAD_LR >= 1e-5` | `animal_id/embedding/config.py`, `tests/unit/test_config.py` (new) | train PR 3 | a806c74 |
| 4 | ✅ **Rename dog-specific public API** — `DogEmbeddingModel` → `AnimalEmbeddingModel`, `DogIdentityDataset` → `IdentityDataset`, drop `DogIdentificationSystem` alias | `animal_id/embedding/models.py`, `animal_id/common/datasets.py`, `animal_id/benchmark/evaluator.py`, all callers | struct PR-1 |
| 5 | ✅ **Rename `AmbidextrousAxolotl` → `AnimalPipeline`** and rename the module file | `animal_id/pipeline/animal_pipeline.py` + all imports | struct PR-2 |

### P1 — Reproducibility, experiment hygiene, ablation unlock

| # | PR title | Files | Source |
|---|----------|-------|--------|
| 6 | **`animal_id/common/seeding.py`** — `set_seed(seed, deterministic=False)` seeding Python / NumPy / torch CPU+CUDA, DataLoader `generator=` + `worker_init_fn`, pass `seed=` into Ultralytics | cross-package | train PR 1 |
| 7 | **Wire W&B into `EmbeddingTrainer`** — per-epoch train/val loss, LR, grad-norm, TAR@FAR; upload `best_model.pt` as `wandb.Artifact`; match YOLO trainers' `project=/name=` | `animal_id/embedding/trainer.py`, YOLO trainers | train PR 2 |
| 8 | **Hydrate `TRAINING_CONFIG` into a dataclass + YAML loader** — pull `label_smoothing`, `grad_clip`, `phase2_warmup_epochs`, `arcface_s`, `arcface_m`, augmentation policy, loss type, backbone onto a single `EmbeddingConfig` with `--config path.yaml` | embedding package | train PR 4 |
| 9 | **Factor the head/loss behind a `HEAD_TYPE` config** — add `SubCenterArcFace` (SOTA recipe), `CosFaceLoss`, and a `MarginHead` base, dispatch in `AnimalEmbeddingModel.__init__` instead of hardcoding `ArcFaceLoss` at line 113 | `animal_id/embedding/{losses.py,models.py,config.py}` | train PR 11 + sota §4 |
| 10 | **Introduce a held-out `identity_test.json`** — stop using val for both early-stopping and headline numbers. Use val for selection, test for the reported MRR/top-k/TAR@FAR artifact | `animal_id/embedding/dataset_converter.py`, `scripts/train_master.py` | bench §3 |
| 11 | **Full-state checkpointing + resume** — save optimizer/scheduler/RNG/epoch/phase/num_classes; accept `resume_from=` in `EmbeddingTrainer.train` | `animal_id/embedding/trainer.py` | train PR 6 |
| 12 | **ONNX parity test in CI** — train 1 epoch on mock data, export, assert `np.allclose` between PyTorch and ONNX outputs; extend to assert embedding is 512-d, L2-normalized (the one hard interop invariant) | `tests/integration/test_onnx_parity.py` (new) | train PR 9, immich §M |

### P2 — SOTA-track experiments (measurable against the new held-out test set)

These are experiments, not refactors, but each is self-contained and measurable against #10.

| # | PR title | Files | Source |
|---|----------|-------|--------|
| 13 | **Backbone swap — MiewID EfficientNetV2-M + Sub-center ArcFace.** Highest single-change expected lift. Clean ONNX export. Verify license allows pet use before shipping | `animal_id/embedding/backbones.py`, new `miew_backbone.py`, `animal_id/embedding/losses.py` | sota rec 1, 2 |
| 14 | **Backbone swap — DINOv3-ConvNeXt-S with ArcFace head.** Future-proof, species-agnostic, higher ceiling but more compute | `animal_id/embedding/backbones.py` | sota rec 3 |

### P3 — Regression safety net

| # | PR title | Files | Source |
|---|----------|-------|--------|
| 15 | **Persist a benchmark artifact JSON + `scripts/diff_benchmarks.py` + CI gate on ≥1% regression** — call `evaluator.save_results()`, stop deleting `temp_ground_truth.json`, add a diff script, wire into CI | `scripts/train_master.py`, `animal_id/benchmark/evaluator.py`, `.github/workflows/python-package-conda.yml` | bench §6 |

### Deferred but worth doing

Not cut because they're unimportant — cut because they pay off after the above:

- **Repair detection precision/recall** — `IdentityLoader._load_base_validation` (`animal_id/common/identity_loader.py:62-74`) filters to labelled dogs only, so false_positives ≡ 0 and precision ≡ 1.
- **Unify YOLO trainers under `BaseYoloTrainer`** (~70 lines of duplication) — train PR 8, struct PR-5.
- **Replace corrupt-image recursion** in `animal_id/common/datasets.py:82-86` with skip+metric.
- **Delete `animal_id/common/inference.py`** (fully stubbed, never imported).
- **Add `mypy` step to CI** with permissive baseline to match Immich ML parity.
- **Add `TypedDict` detection/keypoint schemas** in `animal_id/pipeline/models.py`.
- **Remove `pathlib2` dependency; move `wandb`/`transformers` to optional `[train]` extras.**
- **Pin `environment.yml` deps** (or replace with `uv.lock`).
- **Add `--force` guard to destructive dataset converters** that `shutil.rmtree` their output dirs.
- **Bootstrap confidence intervals + per-identity breakdown in the benchmark artifact.**

---

## What's Good — Don't Regress These

- **Protocol-based pipeline abstractions** (`animal_id/pipeline/models.py`) are the right shape, cleanly satisfied by ONNX wrappers and mocks. Don't replace with inheritance.
- **Central `constants.py`** is the single source of truth for paths — preserve as the project grows.
- **Integration tests that run 1 epoch end-to-end** on mock datasets are the highest-value tests in the repo and the reason ablations can be reviewed at all.
- **The numbered script workflow** (`01` → `17`) is friendly to new contributors and works well with `train_master.py` exposing the same steps as callable functions.
- **Pre-commit + ruff in CI** is set up correctly; don't weaken it.
- **Three-way separation** `detection/` / `keypoint/` / `embedding/` with parallel `trainer.py` / `dataset_converter.py` files is a clean mental model even with the duplication.
- **Cosine / L2-normalized / 512-d embedding contract** — the one contract choice that's already right for Immich interop.
- **Keypoint-free inference pipeline** — matches animal re-ID SOTA; don't revive 5-point alignment even though Immich's face pipeline uses it.

---

## Suggested Merge Order

For an engineer picking this up:

1. **Week 1, day 1–2** — PR 3 (LR baseline) + PR 1 (TAR@FAR range) + PR 2 (pair dedup). One-line / near-one-line fixes with huge trust impact.
2. **Week 1, day 3–5** — PR 4 + PR 5 (rename dogs / `AmbidextrousAxolotl`) + PR 6 (seeding). Unblock public-API work and foundation-of-reproducibility.
3. **Week 2** — PR 7 (W&B in trainer), PR 8 (config dataclass), PR 9 (loss factory with Sub-center ArcFace), PR 10 (held-out test), PR 11 (resume). The ML-discipline layer that makes PR 13/14 measurable.
4. **Week 3** — PR 12 (ONNX + embedding-contract parity test), PR 15 (regression artifact + CI gate). Safety net.
5. **Week 4+** — PR 13 (MiewID backbone) then PR 14 (DINOv3) as A/B experiments against the baseline captured in #15.

By end of ~4 engineer-weeks the repo is genuinely defensible as an Immich-adoptable foundation **and** is running a real experimental loop. The remaining deferred items can land opportunistically.

---

## One-Line Summary for Reading Aloud

ResNet50 is stale, TAR@FAR is buggy, LR defaults are broken, and the name `AmbidextrousAxolotl` has to go — but the interface story is friendlier than it first looked (one hard constraint: 512-d L2-normalized cosine; everything else is internal), and four weeks of focused work closes the rest.
