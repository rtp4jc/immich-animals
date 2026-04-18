# Benchmarking & Evaluation Audit — `immich-animals`

**Auditor perspective:** senior ML engineer from Immich OSS evaluating whether these benchmarks are trustworthy enough to gate merges or releases that hijack Immich's face-recognition pipeline.

**Scope of audit:**
- `animal_id/benchmark/evaluator.py` (pipeline-level eval, MRR, top-k, TAR@FAR)
- `animal_id/benchmark/metrics.py` (component-level embedding eval used at training)
- `animal_id/benchmark/visualizer.py`
- `scripts/13_run_full_pipeline.py` + `scripts/train_master.py::run_full_pipeline_benchmark`
- `animal_id/pipeline/ambidextrous_axolotl.py`
- `animal_id/common/identity_loader.py`, `animal_id/common/datasets.py`, `animal_id/embedding/dataset_converter.py`
- `animal_id/tracking/wandb_logger.py`, `.github/workflows/python-package-conda.yml`, `tests/unit/benchmark/test_metrics.py`

---

## 1. Executive Summary

**Verdict: the benchmarking suite is not ready for an Immich merge gate.** It produces numbers, logs them to W&B, and draws a summary PNG, but the numerical core has at least two correctness bugs, evaluation is run on the same split used for model selection, and there is no artifact written to disk that a PR reviewer can diff. Two implementations of TAR@FAR exist (`benchmark/evaluator.py` and `benchmark/metrics.py`) and they do not agree on protocol, definition of positives/negatives, or threshold search space.

**Top severity findings (detail in §2–§5):**

| # | Severity | Finding |
|---|----------|---------|
| F1 | **High** | `BenchmarkEvaluator._binary_search_threshold` searches threshold in `[0, 1]`, but cosine similarities from the pipeline are in `[-1, 1]`. When the true threshold for a target FAR is negative, the search returns `~0` and TAR is silently overestimated. (`evaluator.py:309`) |
| F2 | **High** | TAR@FAR pair construction in `evaluator.py:273-285` counts every ordered query→gallery pair independently, so each pair `(A, B)` is included twice (`A→B` and `B→A`). This inflates `len(diff_scores)` and `len(same_scores)` by 2× without adding information and interacts poorly with per-query sampling assumptions. |
| F3 | **High** | There is no held-out test set. `data/identity_val.json` is the split the embedding model was selected on (via `EmbeddingTrainer.validate()` → `evaluate_embedding_model`), and it is also the set reported by `run_full_pipeline_benchmark`. The headline benchmark numbers are validation-set numbers. |
| F4 | **High** | Two different TAR@FAR implementations exist (training-time `metrics.py:calculate_tar_at_far` and eval-time `evaluator.py._compute_tar_at_far`) with different pair-sampling policies, different threshold estimators (quantile vs. binary search), and a `>` vs. `>=` inconsistency. They cannot be compared. |
| F5 | **Medium** | No benchmark artifact is written by default. `run_full_pipeline_benchmark` creates and **deletes** `outputs/temp_ground_truth.json` (`train_master.py:220`). `BenchmarkEvaluator.save_results` exists but is never called. There is no stored baseline JSON to diff a PR against. |
| F6 | **Medium** | CI (`.github/workflows/python-package-conda.yml`) runs ruff + pytest only. No benchmark, no regression check, no nightly run. The three artifacts in `outputs/test_runs/` are plain text logs, not machine-readable. |
| F7 | **Medium** | Detection precision/recall are computed on the same validation set that contains only positives (`IdentityLoader._load_base_validation` filters to `item.get("identity_label")` → `if item.get("identity_label")`, i.e. keeps only labelled dogs). `detection_precision`, `detection_recall`, `non_animal_images`, and `false_positives` are therefore structurally meaningless on the standard run. |
| F8 | **Medium** | `IdentityLoader.__init__` calls `random.seed(seed)` once at construction (`identity_loader.py:19`). Any library code that subsequently touches `random` (e.g. the logger, Torch DataLoader workers) changes the downstream sample. Seeding is also not applied to `numpy.random`, which `metrics.py` uses. |
| F9 | **Medium** | No component-level eval for the detector or keypoint model is run in `run_full_pipeline_benchmark`. A detector regression and an embedding regression are indistinguishable in the reported numbers. |
| F10 | **Medium** | No confidence intervals, no per-identity breakdown, no class balance report. The benchmark prints three Top-K numbers and walks away. With ~250–300 validation identities at ~5k images this is low-sample for FAR=0.001. |
| F11 | **Low** | `AmbidextrousAxolotl.build_gallery` calls `generate_embedding` on every gallery item, which re-runs detection + keypoints on the query at `predict` time too (duplicate work, not a correctness bug). |
| F12 | **Low** | The only dataset used is DogFaceNet. There is no support for or discussion of standard re-ID benchmarks (ATRW tigers, SeaTurtleID, WildlifeReID-10k, even a DogFaceNet held-out split).  |
| F13 | **Low** | `WandBLogger` is the only persistent store of results, which makes offline diffing (the normal PR review flow) impossible without a W&B account. |

---

## 2. Metric-by-Metric Audit

### 2.1 Detection accuracy / precision / recall (`evaluator.py:192-217`)

**Verdict: correct under its assumptions, but the assumptions do not hold in practice.**

The code is fine: `detection_correct = has_animal_gt == has_animal_pred` (`evaluator.py:158`); precision and recall use the standard TP/FP/FN formulas with zero-division guards.

However:
- `has_animal_gt = item.get("identity_label") is not None` (`evaluator.py:151`). The only ground-truth source is `identity_val.json`, and `IdentityLoader._load_base_validation` filters to items with an identity label (`identity_loader.py:73`: `if item.get("identity_label")`). So on the default run every ground-truth item has `has_animal_gt=True`.
- Consequence: `non_animal_images == 0`, `false_positives ≡ 0`, `detection_precision ≡ 1.0` if there are any true positives, `detection_accuracy ≡ detection_recall`. These three printed numbers carry one bit of information between them.
- The benchmark does not ingest a negatives corpus (random ImageNet, COCO non-dog images, Oxford Pets cats) even though `data/oxford_pets` and `data/coco` exist in the repo.

**Recommendation:** drop precision from the report until a negatives set is added, or wire `IdentityLoader` to yield labelled negatives from COCO/Oxford Pets with `identity_label=None`.

### 2.2 Identity rank computation (`evaluator.py:160-171`)

**Verdict: correct, but fragile.**

```
164:                for rank, (sim_path, _) in enumerate(similar_images, 1):
165:                    sim_rel_path = str(Path(sim_path).relative_to(self.data_root))
166:                    if (
167:                        sim_rel_path in same_identity_images
168:                        and sim_rel_path != item["image_path"]
169:                    ):
170:                        identity_rank = rank
171:                        break
```

- The self-exclusion (`sim_rel_path != item["image_path"]`) is redundant because `AmbidextrousAxolotl.predict` already skips self via string equality (`ambidextrous_axolotl.py:91`) — but the string-equality check there compares absolute paths, while here comparison is via relative paths. The two self-exclusions do agree on the happy path.
- If `Path(sim_path).relative_to(self.data_root)` ever fails (e.g. the gallery contains a path outside `data_root`, which happens for `additional_identities` when run from a different CWD), this raises `ValueError` and kills the whole run. No try/except.
- Identities with only one image in the gallery will always get `identity_rank=None` and silently drop out of the MRR denominator. There is no log of how many queries were unscorable; `animal_images` is printed but `len(valid_ranks)` is not.

### 2.3 Mean Reciprocal Rank (`evaluator.py:220-225`)

**Verdict: definition is correct; population is wrong.**

```
220:        valid_ranks = [
221:            r.identity_rank for r in self.results if r.identity_rank is not None
222:        ]
223:        mean_reciprocal_rank = (
224:            np.mean([1.0 / rank for rank in valid_ranks]) if valid_ranks else 0.0
225:        )
```

The formula `mean(1/rank)` is standard. Problems:
- `identity_rank` is `None` for (a) undetected dogs and (b) singleton identities. These are conditioned out of the mean, so MRR is effectively "MRR given the detector fired and there is at least one other exemplar." This is not what most readers assume.
- No breakdown by identity: with the average dominated by identities that have many gallery exemplars, MRR is not class-balanced. A single identity with 40 images contributes 40× more than a singleton.

### 2.4 Top-K accuracy (`evaluator.py:228-231`)

**Verdict: correct.**

`correct_at_k = sum(1 for rank in valid_ranks if rank <= k)` over the same `valid_ranks`. K∈{1,3,5,10} is fine. Same caveats as MRR regarding population. `top_k_accuracy[10]` is computed but not printed (`evaluator.py:55-57` only shows 1/3/5).

### 2.5 TAR @ FAR — eval-time path (`evaluator.py:252-329`)

**Verdict: buggy. Two concrete issues and one design smell.**

Critical issue 1 — threshold search range:
```
305:    def _binary_search_threshold(
306:        self, sorted_diff_scores: np.ndarray, target_far: float
307:    ) -> float:
308:        """Binary search for threshold that achieves target FAR."""
309:        left, right = 0.0, 1.0
```

Cosine similarity is in `[-1, 1]`. The embeddings emitted by `ONNXEmbedding` are NOT guaranteed to be L2-normalised (it just returns `session.run(...)[0][0]`, `onnx_models.py:105-108`), and `cosine_similarity` in `ambidextrous_axolotl.py:84` does the normalisation internally, so scores in principle span `[-1, 1]`. Even under ArcFace, negative cosines for hard negatives are common.

If the threshold that achieves FAR=0.001 is negative (very plausible for a weak model or a hard-negative-heavy gallery), the binary search **cannot reach it**. It will return a value close to `0.0`, at which point `tar = np.mean(same_scores >= threshold)` is computed against a threshold the search never actually satisfied. The printed `TAR@FAR=0.1%` will be a TAR at a much looser operating point, silently optimistic. No assertion catches this.

Critical issue 2 — pair duplication:
```
273:        # Collect similarity pairs
274:        for query_result in detected_results:
275:            query_identity = query_result.identity_gt
276:
277:            for similar_path, similarity in query_result.similar_images:
278:                abs_similar_path = str(Path(similar_path).resolve())
279:                similar_identity = path_to_identity.get(abs_similar_path)
```

Every image in `detected_results` is a query, and `similar_images` contains every other gallery item. Cosine similarity is symmetric, so `score(A,B) == score(B,A)`; both are appended. `same_identity_scores` and `different_identity_scores` are therefore 2× larger than the unique-pair count. This does not bias the ratio-based FAR, but:
- The quantile is taken over a non-i.i.d. sample (each unique pair appears twice).
- `_binary_search_threshold` and `np.mean(same_scores >= threshold)` implicitly assume the distribution was sampled from distinct pairs.

Design smell — dead break:
```
321:            if abs(far - target_far) < tolerance:
322:                break
```
With N≈4000 negatives, `far` takes values in `{k/4000}`. Target `0.001 = 4/4000`. `tolerance=1e-6` is smaller than `1/4000 ≈ 2.5e-4`, so `break` never triggers; the loop always runs 50 iterations. Not a bug per se (50 halvings of `[0,1]` converge to ~1e-15), but reveals the author expected continuous FAR.

Further issues:
- `>=` is used in the `num_above` count (`side="left"` semantics), but the final TAR also uses `>=` on `same_scores` — consistent. OK.
- No bootstrap or CI on TAR. With ~300 same-identity pairs (typical for DogFaceNet val), the sampling error on TAR@FAR=0.001 is ±several percent.

### 2.6 TAR @ FAR — training-time path (`metrics.py:108-157`)

**Verdict: different protocol, more principled, but still ad hoc.**

- Positive pairs: *all* same-identity combinations (`metrics.py:15-19`). Good.
- Negative pairs: for N<500, enumerated exhaustively and subsampled to `2 × len(positives)`; for N≥500, rejection-sampled with a 5×attempt cap (`metrics.py:70-83`). The cap can silently return fewer pairs than requested with no log.
- Threshold: `np.quantile(neg_scores, 1 - far_threshold)` (`metrics.py:154`). This is the **correct** closed-form equivalent of what `evaluator.py` tries to do with binary search. Definitions don't match across the two files.
- TAR: `np.sum(pos_scores > threshold) / len(pos_scores)` — strict `>`, whereas `evaluator.py` uses `>=`. At a tie the two implementations disagree by `1/len(pos_scores)`.
- `np.random.seed(seed)` is set (`metrics.py:118`), but any call before this that touches `np.random` global state has already happened by the time this runs. Not a bug in isolation.
- Embeddings are moved to a torch tensor and pair scores are computed one at a time in a Python list comprehension (`metrics.py:97-104`). This is O(N²) items but sequential; a batched `F.cosine_similarity` would be ~100× faster. Not a correctness issue.

### 2.7 mAP (`metrics.py:160-178`)

**Verdict: correct.**

`average_precision_score(relevance, query_sims)` is the standard single-query AP; `relevance[i]=0` excludes self. `np.mean` over queries with at least one relevant item → mAP. Only flaw: no macro-average by identity — identities with many exemplars dominate.

### 2.8 Summary of cross-implementation inconsistency

| | `evaluator.py` | `metrics.py` |
|---|---|---|
| Pairs included | All gallery items per query (duplicated) | Exhaustive positives, sampled negatives |
| Threshold method | Binary search in `[0, 1]` | `np.quantile(neg_scores, 1-FAR)` |
| Comparison | `scores >= threshold` | `scores > threshold` |
| Negative search range | `[0, 1]` — **breaks on negative similarities** | Unbounded via quantile |
| FAR grid | `{0.001, 0.01, 0.1}` | `{0.001, 0.01}` |

A maintainer who trains a new embedding and then runs the pipeline benchmark will see **different TAR@FAR numbers at the two stages for the same model**, with no documentation of why.

---

## 3. Experimental Design Findings

### 3.1 No held-out test set (F3)

- `EmbeddingDatasetConverter` (`embedding/dataset_converter.py:76-91`) performs an identity-disjoint train/val split at `val_split_ratio=0.15`, seeded. That's good — open-set protocol is followed.
- `EmbeddingTrainer.validate()` (`embedding/trainer.py:70-82`) runs `evaluate_embedding_model` on this `val_loader` **every epoch** and selects the best checkpoint by `mAP` (`embedding/trainer.py:131`).
- `run_full_pipeline_benchmark` (`scripts/train_master.py:91`) then runs the final pipeline benchmark on the same `data/identity_val.json`.

The best-model selection criterion and the reported benchmark are computed on the same set of images. For a one-off training run this is only mildly bad (mAP is one metric among several), but for any "did this PR regress quality?" workflow it is disqualifying — you cannot re-tune on val and simultaneously claim val numbers are unbiased.

**Required fix:** three-way split: `identity_train` / `identity_val` (model selection) / `identity_test` (locked, reported in CI only). Identity-disjoint across all three.

### 3.2 Gallery vs. query construction

- `BenchmarkEvaluator.evaluate` builds the gallery from every image in `ground_truth` and then uses every one of those images in turn as a query (`evaluator.py:136-149`). This is the standard 1:N closed-set identification protocol — fine.
- But: reproducibility across runs requires stable ordering. `IdentityLoader` calls `random.shuffle(selected_images)` (`identity_loader.py:169`) and subsampling with `random.sample` when `num_images` is set (`identity_loader.py:49`). With the `seed=42` default this is reproducible **only if nothing else perturbs the global `random` module before benchmark time**. In practice `random.seed(42)` is called in `IdentityLoader.__init__` and then `EmbeddingDatasetConverter` also calls `random.seed(42)` (`dataset_converter.py:78`) — which means if the benchmark is run after data prep in the same process, the seed may be in either state depending on execution order.

### 3.3 Class balance

- `identity_val.json` has ~5,300 lines ≈ 5,300 / 4 ≈ 1,300 validation entries (object-per-4-lines format; `wc -l` on pretty-printed JSON). Identity count unknown but inferable from `identity_label` distribution. There is no code anywhere that reports the number of identities, min/max/mean images per identity, or image count per identity. A maintainer cannot see at a glance whether the eval is dominated by a few big identities.

### 3.4 Standard benchmarks

- No usage of or plumbing for: ATRW (Amur Tiger Re-ID), SeaTurtleID, WildlifeReID-10k, PetFace, Cat Individual Re-ID, Dog Reidentification dataset (Hokkaido), OpenCows2020.
- A single-dataset benchmark on DogFaceNet that was also used for training+val is the worst case for external credibility.

### 3.5 Pipeline vs. component eval (F9)

- `run_full_pipeline_benchmark` only runs the **full pipeline** (detector + optional keypoint + embedder). If the detector recall drops from 0.99 to 0.95, `valid_ranks` shrinks accordingly and MRR looks about the same. If the embedder degrades, `valid_ranks` is unchanged but rank quality drops.
- A detector-only eval (mAP / IoU on COCO dogs), keypoint PCK, and an embedder-only eval (on ground-truth crops, no detector in the loop) are missing from the master script. `tests/integration/test_detection_pipeline.py` etc. exist but don't compute standard accuracy metrics in a way the benchmark consumes.

---

## 4. Reproducibility Findings

### 4.1 Seeds

- `IdentityLoader` seeds Python `random` at `__init__` (`identity_loader.py:19`), not `np.random` and not `torch.manual_seed`.
- `metrics.py:calculate_tar_at_far` seeds `np.random` lazily inside the function (`metrics.py:117-118`) — good for that function alone.
- `EmbeddingTrainer` never sets any seed. PyTorch determinism (`torch.use_deterministic_algorithms`, `cudnn.deterministic`) is not configured anywhere.
- Consequence: two runs of `scripts/13_run_full_pipeline.py` against the same ONNX models produce bit-identical embeddings (ONNX runtime is deterministic on CPU), but the **sampled subset** (when `--num-images` is set) may vary if global `random` state drifted, and the **training run** that produced those ONNX models is not reproducible.

### 4.2 Artifacts

- `BenchmarkMetrics` is only rendered to stdout and W&B. `evaluator.save_results()` writes a rich JSON but is **never called** anywhere in the scripts (`grep -r save_results`).
- `outputs/temp_ground_truth.json` is created and deleted in the same run (`train_master.py:220-221`). No record of what was benchmarked.
- `outputs/ambidextrous_axolotl/` contains only PNGs (`metrics_summary.png`, `query_results.png`), last touched Dec 13. No metrics.json, no csv, no run manifest.
- Two runs cannot be diffed without opening W&B.

### 4.3 Environment

- CI pins Python 3.12 via conda (`.github/workflows/python-package-conda.yml:22`). `environment.yml` presumably pins deps but benchmark numbers also depend on ONNX runtime version, `sklearn` version (for `cosine_similarity`), and BLAS. None are pinned against a known-good baseline.

---

## 5. Regression-Detection Gaps

Can a reviewer tell that a PR made embedding quality worse by 2%?

- **From CI: no.** The workflow runs `ruff check`, `ruff format --check`, `pytest`. Pytest covers the two tiny tests in `tests/unit/benchmark/test_metrics.py` which only assert "empty → 0", not numerical correctness.
- **From artifacts: no.** No on-disk baseline. No `outputs/baseline.json`. No file in the repo records last-known-good numbers.
- **From W&B: partially.** `WandBLogger` logs scalar metrics and failure images, grouped by `pipeline-no-keypoints` / `pipeline-with-keypoints` (`train_master.py:167-173`, `train_master.py:199-205`). A human with W&B access can compare two runs visually. No threshold alarms, no per-PR gating.
- **Confidence intervals: no.** All metrics are point estimates. A 2% drop on ~1,300 queries with ~300 positive pairs at FAR=0.001 is inside the noise floor and will look like signal.
- **Per-identity breakdown: no.** A regression that destroys one rare identity's embedding but averages out on the big ones is invisible.

---

## 6. PR-Sized Recommendations

Ordered by impact × effort. Each is scoped to a single PR.

### 6.1 Correctness (must-fix before trusting any number)

**PR-1: Fix TAR@FAR threshold search range (`evaluator.py:305`).**
Replace the binary search over `[0, 1]` with the quantile-based formula already in `metrics.py:154` (`threshold = np.quantile(neg_scores, 1 - target_far)`). Delete `_binary_search_threshold`. Unit-test with a synthetic case whose correct threshold is negative.

**PR-2: Unify TAR@FAR implementations.** Make `metrics.py:calculate_tar_at_far` the single source of truth. Have `evaluator.py` call it with the pipeline-produced embeddings + labels. This also resolves the `>` vs. `>=` inconsistency and the pair-duplication issue (F2) in one shot.

**PR-3: Stop filtering non-labelled images out of validation (F7).** Change `IdentityLoader._load_base_validation` to preserve null-identity items and extend `identity_val.json` to include a held-out set of labelled negatives drawn from Oxford Pets cats / COCO non-dog images. Then detection precision becomes a meaningful number.

### 6.2 Experimental design

**PR-4: Three-way split — introduce `identity_test.json`.** Modify `EmbeddingDatasetConverter` to carve out a locked test set at a separate ratio (e.g., 70/15/15 identity-disjoint). `run_full_pipeline_benchmark` reads `identity_test.json`. Training uses only train+val.

**PR-5: Component-level benchmarks in `train_master`.** Add `run_detector_benchmark` (YOLO val on COCO dogs, report mAP@50 / mAP@50-95) and `run_embedder_benchmark` (run `evaluate_embedding_model` on ground-truth crops — no detector — for the test set). Log all three sets of numbers side by side so a regression can be localised.

**PR-6: External benchmark integration.** Add a loader for at least one of: DogFaceNet held-out, ATRW tigers, SeaTurtleID. Even a single external dataset hugely raises the credibility of comparison claims.

### 6.3 Regression detection

**PR-7: On-disk baseline artifact.** At the end of `run_full_pipeline_benchmark`, always call `evaluator.save_results(outputs/benchmarks/<timestamp>_<git_sha>.json)` and also write a trimmed `metrics.json` with just the headline numbers. Add an `outputs/benchmarks/baseline.json` file committed to the repo; write a `scripts/diff_benchmarks.py` that prints deltas.

**PR-8: CI benchmark job.** Add a second GH Actions job that (a) downloads cached ONNX artifacts, (b) runs `13_run_full_pipeline.py --num-images 500 --no-wandb` on a CPU runner, (c) diffs against `baseline.json`, (d) fails if any headline metric drops by >1% absolute. For full runs, schedule a nightly workflow.

**PR-9: Bootstrap confidence intervals.** In `BenchmarkEvaluator._compute_metrics`, bootstrap-resample queries 1000× and report `(mean, 95% CI)` for MRR, top-k, TAR@FAR. Print them; log them; gate on lower CI in CI.

### 6.4 Ergonomics

**PR-10: Seed everything at entry.** Add a `animal_id/common/seed.py` that seeds Python `random`, `numpy`, `torch`, `torch.cuda`, sets `PYTHONHASHSEED`, and flips CuDNN deterministic. Call it from the two entry points. Log the seed in the output JSON.

**PR-11: Class-balanced + per-identity metrics.** In addition to the current macro averages, report macro-over-identities MRR, and dump a per-identity breakdown into the artifact JSON so regressions on rare identities are visible.

**PR-12: Dataset manifest in output.** Write identity count, image count, per-identity min/mean/max, positive-pair count, negative-pair count into the results JSON. This is effectively free and immediately exposes the class imbalance problem (3.3).

**PR-13: Delete duplicate self-exclusion.** In `ambidextrous_axolotl.py:87-92` and `evaluator.py:164-171`, either compare consistently on absolute paths or consistently on `identity_val.json`-relative paths. Pick one.

---

## 7. Closing Note for the Immich Reviewer

If you are deciding whether to rely on `immich-animals` benchmark numbers to approve a pet-recognition replacement for Immich's face-recognition pipeline:

- **Do not trust** `TAR@FAR` printed by `scripts/13_run_full_pipeline.py` until PR-1 and PR-2 land. The range bug in `evaluator.py:309` can silently over-report TAR at low FAR, which is the operating regime that matters for Immich (users complain about false merges).
- **Do not rely on** the headline Top-K / MRR numbers as absolute quality signals — they are validation-set numbers (F3). They are reasonable as *relative* signals between two runs on identical data, provided nothing in the split changed.
- **Request** the three-way split (PR-4), the CI benchmark job (PR-8), and the on-disk baseline artifact (PR-7) as prerequisites for merging any Immich hijack release. Without these, "did this PR regress?" cannot be answered mechanically.

The good news: the skeleton (injectable models, a single orchestrator, a metrics module already structured around mAP/TAR@FAR, W&B integration) is in place. The above list is fixable in a couple of focused weeks, not months.
