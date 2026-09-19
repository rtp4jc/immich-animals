# Shipping the sidecar: models, image, first release

**Date:** 2026-09-19
**Scope:** how the two ONNX files and the container reach a beta tester who has
Immich in docker compose and will never clone this repo. Plan only; nothing here
is implemented.

The blocker is concrete: `models/` is gitignored, the Dockerfile does
`COPY models/onnx/...`, so a clone cannot build the image.

---

## 1. Licensing

### What the shipped models actually are — verified, not assumed

**`embedding.onnx` is ConvNeXt **V1**-Tiny, not V2.** Three independent checks
agree:

- `embedding.json` records `"backbone": "convnext_tiny"`.
- The local HF cache holds `timm/convnext_tiny.fb_in22k_ft_in1k` alongside the
  V2 tags, so V1 was fetched.
- The ONNX graph settles it. In timm, V1 carries 18 `gamma` (LayerScale) tensors
  and zero GRN; V2 carries 36 GRN tensors and zero `gamma`. The export has
  **zero GRN** initializers, LayerScale folded into `Mul` constants, stage depths
  3/3/9/3, and 28.21M params against a 27.82M V1 trunk plus the projection head.

This matters because the prior audit was right and still is:
`facebookresearch/ConvNeXt-V2` states it is "released under the MIT license
**except ImageNet pre-trained and fine-tuned models which are licensed under a
CC-BY-NC**". Every V2 weight tag is non-commercial. The switch to V1 was
necessary, and the currently-shipped export is on the clean side of it.

V1's licence: the `timm/convnext_tiny.fb_in22k_ft_in1k` model card states
`apache-2.0`; upstream `facebookresearch/ConvNeXt` is MIT. The two disagree, but
both are permissive and both permit redistribution, so the disagreement is
harmless.

**`detector.onnx` is a `yolo11n.pt` fine-tune** (`animal_id/detection/trainer.py`
defaults to it; `scripts/train_master.py` names it). The installed `ultralytics`
8.4.76 declares `AGPL-3.0` in its metadata and classifiers. ONNX export does not
launder this — Ultralytics' stated position explicitly covers trained and
fine-tuned models, not just their Python code.

### What that obliges

AGPL-3.0 is the binding constraint on the whole distribution, because a single
AGPL component pulls the combined work with it. Two obligations:

- **Distribution.** Publishing the image or the weights means offering complete
  corresponding source under AGPL-3.0. For us that is satisfied by the repo being
  public, which it must be anyway.
- **§13, the network clause.** Running it as a service counts as conveying to
  remote users. In a self-hosted sidecar the only remote user is the tester's own
  Immich server, so the obligation is trivially met — but a tester who modifies
  the sidecar and exposes it to other people inherits the source-offer duty, and
  the README should say so in one line.

Ultralytics markets an Enterprise licence as *required* for SaaS and commercial
products. That is their commercial framing, not what the AGPL text requires —
AGPL permits network use provided source is offered. For a free, open,
non-commercial project the two readings converge, so **publishing the whole repo
under AGPL-3.0 resolves it either way**. Immich itself is AGPL-3.0, so this is
also the natural licence for the ecosystem. I am not a lawyer and Ultralytics has
been notably aggressive on this point; see the open questions.

### Training-data terms — a separate axis, and the weaker one

- **Detector**: trained on Stanford Dogs + DogFaceNet. Stanford Dogs is built
  from ImageNet images and inherits ImageNet's **non-commercial research and
  educational use only** terms. Whether model weights are a derivative work of
  their training data is legally unsettled, so this is not a hard bar to
  releasing — but it does mean **we must not advertise the models as usable
  commercially**.
- **Embedder**: trained on DogFaceNet, whose code is MIT but whose images the
  authors describe only as "retrieved from the web", with no stated image
  licence. This is the weakest link in the chain and I could not resolve it.

### Redistribution verdict

| File | Lineage | Redistributable? | Ships under |
| --- | --- | --- | --- |
| `detector.onnx` | `yolo11n.pt` fine-tune (Ultralytics) | Yes — **only** if the whole work is AGPL-3.0 | AGPL-3.0 |
| `embedding.onnx` | `timm/convnext_tiny.fb_in22k_ft_in1k` fine-tune | Yes — upstream Apache-2.0/MIT | AGPL-3.0 (combined work) |
| `embedding.json` | ours | Yes | AGPL-3.0 |
| `embedding_resnet50.onnx` | torchvision ResNet50 (permissive) | Yes, but not shipped — it is the losing ablation arm | — |
| `embedding.pre-ablation.onnx` | **unrecorded** | **Do not ship** | — |

Nothing currently on disk is CC-BY-NC. `embedding.pre-ablation.onnx` is
structurally V1 too (28.26M params, zero GRN), so it is probably clean — but its
run and weight tag are unrecorded, it is not in the Dockerfile, and there is no
reason to publish it. Leave it out of every release artefact.

The one file that genuinely cannot be redistributed is any future V2-backed
export. That is a live risk, not a hypothetical — see B2.

---

## 2. Where the model files should live

| Option | Size / cost | Tester UX | Licence fit | Verdict |
| --- | --- | --- | --- | --- |
| **GitHub Releases** | 2 GiB per file, no total cap, no bandwidth billing | `curl` one-liner, or never seen at all if baked into the image | Fine | **Recommended** |
| Hugging Face Hub | 500 GB free per public repo | Good; needs a second account | Fine; one licence tag per repo, awkward for two differently-licensed files | Optional later mirror |
| git-lfs | GitHub LFS free tier is ~1 GiB storage and ~1 GiB/month bandwidth | Invisible until it breaks | Fine | **Reject** — 118 MB per clone exhausts the free tier almost immediately, and it taxes every contributor who only wants the Python |
| Image only | — | Best, if you only ever use the image | Fine | Not sufficient alone: no standalone artefact, no checksum story for people who build |

**Recommendation: GitHub Releases as the canonical source, with the published
image baking the files in.** The models are release artefacts logically pinned to
a tag; releases pin them to a tag literally. Bandwidth is free and unmetered,
there is one identity to manage rather than two, and `sha256sum -c` against a
committed sums file is a two-line verification.

Hugging Face is a reasonable *mirror* later if discovery matters — Immich hosts
its own 64 ONNX repos there, so testers already know the pattern — but it adds an
account and a second thing to keep in sync for no benefit at this size.

---

## 3. How the image gets built and delivered

| Option | Cost to tester | Failure modes |
| --- | --- | --- |
| **(a) Prebuilt multi-arch image on ghcr.io, models baked in** | `docker compose up -d` | Image size |
| (b) Build-it-yourself, Dockerfile downloads models at build time | Needs the repo, a build, and a toolchain | Build-time network, slow on a Pi |
| (c) Download at first run into a volume (Immich's `/cache` pattern) | `docker compose up -d` | First-run network, partial downloads, latency spike on first request, one more volume |

**Recommendation: (a), prebuilt multi-arch on ghcr.io with the models baked in.**

Reasoning:

- The target tester will not clone a Python repo. (a) is the only option where
  they never touch this codebase.
- Baking in means **the image digest pins the models**. One version number covers
  code and weights, the build is reproducible, and there is no first-run network
  dependency.
- Size is a non-issue: 118 MB of models on a python-slim + onnxruntime + opencv
  base is small next to Immich's own ML image.

**Why not copy Immich's `/cache` pattern**, even though it is the idiomatic one:
Immich downloads at runtime because it has ~64 interchangeable models and the
user picks one in settings (`MACHINE_LEARNING_CACHE_FOLDER`, default
`~/.cache/immich_ml`, mounted as the `model-cache:/cache` named volume). We have
exactly two models and they are never swapped. Runtime download would buy nothing
and add three failure modes. Revisit only if we ship selectable model variants.

Keep (b) alive as the *contributor* path: the Dockerfile should fall back to
downloading the release tarball when `models/onnx/` is absent, so a fresh clone
still builds. That fixes the stated blocker directly.

### Multi-arch is viable — checked

All sidecar dependencies have linux `aarch64` manylinux wheels at their locked
versions: `onnxruntime` 1.30.0, `opencv-python-headless` 5.0.0.93, `orjson`
3.12.0 (the rest are pure Python). ONNX Runtime's default CPU wheel covers both
arches; no GPU variant is involved.

Build with `docker buildx` across **native runners** — `ubuntu-24.04` and
`ubuntu-24.04-arm`, both free for public repos — rather than QEMU. QEMU would
work, since the arm64 layer only unpacks wheels and compiles nothing, but native
is faster and avoids emulation surprises.

**Untested:** nothing has ever run on arm64. Treat arm64 as unsupported until CI
runs `smoke_test.py` on it (B6).

---

## 4. Integrity and versioning

- **Checksums.** Commit `models/onnx/SHA256SUMS` — the sums are tiny even though
  the models are not. Today's values:

  ```
  ef275eb61769edf81bd14222d310b4122659d310529f4af5576ff070d611ebd6  detector.onnx
  f4d9caf7064462aa5cd77b911aa2047086ee2bcce45028ec0663c17d8b099b3d  embedding.onnx
  433b236b309b30feeafd39caa1b3a475ddc653646a09612d9b3db948ed99a4f5  embedding.json
  ```

  The Dockerfile's download path runs `sha256sum -c` so a truncated or tampered
  fetch fails the build instead of shipping quietly.

- **Pinning.** One tag, e.g. `sidecar-v0.1.0`, carries the three files as release
  assets and the image as `ghcr.io/<owner>/animal-ml:0.1.0`, plus `:latest` and
  the commit SHA. Same number on both, so "what are you running" has one answer.

- **Reporting.** `embedding.json` already has a `version` field
  (`20260919_031123_convnext_tiny_final`). Log it, plus the image tag, at startup
  so a bug report identifies the models. Do **not** add it to `GET /ping` —
  Immich polls that for health and the response shape should stay boring.

- **Updates.** A tester runs `docker compose pull && docker compose up -d`. The
  part that needs saying loudly in every release note: **changing `embedding.onnx`
  invalidates every stored embedding.** Immich's existing vectors are not
  comparable to new ones, so the tester must delete existing people and re-run
  **Jobs → Face Detection → All**. A detector change needs re-detection too.
  Therefore: any model change is at minimum a minor bump and carries a
  "re-run face detection" banner. Code-only fixes are patch bumps and need
  nothing.

---

## 5. What a beta tester actually runs

Assumes Immich already runs from `docker-compose.yml`. No clone, no Python.

1. Download the override next to Immich's `docker-compose.yml`:
   `curl -o docker-compose.override.yml <release asset URL>`
   It pins `ghcr.io/<owner>/animal-ml:<version>` and sets `UPSTREAM_ML_URL`,
   `KEEP_HUMAN_FACES`, `DOG_MAX_DISTANCE`, `IMMICH_MAX_DISTANCE`.
2. `docker compose up -d`
3. Immich → **Administration → Settings → Machine Learning**: URL
   `http://animal-ml:3003`, **Min Detection Score `0.3`** (0.7 halves recall on
   dogs), **Max Distance** per the mode below.
4. **Administration → Jobs → Face Detection → All.** "Missing" will not do it —
   existing assets are already marked done.
5. If clusters seem absent, **Account Settings → Features → People** — minimum
   faces is a per-user preference in v3, default 3.

Things they must be told, because each one silently ruins the result:

- **The override must join the same compose project and network** as
  `immich-machine-learning`. If their ML container has a different name,
  `UPSTREAM_ML_URL` has to change with it.
- **`KEEP_HUMAN_FACES` changes what Max Distance should be.** With it on, the
  sidecar rescales dog embeddings onto Immich's human-tuned threshold; with it
  off, they set Max Distance to `DOG_MAX_DISTANCE` themselves.
- **Pointing this at a real library mixes dogs into People, and there is no
  one-click undo.** Recommend a test instance for the beta.
- Expect roughly one cat photo in seven to become a junk person, and wolves and
  dingoes to be detected as dogs.

---

## 6. Release checklist

**Blocking — none of this ships until these are done:**

- **B1. Add `LICENSE` (AGPL-3.0) at the repo root.** There is no licence file
  today, which means all rights reserved: nobody may legally redistribute or fork
  it, and the Ultralytics obligation is unmet. This is the single hardest blocker.
- **B2. Restore a `convnext_tiny` entry to `animal_id/embedding/backbones.py`,
  pinned to the exact tag `convnext_tiny.fb_in22k_ft_in1k`.** The registry
  currently offers only `convnextv2_tiny` / `convnextv2_nano`, and the run
  directory for the shipped export is empty — so the licence-clean backbone
  exists only in the exported artefact and a local HF cache. Anyone retraining
  today lands on CC-BY-NC V2 weights without noticing.
- **B3. Record the licence decision in a committed file.**
  `.planning/6-25-2026-embedding-backbone-ablation/findings.md` does not exist on
  any branch — only `plan.md`. The audit that drove the V1 switch is unwritten.
- **B4. Reconcile `sidecar/README.md` against `sidecar/docker-compose.yml`.** The
  README documents `KEEP_HUMAN_FACES=false` as the default and a flat Max Distance
  of 0.35; the compose file now ships `true` plus `DOG_MAX_DISTANCE` /
  `IMMICH_MAX_DISTANCE`. Testers will follow whichever they read first.
- **B5. Settle the Ultralytics question** (AGPL the repo, as recommended, or
  retrain the detector off a non-AGPL base).
- **B6. Run `smoke_test.py` on arm64 in CI** before claiming arm64 support.

**Then, in order:**

7. Add a `NOTICE` (or README section) attributing Ultralytics AGPL-3.0, ConvNeXt,
   timm, DogFaceNet and Stanford Dogs, and stating the non-commercial
   training-data caveat.
8. Commit `models/onnx/SHA256SUMS`.
9. Teach the Dockerfile to fall back to downloading + verifying the release assets
   when `models/onnx/` is absent, so a clone builds.
10. Add the release workflow: buildx on `ubuntu-24.04` + `ubuntu-24.04-arm`, push
    `linux/amd64,linux/arm64` to ghcr.io, attach the three model files to the tag.
11. Publish a tester-facing `docker-compose.override.yml` as a release asset with
    the image reference pinned.
12. Rewrite `sidecar/README.md` for someone who will not clone the repo: the
    five steps above, the caveats, and the accuracy numbers already measured.
13. Tag `sidecar-v0.1.0`, verify the published image pulls and runs on a clean
    machine, then hand out the link.

---

## 7. Open questions

- **Ultralytics AGPL vs Enterprise.** Their marketing says SaaS needs an
  Enterprise licence; the AGPL text says network use is fine given a source
  offer. Converges for a free open project, but it is your call and possibly a
  lawyer's. I am not one.
- **DogFaceNet image provenance.** "Retrieved from the web", no stated image
  licence. Unresolved, and the weakest link in the embedder's chain.
- **Stanford Dogs / ImageNet non-commercial terms** on the detector's training
  data. Not a bar to a free release, but it rules out ever describing the models
  as commercially usable.
- **Repo must be public** — required for both the AGPL source offer and the free
  arm64 runners. Confirm that is the intent, and decide the ghcr owner/name.
- **Delete `embedding.pre-ablation.onnx`?** Structurally clean, provenance
  unrecorded, no reason to keep it once the ablation is documented.
