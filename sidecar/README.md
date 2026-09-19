# animal-ml sidecar

Adds your dogs to Immich's **People** tab. Immich already detects human faces and
groups them into people and this adds an additional dog detector on top of the
existing models. Human face detection and recognition should work the same as
before.

Dogs appear as people, mixed in with the humans. There is no separate animals
section.

> **First beta, and it focused on dogs.** Cats and other animals are not supported
> yet. A cat will occasionally be detected, but that is not the goal of this
> release or a focus in this round of model training.

> **Read this before pointing it at your main library.** Turning it on requires
> re-running face detection, which **discards the face work you have already done**:
> names, merges, splits and hidden faces are all reset, for people as well as dogs.
> Turning it back off means another re-run and another reset. If you have spent
> time naming people, use a test instance or a library you do not mind re-editing.

## What you need

- Immich running under `docker compose` (v3.0 or newer)
- About 1 GB of disk space

## 1. Add the service

Next to your Immich `docker-compose.yml`, create `docker-compose.override.yml`:

```yaml
services:
  animal-ml:
    container_name: animal_ml
    image: animal-ml
    environment:
      UPSTREAM_ML_URL: http://immich-machine-learning:3003
    restart: always
```

```bash
docker compose up -d animal-ml
```

## 2. Point Immich at it

**Administration → Settings → Machine Learning** → set the URL to
`http://animal-ml:3003` and save.

That is the only setting to change. Leave **Min Detection Score** and **Max
Distance** where they are: they are tuned for human faces and still apply to
them. Dogs need different values, so the sidecar uses its own thresholds.

## 3. Find the dogs

**Administration → Job Queues → Face Detection → All**, then when it finishes,
**Facial Recognition → All**. Expect roughly ten photos a second; the People tab
fills in as it goes.

Seeing only a handful of people afterwards is Immich hiding anyone with fewer
than three photos — **Account Settings → Features → People** lowers that.

Name a dog the way you would name a person, and search finds them by name.

## What works, what does not

Dogs you photograph a lot cluster well. In testing, a dog with 673 photos put
491 of them in one person; a dog with 11 photos scattered across five "people".

- **Dogs with plenty of photos** get one large cluster plus a few strays to merge
- **Dogs with under ~15 photos** may not group at all
- **Similar-looking dogs** get mixed together — two black curly-coated dogs are
  genuinely hard
- **Cats** are detected as people about one photo in seven.

Merging two people in Immich is easy, but splitting one is not, so the defaults
lean towards leaving you a few extra clusters rather than wrongly combining two
dogs.

## Turning it off

Point the Machine Learning URL back at `http://immich-machine-learning:3003` and
re-run Face Detection. Dog people remain listed until you delete them, and — as
above — this second re-run resets face names and merges again.

## Settings

TODO: settings where? This section doesn't say where you can configure these

| Variable | Default | What it does |
| --- | --- | --- |
| `UPSTREAM_ML_URL` | — | Your existing Immich ML container. Required: search and OCR are forwarded to it. |
| `KEEP_HUMAN_FACES` | `true` | `false` serves dogs only and stops detecting human faces. |
| `DOG_MIN_SCORE` | `0.3` | How confident the detector must be. Lower finds more dogs and more cats. |
| `DOG_MAX_DISTANCE` | `0.35` | How alike two dogs must look to count as the same dog. Lower splits more, higher merges more. |
| `IMMICH_MAX_DISTANCE` | `0.5` | The Max Distance in your Immich settings. Change only if you changed that. |

Immich's face-model setting picks the embedder too: `buffalo_l` and anything
larger or unrecognised uses the more accurate ConvNeXt model, `buffalo_m` and
`buffalo_s` use a faster ResNet50. Changing it means re-running face detection,
since embeddings from the two are not comparable.

## Trouble

**"Machine learning server became unhealthy"** — `animal-ml` is not running, or
not on Immich's docker network. Check `docker logs animal_ml`.

**Search stopped working** — `UPSTREAM_ML_URL` is wrong. Text search is
forwarded to Immich's own ML container.

**No dogs at all** — Face Detection was probably run as "Missing" rather than
"All".

**Too many near-duplicate people** — merge them, or raise `DOG_MAX_DISTANCE` to
`0.4` and re-run Facial Recognition.

---

# Development

The main [README](../README.md) covers the models and training. This section is
only what is specific to the sidecar.

`serve.py` is the whole integration, in about 200 lines. Immich talks to machine
learning over HTTP, so this needs no fork of Immich and no patched image — just
a service that answers `POST /predict` and `GET /ping` the way Immich expects.

```
immich-server ──▶ animal-ml ──facial-recognition──▶ our ONNX models
                      └──────everything else──────▶ immich-machine-learning
```

It is its own uv project, separate from the training project at the repo root
to keep the dependencies lighter than the training project.

## Build

Models are not in git. The default build downloads them from the release named
by `MODEL_TAG` and checks them against `SHA256SUMS`.

```bash
# from the repo root
docker buildx build -f sidecar/Dockerfile --load -t animal-ml .

# with the models already on disk
docker buildx build -f sidecar/Dockerfile --build-arg MODEL_SOURCE=local --load -t animal-ml .
```

## Test

`smoke_test.py` posts images exactly as Immich does and asserts the reply shape,
including that `embedding` is a JSON *string* — Immich casts it to a pgvector.

```bash
uv run --project sidecar python sidecar/smoke_test.py dog.jpg --url http://localhost:3003
```

For tuning, `scripts/fetch_validation_set.py` builds a held-out set (121
individual dogs, 1524 photos, 450 dog-free negatives) and
`scripts/evaluate_sidecar.py` sweeps detection recall against the false-positive
rate and clusters the embeddings the way Immich does.

## How the thresholds are handled

Immich applies one Min Detection Score and one Max Distance to every face, and
both are tuned for people. Rather than make users retune them:

- **Detection score.** Immich only forwards `minScore` to the ML server and never
  re-filters, so dogs use `DOG_MIN_SCORE` and the user's setting continues to
  govern human faces upstream. At Immich's 0.7 default we would lose about half
  the dogs; at 0.3 we find ~80% of them.
- **Max Distance.** Mixing each embedding with an independent random unit vector
  maps cosine distance affinely, since random high-dimensional vectors are
  near-orthogonal:

  ```
  d' = (1 - a) + a·d     a = (1 - IMMICH_MAX_DISTANCE) / (1 - DOG_MAX_DISTANCE)
  ```

  At the defaults `a = 0.769`, so a raw distance of 0.35 lands on 0.5. The target
  Gram matrix `a·G + (1-a)I` is positive semi-definite, so this geometry exists;
  the random vectors approximate its exact realisation in 512 dimensions.
  Measured on the built image: `d' = 0.79·d + 0.22`, deterministic per face.

  The cost is a 0.03 residual. Clustering v-measure held at 0.729 on held-out
  internet photos and rose to 0.838 on real personal ones, but fell from 0.777 to
  0.707 on tightly-cropped dataset images. Human embeddings are never touched, so
  disabling the sidecar leaves them valid.

