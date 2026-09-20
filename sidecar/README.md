# animal-ml sidecar

Adds your dogs to Immich's **People** tab. Immich already detects human faces and
groups them into people and this adds an additional dog detector on top of the
existing models. Human face detection and recognition should work the same as
before.

Dogs appear as people, mixed in with the humans. There is no separate animals
section.

**Testing in beta with a focus on dogs.** Cats and other animals are not supported
yet. A cat will occasionally be detected, but that is not the goal of this
release or a focus in this round of model training.

**Use Refresh, not Reset** on the Face Detection queue. Refresh keeps every
face already in your library, so names, merges and hidden people survive both
adding the sidecar and removing it. Reset deletes every detected face first.

## What you need

- Immich running under `docker compose` (v3.0 or newer)
- About 1 GB of disk space

## 1. Add the service

Next to your Immich `docker-compose.yml`, create `docker-compose.override.yml`:

```yaml
services:
  animal-ml:
    container_name: animal_ml
    image: ghcr.io/rtp4jc/animal-ml:0.1.0
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

**Administration → Job Queues → Face Detection → Refresh**.

Expect this to take up to several hours depending on the size of your library. 
The full face detection queue must empty before face recognition starts so 
you will not see partial results until the detection queue is empty.

When completed, name and modify a dog the way you would a person!

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
run **Face Detection → Refresh**. The dog faces will be removed, but
human faces will be untouched. Dog names are not kept, so turning it on again requires
you to add names again.

## Settings

### In the docker container definition
These are the sidecar's settings, not in Immich. 
Add them under `environment:` in the `docker-compose.override.yml` from step 1, then
`docker compose up -d animal-ml` to apply and do **Face Detection → Refresh**.

| Variable | Default | What it does |
| --- | --- | --- |
| `UPSTREAM_ML_URL` | — | Your existing Immich ML container. Required: search and OCR are forwarded to it. |
| `KEEP_HUMAN_FACES` | `true` | `false` serves dogs only and stops detecting human faces. |
| `DOG_MIN_SCORE` | `0.3` | How confident the detector must be. Lower finds more dogs and more cats. |
| `DOG_MAX_DISTANCE` | `0.35` | **SEE NOTE BELOW** How alike two dogs must look to count as the same dog. Lower splits more, higher merges more. |
| `IMMICH_MAX_DISTANCE` | `0.5` | The Max Distance in your Immich settings. Change only if you changed that. |

**NOTE**: `DOG_MAX_DISTANCE` is tricky to change. You can't just do a refresh after changing 
it because previously detected faces do not get a new embedding on refresh. If you need to change it, I would suggest 
disabling the sidecar temporarily, refreshing the face detection again (**Face Detection → Refresh**),
then adding the sidecard again with the new configuration and refreshing. This will result in 
only the named dogs being lost and the human faces remaining unchanged.

### From Immich UI
Immich's face-model setting picks the embedder: `buffalo_l` and anything
larger uses the more accurate ConvNeXt model, `buffalo_m` and `buffalo_s` use a faster 
ResNet50. Changing it requires a **Face Detection → Reset** run (not just Refresh),
since embeddings from the two are not compatible. You could hypothetically follow the same procedure
as changing the `DOG_MAX_DISTANCE` above, but human face embeddings technically need to be 
reset too when you change this setting so I wouldn't.

## Trouble

**"Machine learning server became unhealthy"** — `animal-ml` is not running, or
not on Immich's docker network. Check `docker logs animal_ml`.

**Search stopped working** — `UPSTREAM_ML_URL` is wrong. Text search is
forwarded to Immich's own ML container.

**No dogs at all** — Face Detection was probably run as **Missing**, which only
looks at photos that have never been scanned. Run it as **Refresh**.

**Too many near-duplicate people** — merge them, or raise `DOG_MAX_DISTANCE` to
`0.4` and run Facial Recognition → Reset, which re-clusters everything and drops
names.

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
docker buildx build -f sidecar/Dockerfile --load -t ghcr.io/rtp4jc/animal-ml:0.1.0 .

# with the models already on disk
docker buildx build -f sidecar/Dockerfile --build-arg MODEL_SOURCE=local --load -t ghcr.io/rtp4jc/animal-ml:0.1.0 .
```

Building under the published tag shadows the released image locally, so the
setup in step 1 runs your build without edits. `docker pull` it again to go
back.

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
- **Max Distance.** Our embeddings cluster best around 0.35, so the sidecar
  stretches its own vector space to spread those distances out and land on
  Immich's 0.5 default. `DOG_MAX_DISTANCE` and `IMMICH_MAX_DISTANCE` set the two
  ends of that mapping. Human embeddings are passed through untouched, so
  disabling the sidecar leaves them valid.

  Unlike the detection score, Max Distance is still a live knob for dogs: the
  stretch is fixed, so lowering it in Immich splits dogs more and raising it
  merges more, exactly as it does for people. It just stops being calibrated —
  set `IMMICH_MAX_DISTANCE` to match if you move it far.

