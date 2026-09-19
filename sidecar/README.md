# animal-ml sidecar

Serves Immich's machine-learning HTTP contract with our dog models, so Immich
finds dogs where it thinks it is finding human faces. No Immich fork: the
integration seam is one JSON shape over HTTP.

```
immich-server ──▶ animal-ml (this) ──facial-recognition──▶ our ONNX
                        └─────────everything else────────▶ immich-machine-learning
```

The proxy is not optional — Immich's `urls` list is failover, not routing, so
whichever server answers has to answer every task.

This is its own uv project, deliberately separate from the training project at
the repo root: the container installs `fastapi`/`onnxruntime`/`opencv` and
nothing else, with no path to torch.

## Run it

```bash
docker build -f sidecar/Dockerfile -t animal-ml .          # from the repo root
cp sidecar/docker-compose.yml \
   <immich>/docker/docker-compose.override.yml
docker compose up -d animal-ml
```

Then in Immich, **Administration → Settings → Machine Learning**:

| Setting | Value | Why |
| --- | --- | --- |
| URL | `http://animal-ml:3003` | replaces the stock ML server |
| Min Detection Score | `0.3` | 0.7 is tuned for human faces and drops most dogs |
| Max Distance | leave as-is | the sidecar moves its own geometry onto yours |

`Min Faces` is a per-user preference in v3 (**Account Settings → Features →
People**), not an admin setting.

## Dogs only, or dogs and humans?

| `KEEP_HUMAN_FACES` | Behaviour |
| --- | --- |
| `true` (default) | Dog faces **plus** whatever the stock model finds. Embeddings rescaled so one Max Distance suits both. |
| `false` | Only dogs; human faces stop being detected. Set Max Distance to `DOG_MAX_DISTANCE` yourself. |

Immich applies one `Max Distance` to every face, but our 512-d geometry is not
ArcFace's — ours clusters best at 0.35, Immich defaults to 0.5. Rather than make
the user reconcile that, we move our side.

Mixing each embedding with an independent random unit vector maps cosine
distance affinely, because random high-dimensional vectors are near-orthogonal:

```
d' = (1 - a) + a·d        a = (1 - IMMICH_MAX_DISTANCE) / (1 - DOG_MAX_DISTANCE)
```

At the defaults `a = 0.769`, so a raw distance of 0.35 lands on 0.5 and Immich's
own threshold does the right thing for both. It is not a hack: the target Gram
matrix `a·G + (1-a)I` is positive semi-definite, so this geometry exists — the
random vectors are how you approximate its exact realisation in 512 dimensions.
Measured on the shipped container, `d' = 0.79·d + 0.22` with a residual of 0.03,
and the transform is deterministic per face.

The cost is that residual: clustering v-measure held at 0.729 (Commons) and rose
to 0.838 (real personal photos), but fell from 0.777 to 0.707 on MPDD's tight
crops. Human embeddings are never touched, so disabling the sidecar leaves them
valid.

## Validate

`smoke_test.py` posts images exactly as Immich does and asserts the reply shape,
including that `embedding` is a JSON *string* (Immich casts it to a pgvector):

```bash
uv run --project sidecar python sidecar/smoke_test.py path/to/dog.jpg \
  --url http://localhost:3003
```

For threshold tuning, `scripts/fetch_validation_set.py` builds a held-out set
(121 individual dogs, 1524 photos, plus 450 dog-free negatives) and
`scripts/evaluate_sidecar.py` sweeps both settings against it:

```bash
uv run python scripts/fetch_validation_set.py
uv run python scripts/evaluate_sidecar.py --url http://localhost:3003
```

## Verified

Against a stock Immich **v3.2.2** stack: people clustered from dog photos,
thumbnails cropped from our bounding boxes, CLIP smart search and OCR still
served through the proxy, in both `KEEP_HUMAN_FACES` modes.

On the held-out validation set, at Min Detection Score `0.3`: **80%** of
in-the-wild dog photos get a detection, against a **0%** false-positive rate on
people, landscapes, horses and buildings. What it does fire on is near-neighbour
quadrupeds — 40% of wolf/dingo/jackal photos and 14% of cat photos — so the junk
person risk in a real library is roughly one cat photo in seven. Immich's 0.7
default halves recall.

## Notes

- With `KEEP_HUMAN_FACES=false`, use a test instance or a library you do not
  mind mixing.
- Upgrade exposure is one JSON shape; the contract was unchanged from v3.0 to
  v3.2.2.
