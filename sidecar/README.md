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
| Max Distance | `0.35` | best v-measure/purity balance on held-out photos; 0.30 is purer but leaves 41% unclustered |

`Min Faces` is a per-user preference in v3 (**Account Settings → Features →
People**), not an admin setting.

## Dogs only, or dogs and humans?

| `KEEP_HUMAN_FACES` | Behaviour |
| --- | --- |
| `false` (default) | Only dogs. Human faces stop being detected entirely. |
| `true` | Dog faces **plus** whatever the stock model finds, in one response. |

The cost of `true` is not the code — it is that Immich clusters both with a
single `Max Distance`, and our 512-d embedding geometry is not ArcFace's. A
threshold tuned for dogs will over- or under-merge people. Both modes are
supported so you can measure that rather than guess.

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
