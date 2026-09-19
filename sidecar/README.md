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
| Max Distance | `0.35` | `clustering.eps` from `models/onnx/embedding.json` |
| Min Detection Score | `0.3` | 0.7 is tuned for human faces and drops most dogs |

`Min Faces` is a per-user preference in v3 (**Account Settings → Features →
People**), not an admin setting.

## Verified

Against a stock Immich **v3.2.2** stack with 31 Wikimedia dog photos: 12 people
clustered, correct bounding-box thumbnails, CLIP smart search and OCR still
served through the proxy. Dropping Min Detection Score from 0.7 to 0.3 took
detections from 20 faces across 18 photos to 28 across 25.

## Notes

- Real human faces stop being detected while Immich points here. Use a test
  instance or a library you do not mind mixing.
- Upgrade exposure is one JSON shape; the contract was unchanged from v3.0 to
  v3.2.2.
- `smoke_test.py` posts images exactly as Immich does and asserts the reply
  shape, including that `embedding` is a JSON *string* (Immich casts it to a
  pgvector).

```bash
python sidecar/smoke_test.py path/to/dog.jpg --url http://localhost:3003
```
