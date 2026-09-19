# Decisions

Written 2026-09-19 after the sidecar was verified end-to-end against Immich
v3.2.2. Resolved 2026-09-19.

## 1. Dogs and humans — RESOLVED: make it toggleable

`KEEP_HUMAN_FACES=false` (default) answers `facial-recognition` with dogs only;
human faces stop being detected. `KEEP_HUMAN_FACES=true` also forwards the
request upstream and concatenates the stock model's faces with ours.

The code cost was eight lines. The real cost of `true` is that Immich clusters
both with a single `Max Distance`, and our 512-d embedding geometry is not
ArcFace's — a threshold tuned for dogs will over- or under-merge people. Both
modes ship so that is measurable rather than guessed.

## 2. Min Detection Score — needs the better validation set

Immich's default `0.7` is tuned for human faces: on the throwaway set it found
20 faces in 18 of 31 photos, versus 28 in 25 at `0.3`. `0.3` is what the README
recommends today, but five identities and 31 Wikimedia photos is not a tuning
set, and that measurement says nothing about **false positives** — every one
becomes a junk person in the library.

`scripts/fetch_validation_set.py` and `scripts/evaluate_sidecar.py` exist to
settle this properly: recall on dog photos and false-positive rate on dog-free
photos, swept over `minScore`. Re-run once the user's own photos are in.

Still open: whether the sidecar should ignore Immich's `minScore` and use its
own constant, so a fresh install does not silently behave badly at the 0.7
default. Leaning no — the knob belongs where a user expects it — but the
false-positive numbers should decide.

## 3. Max Distance — RESOLVED: 0.30, but the basis is shakier than it looked

The decision stands: Immich can merge two people into one and has no equivalent
for splitting one, so the cheap error is the one to prefer.

What is worth knowing is that **`eps=0.35` in `embedding.json` was swept on
DogFaceNet's test split** — 149 identities of tight, mostly frontal face crops —
and Immich will never send anything like that. Measured on 27 in-the-wild
photos through the sidecar's own crop geometry, same-identity cosine similarity
averages **0.54**, not the ~0.9 that split implies. A `maxDistance` of 0.30 only
links pairs above 0.70 similarity, which most true pairs miss.

Swept on those 27 photos (5 identities, `min_samples=2`):

| maxDistance | clusters | unassigned | homogeneity | completeness | v-measure |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 4 | 41% | 0.688 | 0.688 | 0.688 |
| 0.30 | 5 | 33% | 0.732 | 0.658 | 0.693 |
| 0.35 | 4 | 22% | 0.665 | 0.715 | 0.689 |
| 0.40 | 5 | 7% | 0.707 | 0.669 | 0.687 |
| 0.45 | 4 | 0% | 0.605 | 0.716 | 0.656 |

V-measure is flat from 0.25 to 0.45 — the threshold barely changes cluster
*quality*, it changes how much is left unassigned. At 0.30 a third of detections
cluster with nothing, and with `minFaces=1` each becomes its own person. That is
the manual merging being signed up for: in the live run it produced 12 people
from 28 faces of 5 dogs.

Re-sweep on the real validation set before treating any of this as settled.

## 3b. Crop geometry — tested, left alone

The embedder trains on DogFaceNet, which is tight, roughly aligned, mostly
frontal dog *faces*. The sidecar feeds it whole-body detector boxes with 10%
padding — visibly a different distribution, full of carpet and grass. That looked
like a train/serve skew worth fixing with the abandoned fork's keypoint stage.

Measured, it is not. Paired on the same photos:

| crop | same | different | margin | top-1 |
| --- | --- | --- | --- | --- |
| body (sidecar today) | 0.595 | 0.166 | 0.429 | 0.714 |
| keypoint face | 0.360 | 0.093 | 0.267 | 0.476 |

Even excluding the crops where the keypoint stage visibly returned background
rather than a face, body still wins (top-1 0.625 vs 0.562). The stale keypoint
model fails often — 6 of 27 crops produced no usable face and ~5 more were
grass or fur — and when it succeeds the crop is frequently a profile, which
DogFaceNet mostly is not.

Also checked whether the body crop's advantage is just background leakage, since
same-identity photos here share rooms: embedding a dog-free strip of each source
image gives a margin of **0.027** and top-1 0.391, near chance. The dog carries
the signal, not the room.

So dropping the keypoint stage was right, and the plan's reasoning for dropping
it (Immich's contract does not need landmarks) was right for the wrong reason —
it happens to also be the better crop.

## 4. Dependency duplication — RESOLVED: separate uv project

`sidecar/` is now its own uv project with its own `pyproject.toml` and
`uv.lock`, and `requirements.txt` is gone. The root `pyproject.toml` and
`uv.lock` are byte-identical to `main` again — the sidecar adds nothing to the
training project, and the container has no path to torch.

Run its tooling with `uv run --project sidecar ...`.

## 5. The test instance

A full Immich v3.2.2 stack runs from `/home/ryan/Code/immich-test` with the
sidecar attached. Tear it down with
`docker compose -f /home/ryan/Code/immich-test/docker/docker-compose.yml down -v`
(`-v` also drops the postgres volume).

## 6. Docker socket access is temporary

It came from `setfacl` on `/var/run/docker.sock`, which does not survive a
docker daemon restart. For something permanent: `sudo usermod -aG docker $USER`,
then log out and back in.

## 7. Overlap with PR #13

PR #13 fixes the same `ONNXEmbedding` ImageNet-normalisation bug this work
surfaced independently, so the duplicate fix was dropped from this branch and
PR #13 owns it. One difference worth a follow-up: #13 hardcodes the ImageNet
constants on the class, while this branch's version read mean/std from the
model's `.json` sidecar, which survives a future export trained on different
statistics. The sidecar service reads the JSON either way, so nothing here
depends on it.
