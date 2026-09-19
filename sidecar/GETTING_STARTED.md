# Find your dogs in Immich

Immich groups the human faces in your photos into People. This adds a small
service that makes it do the same for individual dogs — not "there is a dog
here", but "this is the same dog as in those other 40 photos".

You keep your human faces. Nothing about Immich is modified or replaced; the
service sits next to it and answers the same questions Immich already asks.

> **Beta.** Try it on a test Immich or a library you would not mind rebuilding.
> Undoing it means re-running face detection.

## What you need

- Immich running under `docker compose` (v3.0 or newer)
- About 1 GB of disk and a few minutes of CPU per thousand photos

## 1. Add the service

Put this next to your Immich `docker-compose.yml`, in a file called
`docker-compose.override.yml`:

```yaml
services:
  animal-ml:
    container_name: animal_ml
    image: animal-ml
    environment:
      UPSTREAM_ML_URL: http://immich-machine-learning:3003
    restart: always
```

Then start it:

```bash
docker compose up -d animal-ml
```

## 2. Point Immich at it

In Immich: **Administration → Settings → Machine Learning**.

| Setting | Change it to |
| --- | --- |
| URL | `http://animal-ml:3003` |
| Min Detection Score | `0.3` |
| Max Distance | leave it alone |

**Min Detection Score** is the one that matters. Immich's default of 0.7 is
tuned for human faces and will miss about half your dogs. At 0.3 it finds
roughly 80%.

Leave **Max Distance** as it is — the service adjusts its own numbers to fit
whatever you already have.

Save, then check **Administration → Settings → Machine Learning** shows the
server as reachable.

## 3. Find the dogs

**Administration → Jobs → Face Detection → All**, and when that finishes,
**Facial Recognition → All**.

This re-scans your library, so it takes a while — roughly ten photos a second.
Leave it running; the People tab fills in as it goes.

If you only see a handful of people afterwards, that is Immich hiding anyone
with fewer than three photos. **Account Settings → Features → People** lets you
lower that.

## What to expect

Good photos of a dog you have a lot of pictures of work well. In testing, a dog
with 673 photos put 491 of them into one person. A dog with 11 photos scattered
across five.

So, realistically:

- **Dogs you photograph often** get one big cluster, plus a few strays to merge
- **Dogs with under ~15 photos** may not group at all
- **Similar-looking dogs** get mixed together — two black curly-coated dogs are
  genuinely hard, and the model confuses them the way you might in a thumbnail
- **Cats** occasionally get detected as a "person". About one cat photo in
  seven. Landscapes, buildings and photos of people do not.

Merging two people in Immich is easy; splitting one is not. So the settings
here lean towards leaving a few extra clusters for you to merge rather than
wrongly gluing two dogs together.

Name a dog the same way you name a person, and search works: typing the name
finds their photos.

## Turning it off

Point the Machine Learning URL back at `http://immich-machine-learning:3003`,
then re-run Face Detection. Your dog people will still be listed until you
delete them, and human faces come back on the next run.

## Options

Set these under `environment:` in the override file if you need them.

| Variable | Default | What it does |
| --- | --- | --- |
| `UPSTREAM_ML_URL` | — | Your existing Immich ML container. Required: search and OCR are forwarded to it. |
| `KEEP_HUMAN_FACES` | `true` | `false` serves dogs only and stops detecting human faces. Then set Max Distance to `0.35` yourself. |
| `DOG_MAX_DISTANCE` | `0.35` | How alike two dogs must look to be called the same dog. Lower splits more, higher merges more. |
| `IMMICH_MAX_DISTANCE` | `0.5` | The Max Distance in your Immich settings. Only change this if you changed that. |

## Something went wrong

**"Machine learning server became unhealthy"** — `animal-ml` is not running or
is not on the same docker network as Immich. Check `docker logs animal_ml`.

**Search stopped working** — `UPSTREAM_ML_URL` is wrong. Text search is passed
through to Immich's own ML container; if we cannot reach it, search fails.

**No dogs found at all** — Min Detection Score is probably still 0.7, or Face
Detection was run as "Missing" rather than "All".

**Lots of near-duplicate people** — expected, and merging them is the fastest
fix. If it is excessive, raise `DOG_MAX_DISTANCE` to `0.4` and re-run Facial
Recognition.
