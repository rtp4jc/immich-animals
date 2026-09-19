# Open decisions

Written 2026-09-19, after the sidecar was built and verified end-to-end against
Immich v3.2.2. Nothing here blocks using it — these are choices I made a
defensible default for and want confirmed.

## 1. Should dogs and humans coexist?

Right now pointing Immich at the sidecar means **human faces stop being
detected**, because we answer `facial-recognition` ourselves and never forward
it. That matches the plan ("Immich thinks it is finding human faces, it gets
dogs") and is fine for a dedicated dog library.

If you want both, it is about eight lines: forward the same
`facial-recognition` request upstream too and concatenate its `faces` with
ours. Immich would then cluster people and dogs into one People tab, using one
`maxDistance` for two very different embedding geometries — that shared
threshold is the real cost, not the code.

**Default if you say nothing:** dogs only, as built.

## 2. Min Detection Score

Immich's default is `0.7`, tuned for human faces. On the test set that found
20 faces in 18 of 31 photos. At `0.3` it found 28 in 25, and the extra
detections looked right. `sidecar/README.md` tells you to set `0.3`.

Two things to decide:

- Whether `0.3` holds on your real library, or wants a proper sweep. Five
  identities and 31 Wikimedia photos is not a tuning set.
- Whether the sidecar should ignore Immich's `minScore` entirely and use its
  own constant. Right now we honour whatever the admin setting says, which
  keeps the knob where a user expects it but means a fresh install silently
  behaves badly at the 0.7 default.

## 3. Max Distance / `clustering.eps`

I used `0.35` because that is what `models/onnx/embedding.json` selected. Worth
knowing what that pick optimised: at `eps=0.35` the sweep reports v_measure
`0.828` but purity `0.660`; at `eps=0.30` it is v_measure `0.820` and purity
`0.797`. The sweep chose on v-measure, so it accepted a lot of merging.

For a photo library, wrongly merging two dogs into one person is more annoying
than splitting one dog across two people, which argues for `0.30`. The live
test at `0.35` looked good, but 31 photos will not surface the difference.

**Default if you say nothing:** `0.35`, as the model sidecar says.

## 4. Dependency duplication

The sidecar's runtime deps are declared twice: `sidecar/requirements.txt` for
the container, and the `dev` group in `pyproject.toml` so `smoke_test.py` runs
locally. The alternative is making `sidecar/` its own uv project. I kept the
duplication because it is five package names and it keeps the image free of
torch.

## 5. The test instance

A full Immich v3.2.2 stack is running from `/home/ryan/Code/immich-test` with
31 dog photos and the sidecar attached. Tear it down with
`docker compose -f /home/ryan/Code/immich-test/docker/docker-compose.yml down -v`
(the `-v` also drops the postgres volume). I left it up so you can look at the
People tab.

## 6. Docker socket access is temporary

Access came from `setfacl` on `/var/run/docker.sock`, which does not survive a
docker daemon restart. For something permanent:
`sudo usermod -aG docker $USER`, then log out and back in.
