# Contributing

Thanks for helping! Bug fixes, docs and small improvements can go straight to a
pull request. For anything that changes which animals are found or how they're
grouped, open an issue first with a few example photos: whether a change like
that is a net win gets decided on the benchmark data, and it's cheaper to check
before you write the code.

## The rules

[AGENTS.md](AGENTS.md) has the project's tenets, the evidence each kind of change
needs, and the coding rules. It's written for AI coding assistants, which most
of them read on their own, but it applies to everyone. Pointing your assistant
at it before you start saves a round of review.

## Setup and tests

- **Sidecar only:** `uv run --project sidecar pytest sidecar` needs no models or
  downloads. See [Development](sidecar/README.md#development).
- **Everything else**, including the evaluation: [Setup](README.md#setup) in the
  main README.

If your change can affect what the sidecar returns, the
[sidecar README](sidecar/README.md#comparing-against-main) shows how to compare
it against `main` on the public validation set. Paste the output in the PR.

## What happens to your PR

- CI on a first pull request waits until a maintainer approves the run.
- Keep "Allow edits by maintainers" on. Small fixes and merges with `main` may be
  pushed to your branch as separate commits, without a force-push, rather than
  sent back to you.
