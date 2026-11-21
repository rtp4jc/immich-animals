## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Evidence

<!-- See "Proving a change" in AGENTS.md. Delete the lines that don't apply. -->

- [ ] Tested merged into current `main`
- [ ] `uv run ruff check .`, `uv run pytest` and `uv run --project sidecar pytest sidecar` pass
- [ ] Shouldn't change the sidecar's output: `evaluate_sidecar.py --baseline main` reports 0 changed photos
- [ ] Changes what the sidecar returns: `evaluate_sidecar.py` before and after, and the changed photos it lists, are pasted below
- [ ] Hardware or runtime variant: numbers from that hardware, under the documented memory limit
- [ ] New setting: its README row says who would pick a different value, and why
- [ ] New download in the build: checksummed

<!-- Paste evaluation output here. -->
