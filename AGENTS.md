# Agent notes

Read [README.md](README.md) first: setup, commands, architecture, and the script map live there. The rules below are the ones you can't infer from the repo.

- **Always prefix commands with `uv run`** (`uv run pytest`, `uv run ruff check .`). Non-interactive shells don't get mise's venv activation, so a bare `pytest` fails.
- **In a git worktree**, reuse the main checkout's venv instead of syncing a second copy of torch: `UV_PROJECT_ENVIRONMENT=<repo-root>/.venv uv run --no-sync ...`. `git rev-parse --git-dir` prints `.git/worktrees/<name>` when you're in one; the repo root is everything before `/.git/`.
- **Add dependencies with `uv add <pkg>`** (or `uv add --group dev <pkg>`). Never hand-edit version pins in `pyproject.toml`.
- **Keep docstrings and comments short.** Say why, not what.
- **Never commit absolute paths.** The repo is cloned on several machines; use paths relative to the repo root or `animal_id/common/constants.py`.
- **`.planning/` holds dated design docs, not instructions.** `4-16-2026-production-audit/` is the roadmap the P0/P1 PRs come from. `6-25-2026-embedding-backbone-ablation/` drives `scripts/run_ablation.py`.
