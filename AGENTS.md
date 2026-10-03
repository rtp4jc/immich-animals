# Agent notes

Read [README.md](README.md) first: setup, commands, architecture, and the script map live there. The rules below are the ones you can't infer from the repo.

## Tenets

When two goals conflict, these decide.

- **Recall over precision, within limits.** A missed dog is worse than an occasional false detection, as long as false positives stay rare enough not to clutter the People tab.
- **Over-fragment rather than over-merge.** Merging two people in Immich is easy; splitting one is not.
- **Don't regress existing classes to ship a new one.** A new species waits until the classes we already support hold their numbers.
- **Keep the Immich integration thin until there is an official one.** Immich changes quickly and the models are what matter, so the sidecar does the minimum and avoids anything that would conflict with upstream.
- **State of the art within Immich's model budget.** Aim for the best accuracy achievable at the size and latency of Immich's existing face and CLIP models.
- **Reuse through generic contracts.** Define the problem (detection, embedding, evaluation) as contracts independent of any model or data provider, so backbones, losses and datasets plug in. Think before adding code, and restructure when the first shape turns out wrong.
- **Benchmark on data that looks like a real library.** Select and judge models on owner-style photos of identities never seen in training, not on the training distribution.
- **Write drift-prone details down once, where they're owned.** Metrics, model names and counts live in the model card or the config that sets them; other docs link there instead of restating them.
- **Never harm the user's existing Immich data.** Human faces pass through untouched, and adding or removing the sidecar takes only a Refresh, never a Reset.

## Environment

- **Always prefix commands with `uv run`** (`uv run pytest`, `uv run ruff check .`). Non-interactive shells don't get mise's venv activation, so a bare `pytest` fails.
- **In a git worktree**, reuse the main checkout's venv instead of syncing a second copy of torch: `PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=<repo-root>/.venv uv run --no-sync ...`. `PYTHONPATH` matters: the venv's editable install points at whichever checkout last ran `uv sync`, so without it scripts import someone else's `animal_id`. `git rev-parse --git-dir` prints `.git/worktrees/<name>` when you're in one; the repo root is everything before `/.git/`.
- **Add dependencies with `uv add <pkg>`** (or `uv add --group dev <pkg>`). Never hand-edit version pins in `pyproject.toml`.
- **Never commit absolute paths.** The repo is cloned on several machines; use paths relative to the repo root or `animal_id/common/constants.py`.
- **`.planning/` holds dated design docs, not instructions.** Rationale, stage descriptions, and decisions go there (or in the PR), never in code files.
- **Model card numbers come from `scripts/evaluate_release.py`; metric definitions live in `animal_id/identification/households.py`.** Never quote a number from elsewhere.

## Writing code

The bar is: the smallest diff that does the job, with nothing a reader has to skip over. These rules apply to every file you touch, not only new ones: if you edit a file that breaks them, fix that part too. Concretely:

- **Grep before you write.** Before adding a function, search `animal_id/` and `scripts/` for one that already does it. If two scripts need the same logic, it moves into `animal_id/` in the same PR. Never copy a function between files, even "to keep them independent".
- **Scripts are thin.** A script is argparse plus calls into `animal_id/`. If a script grows past ~150 lines, the logic belongs in the package. Extend an existing script with a flag or subcommand (see `train_master.py`) before creating a new file, and never create a family of scripts for one workflow.
- **Module docstring: at most 5 lines.** What it does and one example invocation. No stage descriptions, no design rationale, no option-by-option tutorial: `--help` and `.planning/` cover those.
- **Function docstrings: one line or none.** Skip it when the name and type hints already say it. No `Args:`/`Returns:` sections.
- **Comments explain why, never what.** Delete any comment that restates the next line. A good comment names a constraint the reader can't see (an API quirk, a train/serve contract, a numeric reason).
- **Handle only cases that can happen.** No `try/except` around code that shouldn't fail, no `None` checks on required arguments, no fallback branches for callers that don't exist. Let it raise.
- **No abstraction for one caller.** No base classes, registries, config objects, or flags with a single user. Three similar lines beat a helper used once. Add the abstraction when the second caller arrives.
- **Replace, don't accumulate.** When new code supersedes old, delete the old path in the same PR. No `_v2`, no "legacy" branches, no deprecated wrappers.
- **Tests cover behaviour, not plumbing.** One test per behaviour through the public function. Don't test argparse wiring, trivial wrappers, or that a constant has its value.

Before you finish: run `uv run ruff check . && uv run ruff format .` and `uv run pytest`, then re-read the diff and delete every line that doesn't change behaviour or explain a non-obvious why.
