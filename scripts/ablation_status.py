"""Regenerate the ablation's live status file from ground truth.

Every section is derived from ``results.csv``, ``latency.csv`` and the queue
log, so the file cannot drift from what actually happened. Health checks append
one line each; anything hand-written lives in ``notes.md`` and is spliced in
untouched.

    uv run python scripts/ablation_status.py
    uv run python scripts/ablation_status.py --health "HEALTHY cell 2/7 epoch 18"

Writes ``outputs/ablation/STATUS.md``.
"""

import argparse
import csv
import datetime
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
STATUS_MD = OUTPUT_DIR / "STATUS.md"
HEALTH_LOG = OUTPUT_DIR / "health.log"
NOTES_MD = OUTPUT_DIR / "notes.md"
QUEUE_LOG = OUTPUT_DIR / "stage_b.log"
LATENCY_CSV = OUTPUT_DIR / "latency.csv"
RESULTS_CSV = OUTPUT_DIR / "results.csv"

# The queue log carries legacy tqdm output with no newlines, so only the tail
# is scanned and every read is capped.
LOG_TAIL_BYTES = 400_000
HEALTH_LINES = 12

# The cells this sweep owes. Ceiling rows recorded earlier are NOT part of it,
# so completion is counted, never judged.
STAGE_B_CELLS = [
    ("resnet50", 42),
    ("resnet50", 1),
    ("resnet50", 2),
    ("convnext_tiny", 42),
    ("convnext_tiny", 1),
    ("convnext_tiny", 2),
    ("megadescriptor_t_224", 42),
]

# Stage C head A/B, on both deploy candidates so selection keeps two options.
STAGE_C_CELLS = [
    ("convnext_tiny", 42),
    ("convnext_tiny", 1),
    ("convnext_tiny", 2),
    ("resnet50", 42),
    ("resnet50", 1),
    ("resnet50", 2),
]


def _tail_text(path: Path, num_bytes: int) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - num_bytes))
        return f.read().decode("utf-8", errors="replace")


def queue_state() -> list[str]:
    """Which cell the sweep is on, and whether it is still moving."""
    running = subprocess.run(
        ["pgrep", "-af", "scripts/run_ablation.py"],
        capture_output=True,
        text=True,
    ).stdout
    # Match the worker, not a shell whose command line merely quotes the path.
    workers = [line for line in running.splitlines() if "/.venv/bin/python" in line]

    text = _tail_text(QUEUE_LOG, LOG_TAIL_BYTES)
    cells = re.findall(r"=== \[(\d+)/(\d+)\] Ablation: (.{0,140}?) ===", text)
    epochs = re.findall(r"Epoch (\d+)/(\d+): ([^\n]{0,110})", text)
    phases = re.findall(r"=== (Phase \d[^=]{0,40}|Linear probe[^=]{0,40}) ===", text)

    lines = []
    if workers:
        pid = workers[0].split()[0]
        lines.append(f"- **Running** (pid {pid})")
    else:
        lines.append("- **Not running** — queue finished, or stopped")
    if cells:
        n, total, desc = cells[-1]
        lines.append(
            f"- Cell **{n} of {total}** in the current sweep: `{desc.strip()}`"
        )
    if phases:
        lines.append(f"- Phase: {phases[-1].strip()}")
    if epochs:
        cur, total, rest = epochs[-1]
        lines.append(f"- Epoch **{cur}/{total}** — {rest.strip()}")
    if QUEUE_LOG.exists():
        mtime = datetime.datetime.fromtimestamp(QUEUE_LOG.stat().st_mtime)
        age = (datetime.datetime.now() - mtime).total_seconds() / 60
        flag = " ⚠️ **stalled?**" if workers and age > 15 else ""
        lines.append(f"- Log last written {age:.0f} min ago{flag}")
    return lines


def _progress(cells: list[tuple[str, int]], head: str, label: str) -> list[str]:
    """Tick off expected cells; completion is a count, not an impression."""
    done = {}
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("mode") != "finetune" or row.get("head") != head:
                    continue
                try:
                    key = (row["backbone"], int(row["seed"]))
                except (KeyError, ValueError):
                    continue
                done[key] = row

    lines, complete = [], 0
    for backbone, seed in cells:
        row = done.get((backbone, seed))
        if row is None:
            lines.append(f"- [ ] {backbone} seed {seed}")
        elif (row.get("status") or "ok") == "failed":
            lines.append(f"- [x] {backbone} seed {seed} — **FAILED**")
            complete += 1
        else:
            lines.append(
                f"- [x] {backbone} seed {seed} — MRR {row.get('mrr')} "
                f"top1 {row.get('top1')}"
            )
            complete += 1
    header = f"**{complete}/{len(cells)} cells recorded**"
    if complete == len(cells):
        header += f" — {label} COMPLETE"
    return [header, ""] + lines


def stage_b_progress() -> list[str]:
    return _progress(STAGE_B_CELLS, "arcface", "Stage B")


def stage_c_progress() -> list[str]:
    return _progress(STAGE_C_CELLS, "subcenter_arcface", "Stage C")


def _table(path: Path, columns: list[str], where=None) -> list[str]:
    if not path.exists():
        return ["_(none yet)_"]
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if where is None or where(r)]
    if not rows:
        return ["_(none yet)_"]
    out = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        out.append("| " + " | ".join((row.get(c) or "–") for c in columns) + " |")
    return out


def summarize(mode: str) -> list[str]:
    """Shell out to the decision rule so there is one implementation of it."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/summarize_ablation.py", "--mode", mode],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    body = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return body or ["_(no cells yet)_"]


def build() -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "# Ablation status",
        "",
        "_Regenerated by `scripts/ablation_status.py`. Do not edit by hand —",
        "hand-written notes belong in `notes.md`, which is spliced in below._",
        "",
        f"Updated: {now}",
        "",
        "## Queue",
        "",
        *queue_state(),
        "",
        "## Stage B — progress",
        "",
        *stage_b_progress(),
        "",
        "## Stage C — progress (head A/B: sub-center ArcFace)",
        "",
        *stage_c_progress(),
        "",
        "## Stage B — decision (fine-tuned, the numbers that ship)",
        "",
        *summarize("finetune"),
        "",
        "## Stage A — ranking (linear probe)",
        "",
        *summarize("probe"),
        "",
        "## Cost axis (all backbones, one instrument)",
        "",
        *_table(
            LATENCY_CSV,
            ["backbone", "license", "params_total_M", "torch_ms", "ort_ms", "onnx_mb"],
        ),
        "",
        "## Failed cells",
        "",
        *_table(
            RESULTS_CSV,
            ["backbone", "mode", "seed", "error"],
            where=lambda r: (r.get("status") or "") == "failed",
        ),
        "",
        f"## Health log (last {HEALTH_LINES})",
        "",
        "```",
        *(_tail_text(HEALTH_LOG, 8000).splitlines()[-HEALTH_LINES:] or ["(none)"]),
        "```",
    ]
    if NOTES_MD.exists():
        parts += ["", "---", "", NOTES_MD.read_text().strip()]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--health", default=None, help="Append one health-check line, then rebuild."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.health:
        stamp = datetime.datetime.now().strftime("%m-%d %H:%M")
        with open(HEALTH_LOG, "a") as f:
            f.write(f"{stamp}  {args.health}\n")

    STATUS_MD.write_text(build())
    print(f"Wrote {STATUS_MD}")


if __name__ == "__main__":
    main()
