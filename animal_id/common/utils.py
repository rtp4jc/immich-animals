"""Locating training run directories."""

import re
from pathlib import Path


def find_latest_run(project_dir: Path, run_name: str) -> Path | None:
    """Latest Ultralytics-style run directory, e.g. 'detector_run10' over 'detector_run2'.

    Returns None if project_dir has no directory matching run_name.
    """
    if not project_dir.is_dir():
        return None

    # Ultralytics appends an incrementing suffix: run_name, run_name2, run_name3...
    pattern = re.compile(rf"^{re.escape(run_name)}(\d*)$")

    numbered = [
        (int(match.group(1) or 0), d)
        for d in project_dir.iterdir()
        if d.is_dir() and (match := pattern.match(d.name))
    ]
    return max(numbered)[1] if numbered else None


def find_latest_timestamped_run(runs_dir: Path = Path("runs")) -> Path | None:
    """Most recently modified run directory, e.g. 'runs/20250907_155408_resnet50'."""
    if not runs_dir.exists():
        return None

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    return max(run_dirs, key=lambda d: d.stat().st_mtime) if run_dirs else None
