#!/usr/bin/env python
"""
Utility functions for the animal_id project.
"""

import re
from pathlib import Path
from typing import Optional


def find_latest_run(project_dir: Path, run_name: str) -> Optional[Path]:
    """
    Finds the latest training run directory.

    Args:
        project_dir: The directory where training runs are stored (e.g., 'models/phase1').
        run_name: The base name for the run (e.g., 'detector_run').

    Returns:
        The path to the latest run directory (e.g., 'models/phase1/detector_run5'),
        or None if no matching directories are found.
    """
    if not project_dir.is_dir():
        return None

    run_dirs = [
        d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)
    ]

    if not run_dirs:
        return None

    latest_run_dir = None
    max_num = -1

    # Regex to find a potential number suffix (e.g., 'run_name2', 'run_name10')
    pattern = re.compile(rf"^{re.escape(run_name)}(\d*)$")

    for d in run_dirs:
        match = pattern.match(d.name)
        if match:
            num_str = match.group(1)
            num = int(num_str) if num_str else 0
            if num > max_num:
                max_num = num
                latest_run_dir = d


def find_latest_timestamped_run(runs_dir: Path = Path("runs")) -> Optional[Path]:
    """
    Finds the latest timestamped training run directory.

    Args:
        runs_dir: The directory where training runs are stored (default: 'runs').

    Returns:
        The path to the latest run directory (e.g., 'runs/20250907_155408_resnet50'),
        or None if no run directories are found.
    """
    if not runs_dir.exists():
        return None

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Find the most recent run directory by modification time
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    return latest_run
