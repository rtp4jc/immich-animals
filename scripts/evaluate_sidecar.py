#!/usr/bin/env python
"""Score a running sidecar on data/sidecar-validation: detection, false positives, clustering.

uv run python scripts/evaluate_sidecar.py --url http://localhost:3005 --cache main
uv run python scripts/evaluate_sidecar.py --url http://localhost:3006 --cache mine --baseline main
"""

import argparse
import json
import logging
from pathlib import Path

import requests
from tqdm import tqdm

from animal_id.benchmark import sidecar
from animal_id.common.constants import DATA_DIR
from animal_id.common.logging_config import setup_logging

logger = setup_logging(__name__, logging.INFO)

TASK = "facial-recognition"
DEFAULT_DIR = DATA_DIR / "sidecar-validation"


def predict(url: str, path: Path, min_score: float) -> dict:
    """POST one image exactly as Immich's machine-learning repository does."""
    entries = {
        TASK: {
            "detection": {"modelName": "buffalo_l", "options": {"minScore": min_score}},
            "recognition": {"modelName": "buffalo_l"},
        }
    }
    with path.open("rb") as fh:
        response = requests.post(
            f"{url}/predict",
            data={"entries": json.dumps(entries)},
            files={"image": (path.name, fh, "image/jpeg")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def collect(
    url: str, records: list[dict], root: Path, cache: Path, refresh: bool
) -> dict:
    """Detections per file at the sweep's lowest threshold, cached so every table is offline."""
    done = {} if refresh or not cache.exists() else json.loads(cache.read_text())
    todo = [r for r in records if r["path"] not in done]
    for rec in tqdm(todo, desc="predict", disable=not todo):
        try:
            reply = predict(url, root / rec["path"], min(sidecar.SCORE_SWEEP))
        except requests.RequestException as exc:
            logger.warning(f"{rec['path']}: {exc}")
            continue
        done[rec["path"]] = [
            {"score": f["score"], "embedding": json.loads(f["embedding"])}
            for f in reply[TASK]
        ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(done))
    return done


def main(args: argparse.Namespace) -> None:
    root = Path(args.data)
    records = json.loads((root / "manifest.json").read_text())["files"]
    cache = root / ".cache" / f"{args.cache}.json"
    preds = collect(args.url, records, root, cache, args.refresh)
    logger.info(f"{len(preds)} of {len(records)} files predicted")

    sidecar.score_sweep(records, preds)
    sidecar.negative_breakdown(records, preds, args.min_score)
    identities = [r for r in records if r["label"] != "negative"]
    for species in sorted({r["species"] for r in identities}):
        sources = sorted({r["source"] for r in identities if r["species"] == species})
        if len(sources) > 1:
            sources.append("all")
        for source in sources:
            sidecar.eps_sweep(records, preds, args.min_score, species, source)
    sidecar.eps_sweep(records, preds, args.min_score, "all", "all")
    if args.baseline:
        baseline = json.loads((root / ".cache" / f"{args.baseline}.json").read_text())
        sidecar.print_changes(records, preds, baseline, args.min_score, args.baseline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:3003")
    parser.add_argument("--data", default=str(DEFAULT_DIR))
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="threshold for the clustering, breakdown and baseline tables",
    )
    parser.add_argument(
        "--cache",
        default="predictions",
        help="cache file name, one per sidecar configuration",
    )
    parser.add_argument(
        "--baseline",
        help="an earlier run's --cache name; lists photos whose detections changed",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached predictions"
    )
    main(parser.parse_args())
