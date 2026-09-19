#!/usr/bin/env python
"""Score the running sidecar against the held-out validation set.

Answers the two questions that pick Immich's settings:

  Min Detection Score — recall on identity photos versus the false-positive rate
  on dog-free photos. Every false positive becomes a junk "person" in the
  library, so the sweep has to show both sides.
  Max Distance — DBSCAN eps over the returned embeddings, scored the same way as
  the sweep in models/onnx/embedding.json.

Predictions are fetched once at the lowest threshold in the sweep and cached;
the sidecar returns a score per detection, so the rest of the sweep is free.

    uv run python scripts/evaluate_sidecar.py --url http://localhost:3005
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from animal_id.common.constants import DATA_DIR
from animal_id.common.logging_config import setup_logging
from animal_id.identification import cluster, cluster_quality

setup_logging()
logger = logging.getLogger(__name__)

TASK = "facial-recognition"
DEFAULT_DIR = DATA_DIR / "sidecar-validation"
SCORE_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7]
EPS_SWEEP = [0.25, 0.3, 0.35, 0.4, 0.45]


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
    """Detections per file, cached so threshold and eps sweeps are offline."""
    done = {} if refresh or not cache.exists() else json.loads(cache.read_text())
    todo = [r for r in records if r["path"] not in done]
    for rec in tqdm(todo, desc="predict", disable=not todo):
        try:
            reply = predict(url, root / rec["path"], min(SCORE_SWEEP))
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


def _hits(detections: list[dict], threshold: float) -> list[dict]:
    return [d for d in detections if d["score"] >= threshold]


def score_sweep(records: list[dict], preds: dict) -> None:
    """Recall on dogs and false-positive rate on non-dogs, per minScore."""
    groups = defaultdict(list)
    for rec in records:
        if rec["path"] not in preds:
            continue
        key = "negative" if rec["label"] == "negative" else f"identity/{rec['source']}"
        groups[key].append(preds[rec["path"]])

    header = f"{'minScore':>9}  " + "  ".join(
        f"{k.split('/')[-1][:10]:>18}" for k in sorted(groups)
    )
    print("\nDetection: share of photos with at least one detection")
    print(header)
    print("-" * len(header))
    for threshold in SCORE_SWEEP:
        cells = []
        for key in sorted(groups):
            rows = groups[key]
            fired = sum(1 for d in rows if _hits(d, threshold))
            per_photo = np.mean([len(_hits(d, threshold)) for d in rows])
            cells.append(f"{fired / len(rows):>10.3f} ({per_photo:>4.2f})")
        print(f"{threshold:>9.2f}  " + "  ".join(cells))
    print("  (share of photos, and mean detections per photo)")
    for key in sorted(groups):
        print(f"  {key}: {len(groups[key])} photos")


def negative_breakdown(records: list[dict], preds: dict, threshold: float) -> None:
    """Which kinds of non-dog photo the detector actually fires on."""
    by_category = defaultdict(lambda: [0, 0])
    for rec in records:
        if rec["label"] != "negative" or rec["path"] not in preds:
            continue
        name = rec["category"].replace("Category:Quality images of ", "")
        name = name.replace("Category:Featured pictures of ", "")
        by_category[name][1] += 1
        if _hits(preds[rec["path"]], threshold):
            by_category[name][0] += 1
    print(f"\nFalse positives by subject at minScore {threshold}")
    for name, (fired, total) in sorted(
        by_category.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)
    ):
        print(f"  {name:<28} {fired:>3}/{total:<4} {fired / max(total, 1):>6.3f}")


def eps_sweep(records: list[dict], preds: dict, threshold: float, source: str) -> None:
    """Cluster the best detection per identity photo, scored against the labels."""
    vectors, labels = [], []
    for rec in records:
        if rec["label"] == "negative" or rec["path"] not in preds:
            continue
        if source != "all" and rec["source"] != source:
            continue
        hits = _hits(preds[rec["path"]], threshold)
        if not hits:
            continue  # a miss is already counted by the detection sweep
        vectors.append(max(hits, key=lambda d: d["score"])["embedding"])
        labels.append(rec["label"])
    if not vectors:
        return

    embeddings = np.asarray(vectors, dtype=np.float32)
    header = (
        f"{'eps':>6}  {'clusters':>8}  {'true_ids':>8}  {'homog':>7}  "
        f"{'compl':>7}  {'v_meas':>7}  {'purity':>7}  {'noise':>7}"
    )
    print(f"\nClustering [{source}]: {len(labels)} crops, minScore {threshold}")
    print(header)
    print("-" * len(header))
    for eps in EPS_SWEEP:
        m = cluster_quality(labels, cluster(embeddings, eps=eps, min_samples=3))
        print(
            f"{eps:>6.2f}  {m['num_clusters']:>8d}  {m['num_true_identities']:>8d}  "
            f"{m['homogeneity']:>7.4f}  {m['completeness']:>7.4f}  "
            f"{m['v_measure']:>7.4f}  {m['purity']:>7.4f}  {m['noise_rate']:>7.4f}"
        )


def main(args: argparse.Namespace) -> None:
    root = Path(args.data)
    records = json.loads((root / "manifest.json").read_text())["files"]
    preds = collect(
        args.url, records, root, root / ".cache" / "predictions.json", args.refresh
    )
    logger.info(f"{len(preds)} of {len(records)} files predicted")

    score_sweep(records, preds)
    negative_breakdown(records, preds, args.min_score)
    for source in sorted({r["source"] for r in records if r["label"] != "negative"}) + [
        "all"
    ]:
        eps_sweep(records, preds, args.min_score, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:3003")
    parser.add_argument("--data", default=str(DEFAULT_DIR))
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="threshold for the clustering and breakdown tables",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached predictions"
    )
    main(parser.parse_args())
