#!/usr/bin/env python
"""
Run Identification Validator

Phase-1 "Immich-like" animal identification validator:
  1. Embed a labeled gallery with the ONNX pipeline.
  2. Cluster embeddings into discovered identities (cosine DBSCAN, mirroring
     Immich face-clustering).
  3. Evaluate cluster quality vs ground-truth identity labels.
  4. Sweep the DBSCAN eps threshold and print a tidy table.
  5. Save results JSON to outputs/identification/<split>_<timestamp>.json.

Usage:
    uv run python scripts/18_run_identification.py --split val --num-images 200
    uv run python scripts/18_run_identification.py --split test --eps 0.4
    uv run python scripts/18_run_identification.py --eps-sweep 0.2 0.3 0.4 0.5 0.6
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

import numpy as np

from animal_id.common.constants import (
    ONNX_DETECTOR_PATH,
    ONNX_EMBEDDING_PATH,
    PROJECT_ROOT,
)
from animal_id.common.identity_loader import IdentityLoader
from animal_id.identification import cluster, cluster_quality, embed_gallery
from animal_id.pipeline.animal_pipeline import AnimalPipeline
from animal_id.pipeline.onnx_models import ONNXDetector, ONNXEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_SPLIT_TO_JSON = {
    "val": "identity_val.json",
    "test": "identity_test.json",
}

_DEFAULT_EPS_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _to_native(obj):
    """Recursively cast numpy scalar types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _build_pipeline() -> AnimalPipeline:
    """Construct the ONNX inference pipeline, erroring cleanly if models missing."""
    if not ONNX_DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Detector ONNX model not found: {ONNX_DETECTOR_PATH}\n"
            "Run scripts/11_export_detector_onnx.py first."
        )
    if not ONNX_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Embedding ONNX model not found: {ONNX_EMBEDDING_PATH}\n"
            "Run scripts/10_export_embedding_onnx.py first."
        )

    detector = ONNXDetector(str(ONNX_DETECTOR_PATH))
    embedding_model = ONNXEmbedding(str(ONNX_EMBEDDING_PATH))
    return AnimalPipeline(
        detector=detector,
        embedding_model=embedding_model,
        use_keypoints=False,
    )


def _resolve_paths(items: list[dict]) -> list[dict]:
    """Resolve relative image_path values against PROJECT_ROOT."""
    resolved = []
    for item in items:
        p = item["image_path"]
        if not Path(p).is_absolute():
            p = str(PROJECT_ROOT / p)
        resolved.append({**item, "image_path": p})
    return resolved


def _print_sweep_table(sweep_rows: list[dict]) -> None:
    """Print a tidy sweep table to stdout."""
    header = (
        f"{'eps':>6}  {'clusters':>8}  {'true_ids':>8}  "
        f"{'v_measure':>9}  {'ari':>7}  {'purity':>7}  {'noise_rate':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in sweep_rows:
        print(
            f"{row['eps']:>6.2f}  {row['num_clusters']:>8d}  "
            f"{row['num_true_identities']:>8d}  "
            f"{row['v_measure']:>9.4f}  {row['adjusted_rand_index']:>7.4f}  "
            f"{row['purity']:>7.4f}  {row['noise_rate']:>10.4f}"
        )
    print()


def main(args: argparse.Namespace) -> None:
    logger.info("Initializing ONNX pipeline...")
    pipeline = _build_pipeline()

    json_filename = _SPLIT_TO_JSON[args.split]
    logger.info(f"Loading gallery from split '{args.split}' ({json_filename})...")
    loader = IdentityLoader(json_filename=json_filename)
    items = loader.load_validation_data(num_images=args.num_images)
    logger.info(f"Loaded {len(items)} items from split.")

    # Resolve relative paths so the script works regardless of cwd.
    items = _resolve_paths(items)

    logger.info("Embedding gallery...")
    embeddings, labels, paths = embed_gallery(pipeline, items, show_progress=True)
    num_embedded = len(labels)
    num_dropped = len(items) - num_embedded
    logger.info(
        f"Embedded {num_embedded} images; {num_dropped} dropped (no detection)."
    )

    if num_embedded == 0:
        logger.error("No embeddings produced — cannot cluster. Exiting.")
        return

    eps_sweep = args.eps_sweep if args.eps_sweep is not None else _DEFAULT_EPS_SWEEP
    logger.info(f"Running eps sweep: {eps_sweep}")

    sweep_rows: list[dict] = []
    for eps_val in eps_sweep:
        pred = cluster(embeddings, eps=eps_val, min_samples=args.min_samples)
        metrics = cluster_quality(labels, pred)
        row = {"eps": eps_val, **metrics}
        sweep_rows.append(row)

    _print_sweep_table(sweep_rows)

    logger.info(f"Computing headline metrics at eps={args.eps}...")
    headline_pred = cluster(embeddings, eps=args.eps, min_samples=args.min_samples)
    headline_metrics = cluster_quality(labels, headline_pred)

    # Per-image assignments for the headline run.
    assignments = [
        {"path": path, "identity_label": label, "cluster": int(cluster_id)}
        for path, label, cluster_id in zip(paths, labels, headline_pred)
    ]

    outputs_dir = PROJECT_ROOT / "outputs" / "identification"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"{args.split}_{timestamp}.json"

    results_data = {
        "config": {
            "split": args.split,
            "num_images_requested": args.num_images,
            "num_images_embedded": num_embedded,
            "num_images_dropped": num_dropped,
            "min_samples": args.min_samples,
            "headline_eps": args.eps,
        },
        "eps_sweep": _to_native(sweep_rows),
        "headline_assignments": _to_native(assignments),
    }

    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)

    m = headline_metrics
    print("=" * 60)
    print(f"Headline results  (eps={args.eps}, min_samples={args.min_samples})")
    print("=" * 60)
    print(
        f"  Embedded / requested : {num_embedded} / {len(items)} ({num_dropped} dropped)"
    )
    print(f"  True identities      : {m['num_true_identities']}")
    print(f"  Clusters found       : {m['num_clusters']}")
    print(f"  V-measure            : {m['v_measure']:.4f}")
    print(f"  Adjusted Rand Index  : {m['adjusted_rand_index']:.4f}")
    print(f"  Purity               : {m['purity']:.4f}")
    print(f"  Noise rate           : {m['noise_rate']:.4f}")
    print(f"\nResults saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Immich-like animal identification clustering and evaluate quality.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        metavar="N",
        help="Number of images to sample (default: all).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        metavar="M",
        help="DBSCAN min_samples / Immich minFaces (default: 3).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        metavar="E",
        help="DBSCAN eps for headline single-run metrics (default: 0.5).",
    )
    parser.add_argument(
        "--eps-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="E",
        help=(
            "Space-separated list of eps values to sweep "
            "(default: 0.2 0.3 0.4 0.5 0.6 0.7)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
