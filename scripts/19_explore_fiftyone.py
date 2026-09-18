#!/usr/bin/env python
"""
FiftyOne Explorer for Animal-ID Embeddings

Phase-2 visual companion to scripts/18_run_identification.py.

Loads the gallery images into a FiftyOne dataset, attaches ground-truth identity
labels and predicted DBSCAN cluster labels, computes a 2-D embedding visualization
(UMAP → t-SNE → PCA fallback chain) and a similarity index, then (optionally)
launches the FiftyOne App.

Usage:
    uv run python scripts/19_explore_fiftyone.py --split val --num-images 200
    uv run python scripts/19_explore_fiftyone.py --no-launch
    uv run python scripts/19_explore_fiftyone.py --viz-method tsne --eps 0.4
"""

import argparse
import logging
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np

from animal_id.common.constants import (
    ONNX_DETECTOR_PATH,
    ONNX_EMBEDDING_PATH,
    PROJECT_ROOT,
)
from animal_id.common.identity_loader import IdentityLoader
from animal_id.identification import cluster, embed_gallery
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


def _compute_visualization(
    dataset: fo.Dataset,
    embeddings: np.ndarray,
    method: str,
    brain_key: str,
) -> str:
    """Compute a 2-D visualization brain run, returning the method used.

    Falls back from 'umap' to 't-SNE' if umap-learn is not installed.
    """
    if method == "umap":
        try:
            fob.compute_visualization(
                dataset,
                embeddings=embeddings,
                method="umap",
                brain_key=brain_key,
            )
            return "umap"
        except Exception as exc:
            # FiftyOne raises ImportError (wrapped or not) when umap-learn is absent.
            exc_text = str(exc).lower()
            if (
                "umap" in exc_text
                or "importerror" in exc_text
                or "no module" in exc_text
            ):
                print(
                    "\numap-learn is not installed — falling back to t-SNE for "
                    "the embedding visualization.\n"
                    "  To enable UMAP: uv add umap-learn\n"
                )
                fob.compute_visualization(
                    dataset,
                    embeddings=embeddings,
                    method="tsne",
                    brain_key=brain_key,
                )
                return "tsne"
            raise

    fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        method=method,
        brain_key=brain_key,
    )
    return method


def main(args: argparse.Namespace) -> None:
    logger.info("Initializing ONNX pipeline...")
    pipeline = _build_pipeline()

    json_filename = _SPLIT_TO_JSON[args.split]
    logger.info(f"Loading gallery from split '{args.split}' ({json_filename})...")
    loader = IdentityLoader(json_filename=json_filename)
    items = loader.load_validation_data(num_images=args.num_images)
    logger.info(f"Loaded {len(items)} items from split.")

    items = _resolve_paths(items)

    logger.info("Embedding gallery...")
    embeddings, labels, paths = embed_gallery(pipeline, items, show_progress=True)
    num_embedded = len(labels)
    num_dropped = len(items) - num_embedded
    logger.info(
        f"Embedded {num_embedded} images; {num_dropped} dropped (no detection)."
    )

    if num_embedded == 0:
        logger.error("No embeddings produced — cannot proceed. Exiting.")
        return

    logger.info(
        f"Clustering with DBSCAN (eps={args.eps}, min_samples={args.min_samples})..."
    )
    pred = cluster(embeddings, eps=args.eps, min_samples=args.min_samples)
    num_clusters = int(np.max(pred)) + 1 if np.any(pred >= 0) else 0
    num_noise = int(np.sum(pred == -1))
    logger.info(f"Clusters found: {num_clusters}  Noise (unassigned): {num_noise}")

    dataset_name = f"animal_id_{args.split}"
    logger.info(f"Building FiftyOne dataset '{dataset_name}'...")

    samples = []
    for path, identity_label, cluster_id in zip(paths, labels, pred):
        sample = fo.Sample(filepath=path)
        sample["ground_truth"] = fo.Classification(label=str(identity_label))
        cluster_label = "unassigned" if cluster_id == -1 else f"cluster_{cluster_id}"
        sample["pred_cluster"] = fo.Classification(label=cluster_label)
        samples.append(sample)

    dataset = fo.Dataset(name=dataset_name, overwrite=True)
    dataset.add_samples(samples)
    logger.info(f"Dataset created with {len(dataset)} samples.")

    prefix = args.brain_key_prefix
    sim_key = f"{prefix}_sim"
    logger.info(f"Computing similarity index (brain_key='{sim_key}')...")
    fob.compute_similarity(dataset, embeddings=embeddings, brain_key=sim_key)
    logger.info("Similarity index done.")

    viz_key = f"{prefix}_viz"
    logger.info(
        f"Computing {args.viz_method.upper()} visualization (brain_key='{viz_key}')..."
    )
    actual_method = _compute_visualization(
        dataset,
        embeddings=embeddings,
        method=args.viz_method,
        brain_key=viz_key,
    )
    logger.info(f"Visualization done (method actually used: {actual_method}).")

    brain_keys = [sim_key, viz_key]

    if args.launch:
        print("\nLaunching FiftyOne App — press Ctrl-C to exit.\n")
        _print_app_tips(sim_key, viz_key)
        session = fo.launch_app(dataset)
        session.wait()
    else:
        print()
        print("=" * 60)
        print("FiftyOne dataset ready (App NOT launched)")
        print("=" * 60)
        print(f"  Dataset name  : {dataset_name}")
        print(f"  Samples       : {len(dataset)}")
        print(f"  Split         : {args.split}")
        print(f"  DBSCAN eps    : {args.eps}  min_samples: {args.min_samples}")
        print(f"  Clusters      : {num_clusters}  Noise: {num_noise}")
        print(f"  Brain keys    : {', '.join(brain_keys)}")
        print(f"  Viz method    : {actual_method}")
        print()
        print("To launch the App later, run:")
        print("    import fiftyone as fo")
        print(f'    dataset = fo.load_dataset("{dataset_name}")')
        print("    session = fo.launch_app(dataset)")
        print("    session.wait()")
        print()
        _print_app_tips(sim_key, viz_key)


def _print_app_tips(sim_key: str, viz_key: str) -> None:
    print("App tips:")
    print("  - Open the Embeddings panel and select brain key:", viz_key)
    print("  - Color by 'ground_truth' to see true identity clusters")
    print("  - Color by 'pred_cluster' to see DBSCAN-discovered clusters")
    print(
        f"  - Use 'Sort by similarity' (brain key: {sim_key}) on any selected"
        " sample to find similar animals"
    )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visual explorer for animal-ID embeddings using FiftyOne. "
            "Loads gallery images, attaches ground-truth + DBSCAN cluster labels, "
            "computes similarity + 2-D visualization, then (optionally) launches the App."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to explore (default: val).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        metavar="N",
        help="Number of images to load (default: all).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.4,
        metavar="E",
        help="DBSCAN eps threshold (default: 0.4, empirically-best operating point).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        metavar="M",
        help="DBSCAN min_samples (default: 3).",
    )
    parser.add_argument(
        "--viz-method",
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="2-D visualization method (default: umap; falls back to tsne if umap-learn absent).",
    )
    parser.add_argument(
        "--launch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the FiftyOne App after building the dataset (default: --launch).",
    )
    parser.add_argument(
        "--brain-key-prefix",
        type=str,
        default="animal",
        metavar="PREFIX",
        help="Prefix for FiftyOne brain run keys (default: animal).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
