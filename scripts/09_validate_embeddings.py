"""
Validates the performance of a trained embedding model.

What it's for:
This script provides a unified entry point for evaluating a trained embedding model.
It can perform both qualitative validation (visualizing nearest neighbors) and
quantitative validation (calculating TAR@FAR metrics).

What it does:
1. Loads the best trained embedding model.
2. Computes embedding vectors for all images in the validation set.
3. Based on command-line flags, it can either:
   - Visualize nearest neighbors: For a few random query images, it finds and
     plots the most similar images from the validation set.
   - Calculate TAR@FAR: It generates thousands of positive and negative pairs,
     calculates their similarity, and computes the True Accept Rate at various
     False Accept Rates.

How to run it:
- This script should be run after a model has been trained.
- To visualize nearest neighbors:
  `python scripts/07_validate_embeddings.py --show-neighbors`
- To calculate TAR@FAR metrics:
  `python scripts/07_validate_embeddings.py --calculate-metrics`
- To run both (default action):
  `python scripts/07_validate_embeddings.py`
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from animal_id.benchmark.metrics import evaluate_embedding_model
from animal_id.common.datasets import DogIdentityDataset

# Adjust path to import from our new package
from animal_id.common.utils import find_latest_timestamped_run
from animal_id.embedding.backbones import BackboneType
from animal_id.embedding.config import DATA_CONFIG, DEFAULT_BACKBONE, TRAINING_CONFIG
from animal_id.embedding.models import DogEmbeddingModel

# --- Configuration ---
FAR_TARGETS = [1e-1, 1e-2, 1e-3, 1e-4]


def visualize_neighbors(
    all_embeddings, all_labels, all_paths, num_queries=5, num_neighbors=5
):
    print("\n--- Nearest Neighbor Visualization ---")
    query_indices = np.random.choice(len(all_paths), num_queries, replace=False)
    fig, axes = plt.subplots(
        num_queries,
        num_neighbors + 1,
        figsize=((num_neighbors + 1) * 2.5, num_queries * 2.5),
    )
    fig.suptitle("Nearest Neighbor Search Results", fontsize=16, y=1.03)

    for i, query_idx in enumerate(query_indices):
        query_embedding = all_embeddings[query_idx]
        query_label = all_labels[query_idx]
        query_path = all_paths[query_idx]
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
        similarities[query_idx] = -1
        top_k_indices = torch.topk(similarities, num_neighbors).indices

        query_img = (
            Image.open(query_path)
            .convert("RGB")
            .resize((DATA_CONFIG["IMG_SIZE"], DATA_CONFIG["IMG_SIZE"]))
        )
        ax = axes[i, 0]
        ax.imshow(query_img)
        ax.set_title(f"Query\nID: {query_label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Query Image", rotation=90, size="large", labelpad=20)

        for j, neighbor_idx in enumerate(top_k_indices):
            neighbor_path = all_paths[neighbor_idx]
            neighbor_label = all_labels[neighbor_idx]
            neighbor_sim = similarities[neighbor_idx].item()
            neighbor_img = (
                Image.open(neighbor_path)
                .convert("RGB")
                .resize((DATA_CONFIG["IMG_SIZE"], DATA_CONFIG["IMG_SIZE"]))
            )
            ax = axes[i, j + 1]
            ax.imshow(neighbor_img)
            title_color = "green" if neighbor_label == query_label else "red"
            ax.set_title(
                f"ID: {neighbor_label}\nSim: {neighbor_sim:.3f}",
                fontsize=10,
                color=title_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir = "outputs/phase2_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "nearest_neighbors.png")
    plt.savefig(save_path)
    print(f"\nSaved nearest neighbor plot to {save_path}")
    plt.show()


def main(args):
    # Try to find model in latest run directory first
    latest_run = find_latest_timestamped_run()
    model_path = None

    if latest_run:
        model_path = latest_run / "best_model.pt"
        if not model_path.exists():
            model_path = None

    # Fall back to old location if not found
    if model_path is None:
        model_path = TRAINING_CONFIG["MODEL_OUTPUT_PATH"]
        if not os.path.exists(model_path):
            print("Error: No trained model found.")
            print(
                f"Checked: runs/*/best_model.pt and {TRAINING_CONFIG['MODEL_OUTPUT_PATH']}"
            )
            print("Please run training first.")
            return

    print(f"Using model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogEmbeddingModel(
        backbone_type=args.backbone,
        num_classes=1001,  # This doesn't matter for inference
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    val_dataset = DogIdentityDataset(
        json_path=DATA_CONFIG["VAL_JSON_PATH"],
        img_size=DATA_CONFIG["IMG_SIZE"],
        is_training=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=DATA_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2
    )

    if args.calculate_metrics:
        print("\n--- Calculating Verification Metrics (TAR@FAR) ---")
        metrics = evaluate_embedding_model(model, val_loader, device)
        print("\n--- Verification Metrics ---")
        print(f"  mAP: {metrics.get('mAP', 0.0):.4f}")
        print(f"  TAR@FAR=1.0%: {metrics.get('TAR@FAR=1%', 0.0):.4f}")
        print(f"  TAR@FAR=0.1%: {metrics.get('TAR@FAR=0.1%', 0.0):.4f}")

    if args.show_neighbors:
        # We need to manually get all embeddings and paths for visualization
        model.eval()
        all_embeddings = []
        all_labels = []
        all_paths = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(
                tqdm(val_loader, desc="Generating Embeddings for Visualization")
            ):
                images = images.to(device)
                embeddings = model.get_embeddings(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.extend(labels.tolist())
                start_idx = i * val_loader.batch_size
                end_idx = start_idx + len(images)
                all_paths.extend(
                    [
                        val_loader.dataset.annotations[j]["file_path"]
                        for j in range(start_idx, end_idx)
                    ]
                )

        all_embeddings = torch.cat(all_embeddings)
        visualize_neighbors(
            all_embeddings, all_labels, all_paths, args.num_queries, args.num_neighbors
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained embedding model.")
    parser.add_argument(
        "--backbone",
        type=BackboneType,
        default=DEFAULT_BACKBONE,
        choices=list(BackboneType),
        help=f"Backbone of the trained model. Default: {DEFAULT_BACKBONE}",
    )
    parser.add_argument(
        "--show-neighbors",
        action="store_true",
        help="Visualize nearest neighbor search results.",
    )
    parser.add_argument(
        "--calculate-metrics",
        action="store_true",
        help="Calculate TAR@FAR verification metrics.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of query images for neighbor visualization.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors to find.",
    )
    args = parser.parse_args()

    if not args.show_neighbors and not args.calculate_metrics:
        print("No action specified. Running both validations.")
        args.show_neighbors = True
        args.calculate_metrics = True

    main(args)
