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
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import torch.nn.functional as F
import os
import sys
import argparse
import matplotlib.pyplot as plt
from PIL import Image

# Adjust path to import from our new package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dog_id.common.datasets import IdentityDataset
from dog_id.embedding.models import EmbeddingNet
from dog_id.embedding.config import TRAINING_CONFIG, DATA_CONFIG, DEFAULT_BACKBONE

# --- Configuration ---
FAR_TARGETS = [1e-1, 1e-2, 1e-3, 1e-4]

def get_all_embeddings(model, data_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader, desc="Generating Embeddings")):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.tolist())
            start_idx = i * data_loader.batch_size
            end_idx = start_idx + len(images)
            all_paths.extend([data_loader.dataset.annotations[j]['file_path'] for j in range(start_idx, end_idx)])
    return torch.cat(all_embeddings), all_labels, all_paths

def calculate_metrics(all_embeddings, all_labels):
    print("\n--- Calculating Verification Metrics (TAR@FAR) ---")
    labels = np.array(all_labels)
    positive_pairs = []
    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for _, idxs in tqdm(labels_to_indices.items(), desc="Generating Positive Pairs"):
        if len(idxs) > 1:
            positive_pairs.extend(list(combinations(idxs, 2)))

    num_negative_pairs = len(positive_pairs) * 2
    negative_pairs = []
    all_indices = set(range(len(labels)))
    pbar = tqdm(total=num_negative_pairs, desc="Generating Negative Pairs")
    while len(negative_pairs) < num_negative_pairs:
        idx1, idx2 = np.random.choice(list(all_indices), 2, replace=False)
        if labels[idx1] != labels[idx2] and (idx1, idx2) not in negative_pairs and (idx2, idx1) not in negative_pairs:
            negative_pairs.append((idx1, idx2))
            pbar.update(1)
    pbar.close()

    print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")
    pos_scores = np.array([F.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0)).item() for i, j in tqdm(positive_pairs, desc="Calculating Positive Scores")])
    neg_scores = np.array([F.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0)).item() for i, j in tqdm(negative_pairs, desc="Calculating Negative Scores")])

    print("\n--- Verification Metrics ---")
    for far in FAR_TARGETS:
        threshold = np.quantile(neg_scores, 1 - far)
        tar = np.sum(pos_scores > threshold) / len(pos_scores)
        print(f"TAR @ FAR={far*100:.3f}%: {tar*100:.2f}%  (Threshold: {threshold:.4f})")

def visualize_neighbors(all_embeddings, all_labels, all_paths, num_queries=5, num_neighbors=5):
    print("\n--- Nearest Neighbor Visualization ---")
    query_indices = np.random.choice(len(all_paths), num_queries, replace=False)
    fig, axes = plt.subplots(num_queries, num_neighbors + 1, figsize=((num_neighbors + 1) * 2.5, num_queries * 2.5))
    fig.suptitle('Nearest Neighbor Search Results', fontsize=16, y=1.03)

    for i, query_idx in enumerate(query_indices):
        query_embedding = all_embeddings[query_idx]
        query_label = all_labels[query_idx]
        query_path = all_paths[query_idx]
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
        similarities[query_idx] = -1
        top_k_indices = torch.topk(similarities, num_neighbors).indices

        query_img = Image.open(query_path).convert('RGB').resize((DATA_CONFIG['IMG_SIZE'], DATA_CONFIG['IMG_SIZE']))
        ax = axes[i, 0]
        ax.imshow(query_img)
        ax.set_title(f"Query\nID: {query_label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0: ax.set_ylabel("Query Image", rotation=90, size='large', labelpad=20)

        for j, neighbor_idx in enumerate(top_k_indices):
            neighbor_path = all_paths[neighbor_idx]
            neighbor_label = all_labels[neighbor_idx]
            neighbor_sim = similarities[neighbor_idx].item()
            neighbor_img = Image.open(neighbor_path).convert('RGB').resize((DATA_CONFIG['IMG_SIZE'], DATA_CONFIG['IMG_SIZE']))
            ax = axes[i, j + 1]
            ax.imshow(neighbor_img)
            title_color = 'green' if neighbor_label == query_label else 'red'
            ax.set_title(f"ID: {neighbor_label}\nSim: {neighbor_sim:.3f}", fontsize=10, color=title_color)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir = 'outputs/phase2_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'nearest_neighbors.png')
    plt.savefig(save_path)
    print(f"\nSaved nearest neighbor plot to {save_path}")
    plt.show()

def main(args):
    model_path = TRAINING_CONFIG['MODEL_OUTPUT_PATH']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingNet(backbone_name=args.backbone, embedding_dim=TRAINING_CONFIG['EMBEDDING_DIM'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    transform = transforms.Compose([transforms.Resize((DATA_CONFIG['IMG_SIZE'], DATA_CONFIG['IMG_SIZE'])), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_dataset = IdentityDataset(json_path=DATA_CONFIG['VAL_JSON_PATH'], transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)

    all_embeddings, all_labels, all_paths = get_all_embeddings(model, val_loader, device)

    if args.show_neighbors:
        visualize_neighbors(all_embeddings, all_labels, all_paths, args.num_queries, args.num_neighbors)
    
    if args.calculate_metrics:
        calculate_metrics(all_embeddings, all_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate a trained embedding model.")
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, help=f"Backbone of the trained model. Default: {DEFAULT_BACKBONE}")
    parser.add_argument('--show-neighbors', action='store_true', help="Visualize nearest neighbor search results.")
    parser.add_argument('--calculate-metrics', action='store_true', help="Calculate TAR@FAR verification metrics.")
    parser.add_argument('--num-queries', type=int, default=5, help="Number of query images for neighbor visualization.")
    parser.add_argument('--num-neighbors', type=int, default=5, help="Number of nearest neighbors to find.")
    args = parser.parse_args()

    if not args.show_neighbors and not args.calculate_metrics:
        print("No action specified. Running both validations.")
        args.show_neighbors = True
        args.calculate_metrics = True

    main(args)