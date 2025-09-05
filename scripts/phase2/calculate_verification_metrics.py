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

# Adjust the path to where your training module is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.datasets import IdentityDataset
from scripts.phase2.embedding_model import EmbeddingNet

# --- Configuration ---
VAL_JSON_PATH = 'data/identity_val.json'
IMG_SIZE = 224
MODEL_PATH = 'models/dog_embedding_best.pt'
BATCH_SIZE = 32
# FAR values to report TAR at
FAR_TARGETS = [1e-1, 1e-2, 1e-3, 1e-4]

def get_all_embeddings(model, data_loader, device):
    """
    Computes and returns embeddings for the entire dataset.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Generating Embeddings"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.tolist())

    return torch.cat(all_embeddings), all_labels

def generate_pairs(labels):
    """
    Generates positive and negative pairs from a list of labels.
    """
    labels = np.array(labels)
    indices = np.arange(len(labels))
    
    # Group indices by label
    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)

    positive_pairs = []
    for label, idxs in tqdm(labels_to_indices.items(), desc="Generating Positive Pairs"):
        if len(idxs) > 1:
            positive_pairs.extend(list(combinations(idxs, 2)))

    # Generate negative pairs
    # For simplicity, we'll sample randomly. For a rigorous result, all possible
    # negative pairs should be considered, but this is computationally expensive.
    num_negative_pairs = len(positive_pairs) * 2 # Sample a reasonable number
    negative_pairs = []
    all_indices = set(range(len(labels)))
    
    pbar = tqdm(total=num_negative_pairs, desc="Generating Negative Pairs")
    while len(negative_pairs) < num_negative_pairs:
        idx1, idx2 = np.random.choice(list(all_indices), 2, replace=False)
        if labels[idx1] != labels[idx2]:
            # Ensure pair is not already in the list (in any order)
            if (idx1, idx2) not in negative_pairs and (idx2, idx1) not in negative_pairs:
                negative_pairs.append((idx1, idx2))
                pbar.update(1)
    pbar.close()

    return positive_pairs, negative_pairs

def main():
    print("--- Calculating Verification Metrics (TAR@FAR) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run training first.")
        return
    
    model = EmbeddingNet(embedding_dim=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # Create data loader for validation set
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = IdentityDataset(json_path=VAL_JSON_PATH, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Get all embeddings
    all_embeddings, all_labels = get_all_embeddings(model, val_loader, device)

    # Generate pairs
    positive_pairs, negative_pairs = generate_pairs(all_labels)
    print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")

    # Calculate similarity scores
    pos_scores = []
    for i, j in tqdm(positive_pairs, desc="Calculating Positive Scores"):
        sim = F.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0))
        pos_scores.append(sim.item())

    neg_scores = []
    for i, j in tqdm(negative_pairs, desc="Calculating Negative Scores"):
        sim = F.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0))
        neg_scores.append(sim.item())

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # Calculate metrics
    print("\n--- Verification Metrics ---")
    for far in FAR_TARGETS:
        # Find the threshold that corresponds to the FAR
        threshold = np.quantile(neg_scores, 1 - far)
        
        # Calculate TAR at this threshold
        tar = np.sum(pos_scores > threshold) / len(pos_scores)
        
        print(f"TAR @ FAR={far*100:.3f}%: {tar*100:.2f}%  (Threshold: {threshold:.4f})")

if __name__ == '__main__':
    main()
