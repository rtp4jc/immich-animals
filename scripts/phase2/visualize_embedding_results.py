import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
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

def get_all_embeddings(model, data_loader, device):
    """
    Computes and returns embeddings for the entire dataset.
    """
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
            # Get file paths from dataset
            start_idx = i * data_loader.batch_size
            end_idx = start_idx + len(images)
            all_paths.extend([data_loader.dataset.annotations[j]['file_path'] for j in range(start_idx, end_idx)])

    return torch.cat(all_embeddings), all_labels, all_paths

def visualize_neighbors(num_queries=5, num_neighbors=5):
    """
    Loads the trained model, finds nearest neighbors for a few query images,
    and saves a visualization of the results.
    """
    print("--- Nearest Neighbor Visualization ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run training first.")
        return
    
    model = EmbeddingNet(embedding_dim=512)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("This can happen if the model architecture in embedding_model.py does not match the saved weights.")
        return
        
    model.to(device)

    # Create data loader for validation set
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = IdentityDataset(json_path=VAL_JSON_PATH, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Get all embeddings
    all_embeddings, all_labels, all_paths = get_all_embeddings(model, val_loader, device)

    # Select random query images
    query_indices = np.random.choice(len(all_paths), num_queries, replace=False)

    # --- Plotting ---
    fig, axes = plt.subplots(num_queries, num_neighbors + 1, figsize=((num_neighbors + 1) * 2.5, num_queries * 2.5))
    fig.suptitle('Nearest Neighbor Search Results', fontsize=16, y=1.03)

    for i, query_idx in enumerate(query_indices):
        query_embedding = all_embeddings[query_idx]
        query_label = all_labels[query_idx]
        query_path = all_paths[query_idx]

        # Compute cosine similarity
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
        
        # Set similarity of query to itself to a low value to exclude it
        similarities[query_idx] = -1

        # Get top N neighbors
        top_k_indices = torch.topk(similarities, num_neighbors).indices

        # Plot query image
        query_img = Image.open(query_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        ax = axes[i, 0]
        ax.imshow(query_img)
        ax.set_title(f"Query\nID: {query_label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Query Image", rotation=90, size='large', labelpad=20)

        # Plot neighbor images
        for j, neighbor_idx in enumerate(top_k_indices):
            neighbor_path = all_paths[neighbor_idx]
            neighbor_label = all_labels[neighbor_idx]
            neighbor_sim = similarities[neighbor_idx].item()

            neighbor_img = Image.open(neighbor_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
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
    print(f"Saved nearest neighbor plot to {save_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize embedding search results.")
    parser.add_argument('--num-queries', type=int, default=5, help="Number of query images to show.")
    parser.add_argument('--num-neighbors', type=int, default=5, help="Number of nearest neighbors to find.")
    args = parser.parse_args()
    
    visualize_neighbors(num_queries=args.num_queries, num_neighbors=args.num_neighbors)
