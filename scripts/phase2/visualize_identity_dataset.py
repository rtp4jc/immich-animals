import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

# Adjust the path to where your training module is located
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.datasets import IdentityDataset

# --- Configuration ---
VAL_JSON_PATH = 'data/identity_val.json'
BATCH_SIZE = 16
IMG_SIZE = 112

def visualize_dataset():
    """
    Loads the validation dataset and displays a batch of images
    with their identity and breed labels.
    """
    if not os.path.exists(VAL_JSON_PATH):
        print(f"Error: Validation JSON not found at {VAL_JSON_PATH}")
        print("Please run `scripts/phase2/prepare_identity_dataset.py` first.")
        return

    # Basic transform for visualization
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # Load dataset
    val_dataset = IdentityDataset(json_path=VAL_JSON_PATH, transform=transform)
    
    # Load breed labels from the json for display
    with open(VAL_JSON_PATH, 'r') as f:
        annotations = json.load(f)
    
    # Check if dataset is empty
    if len(val_dataset) == 0:
        print("Validation dataset is empty. Cannot visualize.")
        return

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get one batch
    images, identity_labels = next(iter(val_loader))

    # Get corresponding breed labels for the batch
    # This is a bit inefficient but fine for a small visualization script
    breed_labels = []
    # This is not robust if shuffle is on, we need to get the indices
    # For visualization, let's just pull from the first N items without shuffle
    
    val_loader_no_shuffle = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    images, identity_labels = next(iter(val_loader_no_shuffle))
    
    breed_labels = [anno['breed_label'] for anno in annotations[:images.size(0)]]


    # --- Visualization ---
    num_images = images.size(0)
    grid_size = math.ceil(math.sqrt(num_images))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy() # Convert from (C, H, W) to (H, W, C)
        identity = identity_labels[i].item()
        breed = breed_labels[i]

        axes[i].imshow(img)
        axes[i].set_title(f"ID: {identity}\nBreed: {breed}", fontsize=8)
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("Identity Dataset Samples", fontsize=16, y=1.02)
    
    # Save the figure
    output_dir = 'outputs/phase2_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'identity_dataset_sample.png')
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == '__main__':
    visualize_dataset()