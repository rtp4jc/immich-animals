import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
from PIL import Image

# Adjust the path to where your training module is located
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.datasets import IdentityDataset

# --- Configuration ---
VAL_JSON_PATH = 'data/identity_val.json'
IMG_SIZE = 224 # Larger size for better viewing
NUM_IDENTITIES_TO_SHOW = 4
MIN_IMAGES_PER_ID = 3

def visualize_dataset_by_id():
    """
    Loads the validation dataset, groups images by identity,
    and displays a gallery to verify identity groupings.
    """
    if not os.path.exists(VAL_JSON_PATH):
        print(f"Error: Validation JSON not found at {VAL_JSON_PATH}")
        print("Please run `scripts/phase2/prepare_identity_dataset.py` first.")
        return

    # Cleaner transform for visualization to avoid distortion
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor()
    ])

    with open(VAL_JSON_PATH, 'r') as f:
        annotations = json.load(f)

    # Group images by identity
    ids_to_images = defaultdict(list)
    for anno in annotations:
        ids_to_images[anno['identity_label']].append(anno)

    # Filter for identities with enough images to display
    valid_ids = [id for id, annos in ids_to_images.items() if len(annos) >= MIN_IMAGES_PER_ID]
    if not valid_ids:
        print(f"No identities found with at least {MIN_IMAGES_PER_ID} images. Cannot create verification plot.")
        return
        
    # Select which identities to show
    ids_to_show = valid_ids[:NUM_IDENTITIES_TO_SHOW]
    
    # Find max number of images for a single ID to set subplot columns
    max_cols = 0
    for id_to_show in ids_to_show:
        if len(ids_to_images[id_to_show]) > max_cols:
            max_cols = len(ids_to_images[id_to_show])

    fig, axes = plt.subplots(len(ids_to_show), max_cols, figsize=(max_cols * 3, len(ids_to_show) * 3))
    fig.suptitle('Verification of Identity Grouping', fontsize=16, y=1.03)

    for i, identity in enumerate(ids_to_show):
        image_annos = ids_to_images[identity]
        breed = image_annos[0]['breed_label'] # Breed is the same for all
        
        # Set row title
        axes[i, 0].set_ylabel(f'ID: {identity}\nBreed: {breed}', rotation=0, labelpad=60, verticalalignment='center', fontsize=10)


        for j, anno in enumerate(image_annos):
            img_path = anno['file_path']
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
            except FileNotFoundError:
                print(f"Warning: File not found: {img_path}. Skipping.")
                continue

            ax = axes[i, j]
            ax.imshow(image_tensor.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f'Image {j+1}')

        # Hide unused subplots in the row
        for j in range(len(image_annos), max_cols):
            axes[i, j].axis('off')

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    
    # Save the figure
    output_dir = 'outputs/phase2_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'identity_verification.png')
    plt.savefig(save_path)
    print(f"Saved verification plot to {save_path}")
    plt.show()


if __name__ == '__main__':
    visualize_dataset_by_id()
