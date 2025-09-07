"""
Visualizes the identity dataset to verify its integrity.

What it's for:
This script is a crucial debugging and verification tool. It allows you to visually inspect
the dataset created by `05_prepare_embedding_data.py` to ensure that the identity and
breed labels are correctly assigned to the images.

What it does:
1. Reads the `data/identity_val.json` file.
2. Groups the images by their assigned `identity_label`.
3. Selects a few identities that have multiple images.
4. Generates and displays a plot where each row corresponds to a single dog identity,
   showing all of its images from the validation set.

How to run it:
- This script should be run after `05_prepare_embedding_data.py`.
- Run from the root of the project:
  `python scripts/06_visualize_embedding_data.py`

How to interpret the results:
The script will save a plot to `outputs/phase2_visualizations/identity_verification.png`
and display it on screen.
- Each row in the plot is a unique dog.
- You should see that all images within a single row belong to the same dog.
- The labels on the y-axis show the `identity_label` and `breed_label` for that row.
- This confirms that the dataset construction was successful.
"""
import json
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
from PIL import Image
import sys
import os

# Adjust path to import from our new package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dog_id.embedding.config import DATA_CONFIG

# --- Configuration ---
NUM_IDENTITIES_TO_SHOW = 4
MIN_IMAGES_PER_ID = 3

def visualize_dataset_by_id():
    """
    Loads the validation dataset, groups images by identity,
    and displays a gallery to verify identity groupings.
    """
    val_json_path = DATA_CONFIG['VAL_JSON_PATH']
    img_size = DATA_CONFIG['IMG_SIZE']

    if not os.path.exists(val_json_path):
        print(f"Error: Validation JSON not found at {val_json_path}")
        print("Please run `scripts/05_prepare_embedding_data.py` first.")
        return

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    with open(val_json_path, 'r') as f:
        annotations = json.load(f)

    ids_to_images = defaultdict(list)
    for anno in annotations:
        ids_to_images[anno['identity_label']].append(anno)

    valid_ids = [id for id, annos in ids_to_images.items() if len(annos) >= MIN_IMAGES_PER_ID]
    if not valid_ids:
        print(f"No identities found with at least {MIN_IMAGES_PER_ID} images. Cannot create verification plot.")
        return
        
    ids_to_show = valid_ids[:NUM_IDENTITIES_TO_SHOW]
    
    max_cols = max(len(ids_to_images[id_val]) for id_val in ids_to_show)

    fig, axes = plt.subplots(len(ids_to_show), max_cols, figsize=(max_cols * 3, len(ids_to_show) * 3))
    if len(ids_to_show) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle('Verification of Identity Grouping', fontsize=16, y=1.03)

    for i, identity in enumerate(ids_to_show):
        image_annos = ids_to_images[identity]
        breed = image_annos[0]['breed_label']
        
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

        for j in range(len(image_annos), max_cols):
            axes[i, j].axis('off')

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    
    output_dir = 'outputs/phase2_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'identity_verification.png')
    plt.savefig(save_path)
    print(f"Saved verification plot to {save_path}")
    plt.show()

if __name__ == '__main__':
    visualize_dataset_by_id()
