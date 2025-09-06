"""
Main script for training the dog embedding model.

What it's for:
This script orchestrates the entire training process. It brings together the dataset,
model architecture, and loss functions to train a model that can effectively
distinguish between different dogs.

What it does:
Implements an advanced two-stage fine-tuning process:
1. Stage 1 (Warm-up): The script first freezes the pre-trained backbone of the
   model and trains only the newly added classification "head" for a few epochs.
   This allows the new layers to adapt safely without destabilizing the learned
   features.
2. Stage 2 (Fine-tuning): The script then unfreezes the entire model and trains all
   layers with a lower learning rate for the backbone. This allows the whole
   network to specialize for the dog identification task.
3. Early Stopping: In both stages, the script monitors the validation loss. If the
   loss does not improve for a set number of epochs (patience), training is
   stopped automatically to prevent overfitting and save time.
4. Model Saving: The script saves the model's weights to `models/dog_embedding_best.pt`
   whenever a new best validation loss is achieved.

How to run it:
- To run the full, advanced training process:
  `python training/train_embedding.py`
- To visualize a batch of augmented training data first:
  `python training/train_embedding.py --visualize-batch`

How to interpret the results:
The script will print the training and validation loss and accuracy for each epoch.
It will save the best performing model based on the lowest validation loss.
The final output will be the path to this saved model file.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import os
import sys
import math
import numpy as np
from tqdm import tqdm

# Adjust path to import from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.datasets import IdentityDataset
from scripts.phase2.embedding_model import EmbeddingNet
from training.losses import ArcFaceLoss

# --- Configuration ---
TRAIN_JSON_PATH = 'data/identity_train.json'
VAL_JSON_PATH = 'data/identity_val.json'
IMG_SIZE = 224
BATCH_SIZE = 32
EMBEDDING_DIM = 512
MODEL_OUTPUT_PATH = 'models/dog_embedding_best.pt'

# --- Advanced Training Hyperparameters ---
HARDWARE_WORKERS = 2 # Number of CPU workers for data loading
WARMUP_EPOCHS = 15
FULL_TRAIN_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

HEAD_LR = 1e-3       # Learning rate for the new layers during warmup
BACKBONE_LR = 1e-5   # Lower learning rate for the backbone during fine-tuning
FULL_TRAIN_LR = 1e-4 # Learning rate for the head during fine-tuning

def get_data_loaders(batch_size):
    """
    Sets up the datasets and data loaders for training and validation.
    Includes data augmentation for the training set.
    """
    # Augmentations for the training set
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    # Simple resize and normalization for the validation set
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = IdentityDataset(json_path=TRAIN_JSON_PATH, transform=train_transform)
    val_dataset = IdentityDataset(json_path=VAL_JSON_PATH, transform=val_transform)

    # Get number of classes for ArcFace
    num_classes = len(set([anno['identity_label'] for anno in train_dataset.annotations]))
    print(f"Found {num_classes} unique identities for training.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=HARDWARE_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=HARDWARE_WORKERS, pin_memory=True)

    return train_loader, val_loader, num_classes

def visualize_batch(data_loader):
    """
    Pulls a single batch from the data_loader and saves a visualization.
    """
    print("Visualizing a batch of augmented training data...")
    images, labels = next(iter(data_loader))

    grid_size = math.ceil(math.sqrt(images.size(0)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    # Inverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for i in range(images.size(0)):
        img = inv_normalize(images[i]).permute(1, 2, 0).numpy()
        # Clip to [0, 1] range before converting to uint8
        img = img.clip(0, 1)
        img = (img * 255).astype('uint8')
        
        axes[i].imshow(img)
        axes[i].set_title(f"ID: {labels[i].item()}", fontsize=8)
        axes[i].axis('off')

    for j in range(images.size(0), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("Augmented Training Batch Sample", fontsize=16, y=1.02)
    
    output_dir = 'outputs/phase2_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'augmented_training_batch.png')
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

def train_one_epoch(model, criterion_arcface, criterion_ce, optimizer, data_loader, device, epoch, total_epochs, stage_name):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{total_epochs} [{stage_name}]")

    for i, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images)
        arcface_logits = criterion_arcface(embeddings, labels)
        loss = criterion_ce(arcface_logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}'})
    
    return running_loss / len(data_loader)

def validate(model, criterion_arcface, criterion_ce, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            arcface_logits = criterion_arcface(embeddings, labels)
            loss = criterion_ce(arcface_logits, labels)
            running_loss += loss.item()
            _, predicted = torch.max(arcface_logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            val_acc = 100 * correct_predictions / total_samples
            pbar.set_postfix({'Val Acc': f'{val_acc:.2f}%'})
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples
    return avg_loss, accuracy

def main(args: argparse.Namespace):
    print("Starting advanced training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_data_loaders(BATCH_SIZE)

    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    criterion_arcface = ArcFaceLoss(in_features=EMBEDDING_DIM, out_features=num_classes).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    overall_best_val_loss = np.inf

    # --- Stage 1: Warm-up (Train Head Only) ---
    if not args.skip_warmup:
        print("\n--- Stage 1: Warming up classification head ---")
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        head_params = list(model.projection_head.parameters()) + list(criterion_arcface.parameters())
        optimizer = optim.Adam(head_params, lr=HEAD_LR)
        
        stage_best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(WARMUP_EPOCHS):
            train_loss = train_one_epoch(model, criterion_arcface, criterion_ce, optimizer, train_loader, device, epoch, WARMUP_EPOCHS, "Warm-up")
            val_loss, val_acc = validate(model, criterion_arcface, criterion_ce, val_loader, device)
            print(f"Epoch {epoch+1}/{WARMUP_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_loss < overall_best_val_loss:
                overall_best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
                print(f"New overall best val loss. Saved model to {MODEL_OUTPUT_PATH}")

            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered for warm-up after {EARLY_STOPPING_PATIENCE} epochs.")
                break

    # --- Stage 2: Full Fine-Tuning ---
    print("\n--- Stage 2: Full model fine-tuning ---")
    if not args.skip_warmup:
        print(f"Loading best model from warm-up phase (Overall Best Val Loss: {overall_best_val_loss:.4f})")
        model.load_state_dict(torch.load(MODEL_OUTPUT_PATH))
    
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': BACKBONE_LR},
        {'params': model.projection_head.parameters(), 'lr': FULL_TRAIN_LR},
        {'params': criterion_arcface.parameters(), 'lr': FULL_TRAIN_LR}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Reset early stopping counters for the fine-tuning stage
    stage_best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(FULL_TRAIN_EPOCHS):
        train_loss = train_one_epoch(model, criterion_arcface, criterion_ce, optimizer, train_loader, device, epoch, FULL_TRAIN_EPOCHS, "Fine-Tune")
        val_loss, val_acc = validate(model, criterion_arcface, criterion_ce, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{FULL_TRAIN_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_loss < overall_best_val_loss:
            overall_best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
            print(f"New overall best val loss. Saved model to {MODEL_OUTPUT_PATH}")

        if val_loss < stage_best_val_loss:
            stage_best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print("\nTraining finished.")
    print(f"Best overall validation loss: {overall_best_val_loss:.4f}")
    print(f"Best model saved at: {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dog embedding model.")
    parser.add_argument('--visualize-batch', action='store_true', help="Visualize a batch of training data to see augmentations.")
    parser.add_argument('--skip-warmup', action='store_true', help="Skip the warm-up phase and start with full fine-tuning.")
    args = parser.parse_args()

    if args.visualize_batch:
        train_loader, _, _ = get_data_loaders(batch_size=16)
        visualize_batch(train_loader)
    else:
        main(args)