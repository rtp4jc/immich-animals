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
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_OUTPUT_PATH = 'models/dog_embedding_best.pt'

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

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

def train_one_epoch(model, criterion_arcface, criterion_ce, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [T]")

    for i, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(images)
        arcface_logits = criterion_arcface(embeddings, labels)
        loss = criterion_ce(arcface_logits, labels)

        # Backward pass and optimize
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

            # Calculate accuracy
            _, predicted = torch.max(arcface_logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            val_acc = 100 * correct_predictions / total_samples
            pbar.set_postfix({'Val Acc': f'{val_acc:.2f}%'})

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples
    return avg_loss, accuracy

def main():
    print("Starting training process...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, num_classes = get_data_loaders(BATCH_SIZE)

    # Initialize model, loss, and optimizer
    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    criterion_arcface = ArcFaceLoss(in_features=EMBEDDING_DIM, out_features=num_classes).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    
    params = list(model.parameters()) + list(criterion_arcface.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    print("Setup complete. Starting training loop.")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, criterion_arcface, criterion_ce, optimizer, train_loader, device, epoch)
        val_loss, val_acc = validate(model, criterion_arcface, criterion_ce, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
            print(f"New best model saved to {MODEL_OUTPUT_PATH} with accuracy: {val_acc:.2f}%")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {MODEL_OUTPUT_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dog embedding model.")
    parser.add_argument('--visualize-batch', action='store_true',
                        help="Visualize a batch of training data to see augmentations.")
    args = parser.parse_args()

    if args.visualize_batch:
        train_loader, _, _ = get_data_loaders(batch_size=16)
        visualize_batch(train_loader)
    else:
        main()
