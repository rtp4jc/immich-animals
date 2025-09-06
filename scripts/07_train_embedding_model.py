"""
Main executable script for training the embedding model.
"""
import torch
import argparse
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Adjust path to import from our new package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dog_id.common.datasets import IdentityDataset
from dog_id.embedding.models import EmbeddingNet
from dog_id.embedding.losses import ArcFaceLoss
from dog_id.embedding.trainer import Trainer
from dog_id.embedding.config import TRAINING_CONFIG, DATA_CONFIG, DEFAULT_BACKBONE

def get_data_loaders(batch_size):
    """
    Sets up the datasets and data loaders for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((DATA_CONFIG['IMG_SIZE'], DATA_CONFIG['IMG_SIZE'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((DATA_CONFIG['IMG_SIZE'], DATA_CONFIG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = IdentityDataset(json_path=DATA_CONFIG['TRAIN_JSON_PATH'], transform=train_transform)
    val_dataset = IdentityDataset(json_path=DATA_CONFIG['VAL_JSON_PATH'], transform=val_transform)

    num_classes = len(set([anno['identity_label'] for anno in train_dataset.annotations]))
    print(f"Found {num_classes} unique identities for training.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=TRAINING_CONFIG['HARDWARE_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=TRAINING_CONFIG['HARDWARE_WORKERS'], pin_memory=True)

    return train_loader, val_loader, num_classes

def main(args):
    print("--- Embedding Model Training --- ")
    train_loader, val_loader, num_classes = get_data_loaders(DATA_CONFIG['BATCH_SIZE'])

    model = EmbeddingNet(
        backbone_name=args.backbone,
        embedding_dim=TRAINING_CONFIG['EMBEDDING_DIM']
    )

    arcface_loss = ArcFaceLoss(
        in_features=TRAINING_CONFIG['EMBEDDING_DIM'],
        out_features=num_classes
    )

    trainer = Trainer(model, arcface_loss, TRAINING_CONFIG)
    trainer.train(train_loader, val_loader, skip_warmup=args.skip_warmup)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dog embedding model.")
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, 
                        help=f"Model backbone to use ('efficientnet_b0', 'mobilenet_v3_small'). Default: {DEFAULT_BACKBONE}")
    parser.add_argument('--skip-warmup', action='store_true', 
                        help="Skip the warm-up phase and start with full fine-tuning.")
    args = parser.parse_args()
    main(args)