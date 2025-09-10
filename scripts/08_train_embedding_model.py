#!/usr/bin/env python
"""
Train the embedding model with comprehensive metrics tracking.
"""
import sys
import json
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dog_id.common.datasets import DogIdentityDataset
from dog_id.embedding.models import DogEmbeddingModel
from dog_id.embedding.trainer import EmbeddingTrainer
from dog_id.embedding.config import DEFAULT_BACKBONE, TRAINING_CONFIG, DATA_CONFIG
from dog_id.embedding.backbones import BackboneType

def calculate_tar_at_far(embeddings, labels, far_threshold=0.01):
    """Calculate True Accept Rate at given False Accept Rate using same method as validation script."""
    import torch.nn.functional as F
    from itertools import combinations
    from collections import defaultdict
    
    # Convert to torch tensors if needed
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    
    labels = np.array(labels)
    
    # Generate positive pairs (same identity)
    positive_pairs = []
    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    
    for _, idxs in labels_to_indices.items():
        if len(idxs) > 1:
            positive_pairs.extend(list(combinations(idxs, 2)))
    
    # Generate negative pairs (different identities) - 2x positive pairs
    num_negative_pairs = len(positive_pairs) * 2
    negative_pairs = []
    all_indices = set(range(len(labels)))
    
    while len(negative_pairs) < num_negative_pairs:
        idx1, idx2 = np.random.choice(list(all_indices), 2, replace=False)
        if labels[idx1] != labels[idx2] and (idx1, idx2) not in negative_pairs and (idx2, idx1) not in negative_pairs:
            negative_pairs.append((idx1, idx2))
    
    # Calculate similarity scores
    pos_scores = np.array([F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item() 
                          for i, j in positive_pairs])
    neg_scores = np.array([F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item() 
                          for i, j in negative_pairs])
    
    # Calculate threshold and TAR using quantile method
    threshold = np.quantile(neg_scores, 1 - far_threshold)
    tar = np.sum(pos_scores > threshold) / len(pos_scores)
    
    return tar, threshold

def evaluate_model(model, dataloader, device):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model.get_embeddings(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    tar_1, threshold_1 = calculate_tar_at_far(all_embeddings, all_labels, 0.01)
    tar_001, threshold_001 = calculate_tar_at_far(all_embeddings, all_labels, 0.001)
    
    # Calculate mAP (simplified version)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(all_embeddings)
    
    # Calculate average precision for each query
    aps = []
    for i in range(len(all_labels)):
        # Get similarities for this query
        query_sims = similarities[i]
        # Create relevance labels (same identity = relevant)
        relevance = (all_labels == all_labels[i]).astype(int)
        relevance[i] = 0  # Exclude self
        
        if np.sum(relevance) > 0:  # Only if there are relevant items
            ap = average_precision_score(relevance, query_sims)
            aps.append(ap)
    
    map_score = np.mean(aps) if aps else 0.0
    
    return {
        'mAP': map_score,
        'TAR@FAR=1%': tar_1,
        'TAR@FAR=0.1%': tar_001,
        'threshold_1%': threshold_1,
        'threshold_0.1%': threshold_001,
        'num_identities': len(np.unique(all_labels))
    }

def create_run_directory():
    """Create a new directory for this training run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_name = DEFAULT_BACKBONE.value
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_{backbone_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_run_config(run_dir, backbone_type):
    """Save configuration for this run."""
    config = {
        'backbone': backbone_type.value,
        'training_config': TRAINING_CONFIG,
        'data_config': DATA_CONFIG,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

def main():
    # Create run directory
    run_dir = create_run_directory()
    print(f"Training run directory: {run_dir}")
    
    # Save configuration
    save_run_config(run_dir, DEFAULT_BACKBONE)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = DogIdentityDataset(
        json_path=PROJECT_ROOT / DATA_CONFIG['TRAIN_JSON_PATH'],
        img_size=DATA_CONFIG['IMG_SIZE'],
        is_training=True
    )
    
    val_dataset = DogIdentityDataset(
        json_path=PROJECT_ROOT / DATA_CONFIG['VAL_JSON_PATH'],
        img_size=DATA_CONFIG['IMG_SIZE'],
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=DATA_CONFIG['BATCH_SIZE'], 
        shuffle=True,
        num_workers=TRAINING_CONFIG['HARDWARE_WORKERS']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=DATA_CONFIG['BATCH_SIZE'], 
        shuffle=False,
        num_workers=TRAINING_CONFIG['HARDWARE_WORKERS']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of identities: {train_dataset.num_classes}")
    
    # Create model
    model = DogEmbeddingModel(
        backbone_type=DEFAULT_BACKBONE,
        num_classes=train_dataset.num_classes,
        embedding_dim=TRAINING_CONFIG['EMBEDDING_DIM']
    ).to(device)
    
    print(f"Model created with backbone: {DEFAULT_BACKBONE.value}")
    
    # Create trainer
    trainer = EmbeddingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir
    )
    
    # Train model
    print("Starting training...")
    best_model_path = trainer.train(
        warmup_epochs=TRAINING_CONFIG['WARMUP_EPOCHS'],
        full_epochs=TRAINING_CONFIG['FULL_TRAIN_EPOCHS'],
        head_lr=TRAINING_CONFIG['HEAD_LR'],
        backbone_lr=TRAINING_CONFIG['BACKBONE_LR'],
        full_lr=TRAINING_CONFIG['FULL_TRAIN_LR'],
        patience=TRAINING_CONFIG['EARLY_STOPPING_PATIENCE']
    )
    
    # Evaluate best model
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    
    # Save final metrics (convert numpy types to Python types for JSON)
    final_results = {
        'best_model_path': str(best_model_path),
        'train_metrics': {k: float(v) if hasattr(v, 'item') else v for k, v in train_metrics.items()},
        'val_metrics': {k: float(v) if hasattr(v, 'item') else v for k, v in val_metrics.items()},
        'backbone': DEFAULT_BACKBONE.value,
        'config': {
            'training': TRAINING_CONFIG,
            'data': DATA_CONFIG
        }
    }
    
    with open(run_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print results
    print(f"\n=== FINAL RESULTS ===")
    print(f"Backbone: {DEFAULT_BACKBONE.value}")
    print(f"Run Directory: {run_dir}")
    print(f"\nValidation Metrics:")
    print(f"  mAP: {val_metrics['mAP']:.4f}")
    print(f"  TAR@FAR=1%: {val_metrics['TAR@FAR=1%']:.4f}")
    print(f"  TAR@FAR=0.1%: {val_metrics['TAR@FAR=0.1%']:.4f}")
    print(f"\nTrain Metrics:")
    print(f"  mAP: {train_metrics['mAP']:.4f}")
    print(f"  TAR@FAR=1%: {train_metrics['TAR@FAR=1%']:.4f}")
    print(f"  TAR@FAR=0.1%: {train_metrics['TAR@FAR=0.1%']:.4f}")

if __name__ == "__main__":
    main()
