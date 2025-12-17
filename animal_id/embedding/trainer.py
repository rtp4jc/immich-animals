"""
Trainer class for the embedding model.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from animal_id.benchmark.metrics import evaluate_embedding_model


class EmbeddingTrainer:
    def __init__(self, model, train_loader, val_loader, device, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)

        # Label smoothing helps prevent overfitting on specific identities
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Global best model tracking
        # We track mAP (Mean Average Precision) because this is an Open-Set problem.
        # We cannot use classification accuracy for validation because validation
        # identities are not in the training set (Open-Set protocol).
        self.best_val_metric = -1.0

        # Per-phase early stopping tracking
        self.phase_best_val_metric = -1.0
        self.patience_counter = 0

        # Metrics tracking
        self.epoch_metrics = []

    def train_epoch(self, optimizer):
        """Train for one epoch using ArcFace loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            # Forward pass through backbone AND ArcFace head
            logits = self.model(images, labels)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def validate(self):
        """
        Validate the model using Embedding Metrics (mAP, TAR).

        CRITICAL: We cannot use CrossEntropyLoss for validation because the validation
        set contains identities (classes) that the model has never seen (Open-Set).
        The ArcFace head only knows about training identities.

        Instead, we generate embeddings for all validation images and measure how well
        they cluster by identity using cosine similarity.
        """
        metrics = evaluate_embedding_model(self.model, self.val_loader, self.device)
        return metrics

    def save_checkpoint(self, epoch, val_metric, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_mAP": val_metric,
        }

        # Save latest checkpoint
        checkpoint_path = self.run_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.run_dir / "best_model.pt"
            torch.save(self.model.state_dict(), best_path)
            return best_path

        return None

    def _train_and_validate_epoch(
        self, optimizer, epoch, total_epochs, phase, patience
    ):
        """Helper method to train and validate one epoch with timing and logging."""
        epoch_start = time.time()

        train_loss = self.train_epoch(optimizer)

        # Validation Step
        val_metrics = self.validate()

        # Use mAP as the primary metric for model selection
        current_metric = val_metrics.get("mAP", 0.0)

        epoch_time = time.time() - epoch_start

        # Track metrics
        epoch_data = {
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "epoch_time": epoch_time,
        }
        self.epoch_metrics.append(epoch_data)

        # Check if this is the best model globally (across all phases)
        is_global_best = current_metric > self.best_val_metric
        if is_global_best:
            self.best_val_metric = current_metric
            self.save_checkpoint(epoch, current_metric, is_best=True)

        # Check if this is the best model in this phase (for early stopping)
        is_phase_best = current_metric > self.phase_best_val_metric
        if is_phase_best:
            self.phase_best_val_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Always save latest checkpoint
        if not is_global_best:
            self.save_checkpoint(epoch, current_metric, is_best=False)

        best_indicator = " [BEST]" if is_global_best else ""

        print(
            f"Epoch {(epoch % total_epochs) + 1}/{total_epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val mAP: {current_metric:.4f}, "
            f"TAR@1%: {val_metrics.get('TAR@FAR=1%', 0.0):.4f}, "
            f"Time: {epoch_time:.1f}s{best_indicator}"
        )

        # Early stopping check (based on phase performance)
        if self.patience_counter >= patience:
            print(
                f"Early stopping triggered in {phase} phase after {self.patience_counter} epochs without improvement"
            )
            return True  # Signal to stop training

        return False  # Continue training

    def train(
        self, warmup_epochs, full_epochs, head_lr, backbone_lr, full_lr, patience
    ):
        """Full training loop with warmup and fine-tuning phases."""
        print(f"Starting training in run directory: {self.run_dir}")

        # Phase 1: Warmup (freeze backbone)
        print(f"\n=== Phase 1: Warmup ({warmup_epochs} epochs) ===")
        self.model.freeze_backbone()
        optimizer = optim.Adam(self.model.parameters(), lr=head_lr)

        for epoch in range(warmup_epochs):
            should_stop = self._train_and_validate_epoch(
                optimizer, epoch, warmup_epochs, "warmup", patience
            )
            if should_stop:
                break

        # Phase 2: Full training (unfreeze backbone)
        print(f"\n=== Phase 2: Full Training ({full_epochs} epochs) ===")

        # Load the best model from Phase 1 before starting Phase 2
        best_phase1_path = self.run_dir / "best_model.pt"
        if best_phase1_path.exists():
            print(f"Loading best Phase 1 model: {best_phase1_path}")
            self.model.load_state_dict(
                torch.load(best_phase1_path, map_location=self.device)
            )

        self.model.unfreeze_backbone()

        # Reset phase tracking for new phase
        self.phase_best_val_metric = -1.0
        self.patience_counter = 0

        # Different learning rates for backbone and head
        optimizer = optim.Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": backbone_lr},
                {"params": self.model.head.parameters(), "lr": full_lr},
            ]
        )

        # Sequential scheduler: Linear warmup → Cosine annealing
        warmup_epochs_phase2 = 20
        linear_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs_phase2
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=full_epochs - warmup_epochs_phase2
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_epochs_phase2],
        )

        for epoch in range(full_epochs):
            should_stop = self._train_and_validate_epoch(
                optimizer, warmup_epochs + epoch, full_epochs, "full_training", patience
            )
            scheduler.step()
            if should_stop:
                break

        # Save training metrics
        with open(self.run_dir / "training_metrics.json", "w") as f:
            # Helper to convert numpy/tensor values to float
            def convert(o):
                if hasattr(o, "item"):
                    return o.item()
                if isinstance(o, dict):
                    return {k: convert(v) for k, v in o.items()}
                return o

            json.dump([convert(m) for m in self.epoch_metrics], f, indent=2)

        print(f"Training completed. Best validation mAP: {self.best_val_metric:.4f}")
        return self.run_dir / "best_model.pt"
