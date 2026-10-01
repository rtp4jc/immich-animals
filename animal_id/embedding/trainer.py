"""
Trainer class for the embedding model.
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from animal_id.benchmark.metrics import evaluate_embedding_model
from animal_id.embedding.config import HEAD_CONFIG, TRAINING_CONFIG

logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    def __init__(self, model, train_loader, val_loader, device, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)

        # Read it off the head so a configured value cannot silently do nothing.
        smoothing = getattr(getattr(model, "head", None), "label_smoothing", None)
        if not isinstance(smoothing, (int, float)):
            smoothing = HEAD_CONFIG.label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(smoothing))

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

    def _autocast(self):
        # bf16 halves ViT-B's step time; the margin head stays fp32 (below).
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        )

    def train_epoch(self, optimizer, scheduler=None):
        """Train for one epoch with the margin head."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            with self._autocast():
                embeddings = self.model.get_embeddings(images)
            logits = self.model.head(embeddings.float(), labels)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
        The margin head only knows about training identities.

        Instead, we generate embeddings for all validation images and measure how well
        they cluster by identity using cosine similarity.
        """
        with self._autocast():
            return evaluate_embedding_model(self.model, self.val_loader, self.device)

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
        self, optimizer, epoch, total_epochs, phase, patience, scheduler=None
    ):
        """Helper method to train and validate one epoch with timing and logging."""
        epoch_start = time.time()

        train_loss = self.train_epoch(optimizer, scheduler)

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

        logger.info(
            f"Epoch {(epoch % total_epochs) + 1}/{total_epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val mAP: {current_metric:.4f}, "
            f"TAR@1%: {val_metrics.get('TAR@FAR=1%', 0.0):.4f}, "
            f"Time: {epoch_time:.1f}s{best_indicator}"
        )

        # Early stopping check (based on phase performance)
        if self.patience_counter >= patience:
            logger.info(
                f"Early stopping triggered in {phase} phase after {self.patience_counter} epochs without improvement"
            )
            return True  # Signal to stop training

        return False  # Continue training

    @staticmethod
    def _convert_metric(value):
        """Recursively convert numpy/tensor values to plain floats for JSON."""
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, dict):
            return {
                key: EmbeddingTrainer._convert_metric(inner)
                for key, inner in value.items()
            }
        return value

    def _head_params(self):
        return list(self.model.backbone.projection_head.parameters()) + list(
            self.model.head.parameters()
        )

    def _trunk_groups(self, lr):
        """Trunk param groups; a ViT's blocks decay by ``layer_decay`` from the top down."""
        trunk = self.model.backbone.feature_extractor
        blocks = getattr(trunk, "blocks", None)
        if blocks is None:
            return [{"params": trunk.parameters(), "lr": lr}]
        decay, n = TRAINING_CONFIG.layer_decay, len(blocks)
        groups = [
            {"params": block.parameters(), "lr": lr * decay ** (n - 1 - i)}
            for i, block in enumerate(blocks)
        ]
        rest = [
            (name, p)
            for name, p in trunk.named_parameters()
            if not name.startswith("blocks.")
        ]
        # Final norm sits above the blocks; patch/position embeddings below them.
        groups.append(
            {"params": [p for name, p in rest if name.startswith("norm")], "lr": lr}
        )
        groups.append(
            {
                "params": [p for name, p in rest if not name.startswith("norm")],
                "lr": lr * decay**n,
            }
        )
        return groups

    def train(
        self,
        warmup_epochs,
        full_epochs,
        head_lr,
        backbone_lr,
        patience,
        linear_probe=False,
    ):
        """Head warmup on a frozen trunk, then the whole model under one-cycle AdamW.

        When ``linear_probe`` is True, the trunk stays frozen and only the
        projection + margin head train (``warmup_epochs`` epochs, no fine-tune
        phase) — a per-backbone-LR-free feature-quality probe.
        """
        logger.info(f"Starting training in run directory: {self.run_dir}")
        weight_decay = TRAINING_CONFIG.weight_decay

        phase = "linear_probe" if linear_probe else "warmup"
        logger.info(f"\n=== {phase} ({warmup_epochs} epochs, frozen trunk) ===")
        self.model.freeze_feature_extractor()
        optimizer = optim.AdamW(
            self._head_params(), lr=head_lr, weight_decay=weight_decay
        )
        for epoch in range(warmup_epochs):
            if self._train_and_validate_epoch(
                optimizer, epoch, warmup_epochs, phase, patience
            ):
                break

        if not linear_probe and full_epochs:
            logger.info(f"\n=== Full training ({full_epochs} epochs) ===")
            self.model.load_state_dict(
                torch.load(self.run_dir / "best_model.pt", map_location=self.device)
            )
            self.model.unfreeze_feature_extractor()
            self.phase_best_val_metric = -1.0
            self.patience_counter = 0

            groups = [{"params": self._head_params(), "lr": head_lr}]
            groups += self._trunk_groups(backbone_lr)
            optimizer = optim.AdamW(groups, weight_decay=weight_decay)
            # Per-step warmup (10%) then cosine decay, each group to its own peak.
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g["lr"] for g in groups],
                total_steps=full_epochs * len(self.train_loader),
                pct_start=0.1,
            )
            for epoch in range(full_epochs):
                if self._train_and_validate_epoch(
                    optimizer,
                    warmup_epochs + epoch,
                    full_epochs,
                    "full_training",
                    patience,
                    scheduler,
                ):
                    break

        with open(self.run_dir / "training_metrics.json", "w") as f:
            json.dump(
                [self._convert_metric(m) for m in self.epoch_metrics], f, indent=2
            )
        logger.info(
            f"Training completed. Best validation mAP: {self.best_val_metric:.4f}"
        )
        return self.run_dir / "best_model.pt"
