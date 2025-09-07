"""
Trainer class for the embedding model.
"""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

class EmbeddingTrainer:
    def __init__(self, model, train_loader, val_loader, device, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Global best model tracking (across all phases)
        self.best_val_loss = float('inf')
        
        # Per-phase early stopping tracking
        self.phase_best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.epoch_metrics = []

    def train_epoch(self, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(images, labels)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits = self.model(images, labels)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = correct / total
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
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

    def _train_and_validate_epoch(self, optimizer, epoch, total_epochs, phase, patience):
        """Helper method to train and validate one epoch with timing and logging."""
        epoch_start = time.time()
        
        train_loss = self.train_epoch(optimizer)
        val_loss, val_acc = self.validate()
        
        epoch_time = time.time() - epoch_start
        
        # Track metrics
        epoch_data = {
            'epoch': epoch + 1,
            'phase': phase,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch_time': epoch_time
        }
        self.epoch_metrics.append(epoch_data)
        
        # Check if this is the best model globally (across all phases)
        is_global_best = val_loss < self.best_val_loss
        if is_global_best:
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
        
        # Check if this is the best model in this phase (for early stopping)
        is_phase_best = val_loss < self.phase_best_val_loss
        if is_phase_best:
            self.phase_best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Always save latest checkpoint
        if not is_global_best:
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
        
        best_indicator = " [BEST]" if is_global_best else ""
        
        print(f"Epoch {(epoch % total_epochs) + 1}/{total_epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Time: {epoch_time:.1f}s{best_indicator}")
        
        # Early stopping check (based on phase performance)
        if self.patience_counter >= patience:
            print(f"Early stopping triggered in {phase} phase after {self.patience_counter} epochs without improvement")
            return True  # Signal to stop training
        
        return False  # Continue training

    def train(self, warmup_epochs, full_epochs, head_lr, backbone_lr, full_lr, patience):
        """Full training loop with warmup and fine-tuning phases."""
        print(f"Starting training in run directory: {self.run_dir}")
        
        # Phase 1: Warmup (freeze backbone)
        print(f"\n=== Phase 1: Warmup ({warmup_epochs} epochs) ===")
        self.model.freeze_backbone()
        optimizer = optim.Adam(self.model.parameters(), lr=head_lr)
        
        for epoch in range(warmup_epochs):
            should_stop = self._train_and_validate_epoch(optimizer, epoch, warmup_epochs, 'warmup', patience)
            if should_stop:
                break
        
        # Phase 2: Full training (unfreeze backbone)
        print(f"\n=== Phase 2: Full Training ({full_epochs} epochs) ===")
        
        # Load the best model from Phase 1 before starting Phase 2
        best_phase1_path = self.run_dir / "best_model.pt"
        if best_phase1_path.exists():
            print(f"Loading best Phase 1 model: {best_phase1_path}")
            self.model.load_state_dict(torch.load(best_phase1_path, map_location=self.device))
        
        self.model.unfreeze_backbone()
        
        # Reset phase tracking for new phase
        self.phase_best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Different learning rates for backbone and head
        optimizer = optim.Adam([
            {'params': self.model.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.model.head.parameters(), 'lr': full_lr}
        ])
        
        # Add learning rate scheduler for Phase 2
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        for epoch in range(full_epochs):
            should_stop = self._train_and_validate_epoch(optimizer, warmup_epochs + epoch, full_epochs, 'full_training', patience)
            scheduler.step()  # Step learning rate scheduler
            if should_stop:
                break
        
        # Save training metrics
        with open(self.run_dir / "training_metrics.json", "w") as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        return self.run_dir / "best_model.pt"
