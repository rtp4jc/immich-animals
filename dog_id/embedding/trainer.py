"""
Generic trainer for the embedding model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

class Trainer:
    """
    A class to encapsulate the training and validation loop for the embedding model.
    """
    def __init__(self, model, arcface_loss, config):
        self.model = model
        self.arcface_loss = arcface_loss
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.arcface_loss.to(self.device)
        self.ce_loss = nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, skip_warmup=False):
        """
        Executes the full two-stage training process.
        """
        print(f"Using device: {self.device}")
        os.makedirs(os.path.dirname(self.config['MODEL_OUTPUT_PATH']), exist_ok=True)
        overall_best_val_loss = np.inf

        # --- Stage 1: Warm-up (Train Head Only) ---
        if not skip_warmup:
            overall_best_val_loss = self._warmup_phase(train_loader, val_loader)

        # --- Stage 2: Full Fine-Tuning ---
        self._finetune_phase(train_loader, val_loader, overall_best_val_loss)

        print("\nTraining finished.")
        print(f"Best overall validation loss: {overall_best_val_loss:.4f}")
        print(f"Best model saved at: {self.config['MODEL_OUTPUT_PATH']}")

    def _warmup_phase(self, train_loader, val_loader):
        print("\n--- Stage 1: Warming up classification head ---")
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        
        head_params = list(self.model.projection_head.parameters()) + list(self.arcface_loss.parameters())
        optimizer = optim.Adam(head_params, lr=self.config['HEAD_LR'])
        
        stage_best_val_loss = np.inf
        epochs_no_improve = 0
        overall_best_val_loss = np.inf

        for epoch in range(self.config['WARMUP_EPOCHS']):
            self._train_one_epoch(optimizer, train_loader, epoch, self.config['WARMUP_EPOCHS'], "Warm-up")
            val_loss, _ = self._validate(val_loader)

            if val_loss < overall_best_val_loss:
                overall_best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config['MODEL_OUTPUT_PATH'])
                print(f"New overall best val loss. Saved model to {self.config['MODEL_OUTPUT_PATH']}")

            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config['EARLY_STOPPING_PATIENCE']:
                print(f"\nEarly stopping triggered for warm-up after {self.config['EARLY_STOPPING_PATIENCE']} epochs.")
                break
        return overall_best_val_loss

    def _finetune_phase(self, train_loader, val_loader, overall_best_val_loss):
        print("\n--- Stage 2: Full model fine-tuning ---")
        if os.path.exists(self.config['MODEL_OUTPUT_PATH']):
            print(f"Loading best model from warm-up phase (Overall Best Val Loss: {overall_best_val_loss:.4f})")
            self.model.load_state_dict(torch.load(self.config['MODEL_OUTPUT_PATH']))
        
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = True

        optimizer = optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': self.config['BACKBONE_LR']},
            {'params': self.model.projection_head.parameters(), 'lr': self.config['FULL_TRAIN_LR']},
            {'params': self.arcface_loss.parameters(), 'lr': self.config['FULL_TRAIN_LR']}
        ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        stage_best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(self.config['FULL_TRAIN_EPOCHS']):
            self._train_one_epoch(optimizer, train_loader, epoch, self.config['FULL_TRAIN_EPOCHS'], "Fine-Tune")
            val_loss, _ = self._validate(val_loader)
            scheduler.step()

            if val_loss < overall_best_val_loss:
                overall_best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config['MODEL_OUTPUT_PATH'])
                print(f"New overall best val loss. Saved model to {self.config['MODEL_OUTPUT_PATH']}")

            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config['EARLY_STOPPING_PATIENCE']:
                print(f"\nEarly stopping triggered after {self.config['EARLY_STOPPING_PATIENCE']} epochs.")
                break

    def _train_one_epoch(self, optimizer, data_loader, epoch, total_epochs, stage_name):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{total_epochs} [{stage_name}]")

        for i, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            embeddings = self.model(images)
            arcface_logits = self.arcface_loss(embeddings, labels)
            loss = self.ce_loss(arcface_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}'})
    
    def _validate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        pbar = tqdm(data_loader, total=len(data_loader), desc="Validating")
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings = self.model(images)
                arcface_logits = self.arcface_loss(embeddings, labels)
                loss = self.ce_loss(arcface_logits, labels)
                running_loss += loss.item()
                _, predicted = torch.max(arcface_logits.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                val_acc = 100 * correct_predictions / total_samples
                pbar.set_postfix({'Val Acc': f'{val_acc:.2f}%'})
        avg_loss = running_loss / len(data_loader)
        accuracy = 100 * correct_predictions / total_samples
        print(f"Validation Loss: {avg_loss:.4f}, Validation Acc: {accuracy:.2f}%")
        return avg_loss, accuracy
