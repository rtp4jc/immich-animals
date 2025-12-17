"""
Central configuration file for the embedding model pipeline.
"""

from .backbones import BackboneType

# --- Model Configuration ---
# Default backbone to use for training and inference.
DEFAULT_BACKBONE = BackboneType.RESNET50

# --- Training Hyperparameters ---
TRAINING_CONFIG = {
    "MODEL_OUTPUT_PATH": "models/dog_embedding_best.pt",
    "EMBEDDING_DIM": 512,
    "HARDWARE_WORKERS": 8,
    "WARMUP_EPOCHS": 25,
    "FULL_TRAIN_EPOCHS": 45,
    "EARLY_STOPPING_PATIENCE": 5,
    "HEAD_LR": 1e-8,  # Back to very conservative (Phase 1 already gets close to optimal)
    "BACKBONE_LR": 1e-9,  # Extremely small for fine-tuning
    "FULL_TRAIN_LR": 1e-9,  # Match backbone LR
}

# --- Data Configuration ---
DATA_CONFIG = {
    "TRAIN_JSON_PATH": "data/identity_train.json",
    "VAL_JSON_PATH": "data/identity_val.json",
    "DOGFACENET_PATH": "data/dogfacenet/DogFaceNet_224resized/after_4_bis",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
}
