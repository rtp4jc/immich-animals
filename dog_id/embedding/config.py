"""
Central configuration file for the embedding model pipeline.
"""

# --- Model Configuration ---
# Default backbone to use for training and inference.
# Options: 'efficientnet_b0', 'mobilenet_v3_small'
DEFAULT_BACKBONE = 'efficientnet_b0'

# --- Training Hyperparameters ---
TRAINING_CONFIG = {
    'MODEL_OUTPUT_PATH': 'models/dog_embedding_best.pt',
    'EMBEDDING_DIM': 512,
    'HARDWARE_WORKERS': 2,
    'WARMUP_EPOCHS': 25,
    'FULL_TRAIN_EPOCHS': 45,
    'EARLY_STOPPING_PATIENCE': 5,
    'HEAD_LR': 1e-3,
    'BACKBONE_LR': 1e-5,
    'FULL_TRAIN_LR': 1e-4,
}

# --- Data Configuration ---
DATA_CONFIG = {
    'TRAIN_JSON_PATH': 'data/identity_train.json',
    'VAL_JSON_PATH': 'data/identity_val.json',
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
}
