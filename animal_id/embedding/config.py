"""
Central configuration file for the embedding model pipeline.
"""

from .backbones import BackboneType
from .losses import HeadType

# --- Model Configuration ---
# Default backbone to use for training and inference.
DEFAULT_BACKBONE = BackboneType.RESNET50

# --- Margin Head Configuration ---
# Selects and parameterizes the train-time margin head. The default reproduces
# the historical hardcoded behavior exactly: ArcFace with s=30, m=0.50, label
# smoothing 0.1. Swapping HEAD_TYPE to SUBCENTER_ARCFACE (the MiewID recipe) or
# COSFACE is a one-line change and only affects training; the embedding output
# and ONNX inference path are unchanged.
HEAD_CONFIG = {
    "HEAD_TYPE": HeadType.ARCFACE,
    # Shared margin-head hyperparameters (None => use the head's own default).
    "ARCFACE_S": 30.0,
    "ARCFACE_M": 0.50,
    "LABEL_SMOOTHING": 0.1,
    # Sub-center ArcFace: number of sub-centers per class.
    "SUB_CENTER_K": 3,
    # CosFace additive cosine margin (only used when HEAD_TYPE == COSFACE).
    "COSFACE_M": 0.35,
}

# --- Training Hyperparameters ---
TRAINING_CONFIG = {
    "MODEL_OUTPUT_PATH": "models/dog_embedding_best.pt",
    "EMBEDDING_DIM": 512,
    "HARDWARE_WORKERS": 8,
    "WARMUP_EPOCHS": 25,
    "FULL_TRAIN_EPOCHS": 45,
    "EARLY_STOPPING_PATIENCE": 10,
    "HEAD_LR": 1e-4,  # ArcFace head warmup; standard Adam range for metric-learning head
    "BACKBONE_LR": 1e-6,  # Fine-tune pretrained ResNet50; 100x smaller than head
    "FULL_TRAIN_LR": 1e-5,  # Head in phase 2; differential LR above backbone
}

# --- Data Configuration ---
DATA_CONFIG = {
    "TRAIN_JSON_PATH": "data/identity_train.json",
    "VAL_JSON_PATH": "data/identity_val.json",
    "TEST_JSON_PATH": "data/identity_test.json",
    "DOGFACENET_PATH": "data/dogfacenet/DogFaceNet_224resized/after_4_bis",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
}
