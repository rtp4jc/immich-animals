"""Central configuration for the embedding model pipeline."""

from dataclasses import dataclass

from .backbones import BackboneType
from .losses import HeadType

# Default backbone for training and inference.
DEFAULT_BACKBONE = BackboneType.RESNET50


@dataclass(frozen=True)
class HeadConfig:
    """Train-time margin head and its hyperparameters.

    The defaults reproduce the project's historical behavior: ArcFace, s=30, m=0.50,
    label smoothing 0.1. Switching ``head_type`` to SUBCENTER_ARCFACE (the MiewID
    recipe) or COSFACE only affects training — the embedding output and the ONNX
    inference path are unchanged.
    """

    head_type: HeadType = HeadType.ARCFACE
    s: float = 30.0
    m: float = 0.50
    label_smoothing: float = 0.1
    k: int = 3  # Sub-center ArcFace: sub-centers per class.
    cosface_m: float = 0.35  # CosFace uses its own additive cosine margin.

    def head_kwargs(self) -> dict:
        """Constructor keyword arguments for the selected head."""
        kwargs = {"s": self.s, "m": self.m, "label_smoothing": self.label_smoothing}
        if self.head_type is HeadType.SUBCENTER_ARCFACE:
            kwargs["k"] = self.k
        elif self.head_type is HeadType.COSFACE:
            kwargs["m"] = self.cosface_m
        return kwargs


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    model_output_path: str = "models/dog_embedding_best.pt"
    embedding_dim: int = 512
    hardware_workers: int = 8
    warmup_epochs: int = 25
    full_train_epochs: int = 45
    early_stopping_patience: int = 10
    # ArcFace head warmup; standard Adam range for a metric-learning head.
    head_lr: float = 1e-4
    # Fine-tune the pretrained trunk; 100x smaller than the head.
    backbone_lr: float = 1e-6
    # Head in phase 2; differential LR above the backbone.
    full_train_lr: float = 1e-5


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths and input shape."""

    train_json_path: str = "data/identity_train.json"
    val_json_path: str = "data/identity_val.json"
    test_json_path: str = "data/identity_test.json"
    dogfacenet_path: str = "data/dogfacenet/DogFaceNet_224resized/after_4_bis"
    img_size: int = 224
    batch_size: int = 32


HEAD_CONFIG = HeadConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()

__all__ = [
    "DATA_CONFIG",
    "DEFAULT_BACKBONE",
    "HEAD_CONFIG",
    "TRAINING_CONFIG",
    "DataConfig",
    "HeadConfig",
    "TrainingConfig",
]
