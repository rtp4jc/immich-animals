"""Central configuration for the embedding model pipeline."""

from dataclasses import dataclass

from animal_id.data.sample import Source

from .backbones import BackboneType
from .losses import HeadType

# Default backbone for training and inference.
DEFAULT_BACKBONE = BackboneType.DINOV2_B


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
    # Phase 1 trains the projection + margin head on a frozen trunk.
    warmup_epochs: int = 5
    full_train_epochs: int = 30
    early_stopping_patience: int = 10
    # Projection + margin head, both phases.
    head_lr: float = 1e-3
    # Peak trunk LR in phase 2; ViT blocks below the top decay by layer_decay each,
    # so the self-supervised early layers barely move.
    backbone_lr: float = 1e-5
    layer_decay: float = 0.8
    weight_decay: float = 0.05


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths and input shape."""

    train_json_path: str = "data/identity_train.json"
    val_json_path: str = "data/identity_val.json"
    test_json_path: str = "data/identity_test.json"
    sources: tuple[Source, ...] = (Source.DOGFACENET, Source.DOGREID, Source.MPDD)
    # Checkpoints are chosen on this source's val identities: owner phone photos,
    # where selecting on DogFaceNet's aligned faces picked the wrong models.
    select_on: Source = Source.DOGREID
    min_images: int = 2
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
