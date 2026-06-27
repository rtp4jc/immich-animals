"""
Defines the embedding model architecture.

What it's for:
This script defines the neural network used to generate identity embeddings from dog images.
It uses a backbone (e.g., EfficientNet) created by the backbone factory and adds a
custom head to produce the final L2-normalized embedding vector.

What it does:
1. Defines the `EmbeddingNet` class, a PyTorch `nn.Module`.
2. Uses the `get_backbone` factory to instantiate a feature extractor and get its
   output feature dimension.
3. Adds a new "projection head" that takes the features from the backbone and maps
   them to a 512-dimensional embedding space.
4. L2-normalizes the output vector, which is crucial for metric learning.
5. Includes a self-testing block to verify the model's architecture.

How to run it:
- This script is not typically run directly. It is imported by other scripts.
- To run the self-test, run from the project root:
  `python -m animal_id.embedding.models`
"""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from animal_id.embedding.backbones import BackboneType, get_backbone
from animal_id.embedding.config import HEAD_CONFIG
from animal_id.embedding.losses import HeadType, build_head


def _head_kwargs_from_config(head_type, config):
    """
    Map the project ``HEAD_CONFIG`` dict to keyword arguments for a margin head.

    Only the keys relevant to the selected ``head_type`` are forwarded. The
    shared scale/margin/label-smoothing keys map to ``s`` / ``m`` /
    ``label_smoothing``; per-head extras (sub-center ``k``, CosFace ``m``) are
    added on top. ``build_head`` drops any ``None`` values so each head falls
    back to its own defaults when a key is unset.
    """
    kwargs = {
        "s": config.get("ARCFACE_S"),
        "m": config.get("ARCFACE_M"),
        "label_smoothing": config.get("LABEL_SMOOTHING"),
    }
    if head_type == HeadType.SUBCENTER_ARCFACE:
        kwargs["k"] = config.get("SUB_CENTER_K")
    elif head_type == HeadType.COSFACE:
        # CosFace uses its own additive cosine margin rather than ARCFACE_M.
        kwargs["m"] = config.get("COSFACE_M")
    return kwargs


class EmbeddingNet(nn.Module):
    """
    A generic embedding network that uses a backbone from the factory.
    """

    def __init__(
        self,
        backbone_type: BackboneType,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout_prob: float = 0.5,
    ):
        """
        Args:
            backbone_type (BackboneType): Type of backbone to use.
            embedding_dim (int): The dimensionality of the output embedding vector.
            pretrained (bool): Whether to use weights pre-trained on ImageNet for the backbone.
            dropout_prob (float): Probability for the dropout layer.
        """
        super(EmbeddingNet, self).__init__()

        self.feature_extractor, num_features = get_backbone(backbone_type, pretrained)

        # Global pooling + flatten now live inside each backbone wrapper
        # (the feature extractor emits a flat (B, num_features) vector), so the
        # projection head is just Dropout + Linear. For the torchvision CNNs
        # this is numerically identical to the previous GAP->Flatten->Dropout->
        # Linear head — the pooling was simply relocated into the extractor.
        self.projection_head = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_features, embedding_dim),
        )

        # Initialize the weights of the new projection head
        self.projection_head[-1].apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass to extract embeddings.
        """
        features = self.feature_extractor(x)
        embeddings = self.projection_head(features)

        # L2 normalize the embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings


class AnimalEmbeddingModel(nn.Module):
    """
    Complete animal embedding model with ArcFace loss.
    """

    def __init__(
        self,
        backbone_type: BackboneType,
        num_classes: Optional[int] = None,
        embedding_dim: int = 512,
        pretrained: bool = True,
        head_type: Optional[HeadType] = None,
        head_config: Optional[dict] = None,
    ):
        """
        Args:
            backbone_type (BackboneType): The backbone architecture.
            num_classes (Optional[int]): Number of identity classes.
                                         If None, model is in inference mode (no classification head).
            embedding_dim (int): Size of embedding vector.
            pretrained (bool): Use ImageNet weights.
            head_type (Optional[HeadType]): Margin head to use. Defaults to
                ``head_config["HEAD_TYPE"]`` (ArcFace via the project config),
                preserving historical behavior.
            head_config (Optional[dict]): Head-hyperparameter dict. Defaults to
                ``animal_id.embedding.config.HEAD_CONFIG``.
        """
        super(AnimalEmbeddingModel, self).__init__()

        self.backbone = EmbeddingNet(
            backbone_type, embedding_dim, pretrained=pretrained
        )

        if num_classes is not None:
            head_config = head_config if head_config is not None else HEAD_CONFIG
            head_type = head_type if head_type is not None else head_config["HEAD_TYPE"]
            head_type = HeadType(head_type)
            self.head = build_head(
                head_type,
                embedding_dim,
                num_classes,
                **_head_kwargs_from_config(head_type, head_config),
            )
        else:
            self.head = None

    def forward(self, x, labels=None):
        """Forward pass for training."""
        embeddings = self.backbone(x)
        if self.head is not None and labels is not None:
            # Training mode - return logits for loss calculation
            return self.head(embeddings, labels)

        # Inference mode - return embeddings
        return embeddings

    def get_embeddings(self, x):
        """Get embeddings without ArcFace head."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_feature_extractor(self):
        """Freeze the backbone trunk, leaving the projection head trainable."""
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        """Unfreeze the backbone trunk."""
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = True
