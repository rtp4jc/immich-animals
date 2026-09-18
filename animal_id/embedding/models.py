"""Embedding model architecture: a backbone plus an L2-normalized projection head."""

from dataclasses import replace

import torch.nn as nn
import torch.nn.functional as F

from animal_id.embedding.backbones import BackboneType, get_backbone
from animal_id.embedding.config import HEAD_CONFIG, HeadConfig
from animal_id.embedding.losses import HeadType, build_head


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
        super().__init__()

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
        num_classes: int | None = None,
        embedding_dim: int = 512,
        pretrained: bool = True,
        head_type: HeadType | None = None,
        head_config: HeadConfig | None = None,
    ):
        """
        Args:
            num_classes: Number of identity classes. None puts the model in inference
                mode, with no margin head.
            head_type: Overrides ``head_config.head_type``.
            head_config: Defaults to ``animal_id.embedding.config.HEAD_CONFIG``.
        """
        super().__init__()

        self.backbone = EmbeddingNet(
            backbone_type, embedding_dim, pretrained=pretrained
        )

        if num_classes is not None:
            config = head_config if head_config is not None else HEAD_CONFIG
            if head_type is not None:
                config = replace(config, head_type=HeadType(head_type))
            self.head = build_head(
                config.head_type, embedding_dim, num_classes, **config.head_kwargs()
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
