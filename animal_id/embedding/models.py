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

import torch
import torch.nn as nn
import torch.nn.functional as F

from animal_id.embedding.backbones import get_backbone, BackboneType
from animal_id.embedding.losses import ArcFaceLoss


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

        # Define the new projection head with Dropout
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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


class DogEmbeddingModel(nn.Module):
    """
    Complete dog embedding model with ArcFace loss.
    """

    def __init__(
        self,
        backbone_type: BackboneType,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super(DogEmbeddingModel, self).__init__()

        self.backbone = EmbeddingNet(backbone_type, embedding_dim, pretrained=pretrained)
        self.head = ArcFaceLoss(embedding_dim, num_classes)

    def forward(self, x, labels=None):
        """Forward pass for training."""
        embeddings = self.backbone(x)
        if labels is not None:
            # Training mode - return logits for loss calculation
            return self.head(embeddings, labels)
        else:
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
