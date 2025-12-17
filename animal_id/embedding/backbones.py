"""
Backbone factory for creating feature extractors for the embedding model.
"""

from enum import Enum

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet50,
)


class BackboneType(Enum):
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    RESNET50 = "resnet50"


def get_backbone(backbone_type: BackboneType, pretrained: bool = True):
    """
    Selects and instantiates a pre-trained backbone model.

    Args:
        backbone_type (BackboneType): The backbone type to use.
        pretrained (bool): Whether to use pre-trained ImageNet weights.

    Returns:
        A tuple containing:
        - model: The instantiated backbone model's feature extractor.
        - num_features: The number of output features from the model's feature extractor.
    """
    if backbone_type == BackboneType.EFFICIENTNET_B0:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        feature_extractor = model.features
        return feature_extractor, num_features

    elif backbone_type == BackboneType.MOBILENET_V3_SMALL:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        num_features = model.classifier[0].in_features
        feature_extractor = model.features
        return feature_extractor, num_features

    elif backbone_type == BackboneType.RESNET50:
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        num_features = model.fc.in_features  # 2048 for ResNet-50
        # Remove final classification layer and global average pooling
        feature_extractor = nn.Sequential(*list(model.children())[:-2])
        return feature_extractor, num_features

    else:
        raise ValueError(f"Backbone '{backbone_type}' not recognized.")
