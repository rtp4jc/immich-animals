"""
Backbone factory for creating feature extractors for the embedding model.
"""
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights
)

def get_backbone(name: str, pretrained: bool = True):
    """
    Selects and instantiates a pre-trained backbone model.

    Args:
        name (str): The name of the backbone (e.g., 'efficientnet_b0').
        pretrained (bool): Whether to use pre-trained ImageNet weights.

    Returns:
        A tuple containing:
        - model: The instantiated backbone model's feature extractor.
        - num_features: The number of output features from the model's feature extractor.
    """
    if name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        # Return just the feature extractor part of the model
        feature_extractor = model.features
        return feature_extractor, num_features
    
    elif name == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        num_features = model.classifier[0].in_features
        # Return just the feature extractor part of the model
        feature_extractor = model.features
        return feature_extractor, num_features

    else:
        raise ValueError(f"Backbone '{name}' not recognized.")
