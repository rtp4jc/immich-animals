"""Backbone factory for the embedding model.

Every backbone returned by :func:`get_backbone` emits a flat ``(B, num_features)``
pooled vector — pooling lives in the wrapper, not the projection head — so
torchvision CNNs, timm CNNs, and timm transformers share one contract. The return
signature is always ``(feature_extractor, num_features)``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch.nn as nn


class BackboneType(Enum):
    # torchvision
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    RESNET50 = "resnet50"

    # timm, ImageNet-pretrained
    CONVNEXTV2_TINY = "convnextv2_tiny"
    CONVNEXTV2_NANO = "convnextv2_nano"
    EFFICIENTNETV2_RW_M = "efficientnetv2_rw_m"
    TF_EFFICIENTNETV2_M = "tf_efficientnetv2_m"

    # timm Swin, HF-hosted animal-pretrained weights (MegaDescriptor)
    MEGADESCRIPTOR_T_224 = "megadescriptor_t_224"
    MEGADESCRIPTOR_L_384 = "megadescriptor_l_384"


def _torchvision_efficientnet_b0(pretrained: bool):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    feature_extractor = nn.Sequential(
        model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    return feature_extractor, num_features


def _torchvision_mobilenet_v3_small(pretrained: bool):
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    num_features = model.classifier[0].in_features
    feature_extractor = nn.Sequential(
        model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    return feature_extractor, num_features


def _torchvision_resnet50(pretrained: bool):
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    num_features = model.fc.in_features
    # Drop the final fc + avgpool, re-append GAP + flatten into the extractor.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-2],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return feature_extractor, num_features


def _timm_backbone(model_name: str):
    """Loader for a vanilla timm model (ImageNet weights).

    ``num_classes=0, global_pool='avg'`` makes timm return a flat pooled vector.
    """

    def loader(pretrained: bool):
        import timm

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, model.num_features

    return loader


def _timm_hf_backbone(hf_repo: str):
    """Loader for a timm model whose weights live on the HF Hub.

    MegaDescriptor checkpoints (``BVRA/MegaDescriptor-*``) are plain timm Swin
    models; timm pulls config + weights via the ``hf-hub:`` prefix. Even with
    ``pretrained=False`` the architecture lookup hits the network, so mock this
    loader in unit tests.
    """

    def loader(pretrained: bool):
        import timm

        model = timm.create_model(
            f"hf-hub:{hf_repo}",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, model.num_features

    return loader


@dataclass(frozen=True)
class BackboneSpec:
    """Registry entry describing how to build a backbone.

    ``loader(pretrained) -> (feature_extractor, num_features)`` where the
    feature extractor emits a flat ``(B, num_features)`` vector.
    """

    loader: Callable[[bool], tuple[nn.Module, int]]
    # Expected square input size; fixed for transformer backbones.
    input_size: int = 224


_BACKBONE_REGISTRY: dict[BackboneType, BackboneSpec] = {
    BackboneType.EFFICIENTNET_B0: BackboneSpec(_torchvision_efficientnet_b0),
    BackboneType.MOBILENET_V3_SMALL: BackboneSpec(_torchvision_mobilenet_v3_small),
    BackboneType.RESNET50: BackboneSpec(_torchvision_resnet50),
    BackboneType.CONVNEXTV2_TINY: BackboneSpec(_timm_backbone("convnextv2_tiny")),
    BackboneType.CONVNEXTV2_NANO: BackboneSpec(_timm_backbone("convnextv2_nano")),
    BackboneType.EFFICIENTNETV2_RW_M: BackboneSpec(
        _timm_backbone("efficientnetv2_rw_m")
    ),
    BackboneType.TF_EFFICIENTNETV2_M: BackboneSpec(
        _timm_backbone("tf_efficientnetv2_m")
    ),
    BackboneType.MEGADESCRIPTOR_T_224: BackboneSpec(
        _timm_hf_backbone("BVRA/MegaDescriptor-T-224"), input_size=224
    ),
    BackboneType.MEGADESCRIPTOR_L_384: BackboneSpec(
        _timm_hf_backbone("BVRA/MegaDescriptor-L-384"), input_size=384
    ),
}


def get_backbone(backbone_type: BackboneType, pretrained: bool = True):
    """Instantiate a backbone, returning ``(feature_extractor, num_features)``."""
    spec = _BACKBONE_REGISTRY.get(backbone_type)
    if spec is None:
        raise ValueError(f"Backbone '{backbone_type}' not recognized.")
    return spec.loader(pretrained)


def get_backbone_input_size(backbone_type: BackboneType) -> int:
    """Return the expected square input size for a backbone (default 224)."""
    spec = _BACKBONE_REGISTRY.get(backbone_type)
    if spec is None:
        raise ValueError(f"Backbone '{backbone_type}' not recognized.")
    return spec.input_size
