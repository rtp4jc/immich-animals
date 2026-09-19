"""Backbone factory for the embedding model.

Every backbone returned by :func:`get_backbone` emits a flat ``(B, num_features)``
pooled vector — pooling lives in the wrapper, not the projection head — so
torchvision CNNs, timm CNNs, and timm transformers share one contract. The return
signature is always ``(feature_extractor, num_features)``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch.nn as nn


class LicenseTier(Enum):
    """Whether a backbone's *weights* can ship in Immich.

    timm reports the weight license per tag (``timm.get_pretrained_cfg(tag).license``);
    these mirror it. The architecture being Apache says nothing about the weights —
    every ConvNeXt-V2 tag is CC-BY-NC even though timm itself is Apache.
    """

    PERMISSIVE = "permissive"  # Apache-2.0 / BSD — shippable.
    NONCOMMERCIAL = "noncommercial"  # CC-BY-NC — reference ceiling only.
    ENCUMBERED = "encumbered"  # Bespoke terms (DINOv3), needs a license decision.


class BackboneType(Enum):
    # torchvision
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    RESNET50 = "resnet50"

    # timm, ImageNet-pretrained, Apache weights
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"

    # timm, ImageNet-pretrained, CC-BY-NC weights
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
    # Gates shippability, not accuracy. No default: new backbones must declare it.
    license_tier: LicenseTier
    # Expected square input size; fixed for transformer backbones.
    input_size: int = 224


_BACKBONE_REGISTRY: dict[BackboneType, BackboneSpec] = {
    BackboneType.EFFICIENTNET_B0: BackboneSpec(
        _torchvision_efficientnet_b0, LicenseTier.PERMISSIVE
    ),
    BackboneType.MOBILENET_V3_SMALL: BackboneSpec(
        _torchvision_mobilenet_v3_small, LicenseTier.PERMISSIVE
    ),
    BackboneType.RESNET50: BackboneSpec(_torchvision_resnet50, LicenseTier.PERMISSIVE),
    # Weight tags are pinned: the license is a property of the tag, not the arch.
    BackboneType.CONVNEXT_TINY: BackboneSpec(
        _timm_backbone("convnext_tiny.fb_in22k_ft_in1k"), LicenseTier.PERMISSIVE
    ),
    BackboneType.CONVNEXT_SMALL: BackboneSpec(
        _timm_backbone("convnext_small.fb_in22k_ft_in1k"), LicenseTier.PERMISSIVE
    ),
    BackboneType.CONVNEXTV2_TINY: BackboneSpec(
        _timm_backbone("convnextv2_tiny"), LicenseTier.NONCOMMERCIAL
    ),
    BackboneType.CONVNEXTV2_NANO: BackboneSpec(
        _timm_backbone("convnextv2_nano"), LicenseTier.NONCOMMERCIAL
    ),
    BackboneType.EFFICIENTNETV2_RW_M: BackboneSpec(
        _timm_backbone("efficientnetv2_rw_m"), LicenseTier.PERMISSIVE
    ),
    BackboneType.TF_EFFICIENTNETV2_M: BackboneSpec(
        _timm_backbone("tf_efficientnetv2_m"), LicenseTier.PERMISSIVE
    ),
    BackboneType.MEGADESCRIPTOR_T_224: BackboneSpec(
        _timm_hf_backbone("BVRA/MegaDescriptor-T-224"),
        LicenseTier.NONCOMMERCIAL,
        input_size=224,
    ),
    BackboneType.MEGADESCRIPTOR_L_384: BackboneSpec(
        _timm_hf_backbone("BVRA/MegaDescriptor-L-384"),
        LicenseTier.NONCOMMERCIAL,
        input_size=384,
    ),
}


def get_backbone(backbone_type: BackboneType, pretrained: bool = True):
    """Instantiate a backbone, returning ``(feature_extractor, num_features)``."""
    spec = _BACKBONE_REGISTRY.get(backbone_type)
    if spec is None:
        raise ValueError(f"Backbone '{backbone_type}' not recognized.")
    return spec.loader(pretrained)


def get_backbone_license(backbone_type: BackboneType) -> LicenseTier:
    """Return the weight-license tier gating whether a backbone can ship."""
    spec = _BACKBONE_REGISTRY.get(backbone_type)
    if spec is None:
        raise ValueError(f"Backbone '{backbone_type}' not recognized.")
    return spec.license_tier


def get_backbone_input_size(backbone_type: BackboneType) -> int:
    """Return the expected square input size for a backbone (default 224)."""
    spec = _BACKBONE_REGISTRY.get(backbone_type)
    if spec is None:
        raise ValueError(f"Backbone '{backbone_type}' not recognized.")
    return spec.input_size
