"""
Backbone factory for creating feature extractors for the embedding model.

Contract
--------
Every backbone returned by :func:`get_backbone` exposes a feature extractor that
emits a **flat ``(B, num_features)`` pooled vector** (global pooling lives inside
the wrapper, never in the projection head). This keeps a single contract across
torchvision CNNs, timm CNNs, and timm transformer (Swin/ViT) backbones, all of
which would otherwise emit different rank tensors (4D feature maps vs. 2D pooled
vectors vs. token sequences).

The return signature is always ``(feature_extractor, num_features)``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch.nn as nn


class BackboneType(Enum):
    # --- torchvision (ImageNet, permissive) ---
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    RESNET50 = "resnet50"

    # --- timm, ImageNet-pretrained, Apache-2.0 (shippable contenders) ---
    CONVNEXTV2_TINY = "convnextv2_tiny"
    CONVNEXTV2_NANO = "convnextv2_nano"
    EFFICIENTNETV2_RW_M = "efficientnetv2_rw_m"
    TF_EFFICIENTNETV2_M = "tf_efficientnetv2_m"

    # --- timm Swin backbones with HF-hosted animal-pretrained weights ---
    MEGADESCRIPTOR_T_224 = "megadescriptor_t_224"
    MEGADESCRIPTOR_L_384 = "megadescriptor_l_384"

    # --- bespoke HF custom model (EfficientNetV2 trunk) ---
    MIEWID_MSV3 = "miewid_msv3"


def _torchvision_efficientnet_b0(pretrained: bool):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    # model.features emits a 4D feature map; append GAP + flatten so the
    # extractor honours the flat (B, num_features) contract.
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
    num_features = model.fc.in_features  # 2048 for ResNet-50
    # Drop the final fc and the model's own avgpool, then re-append GAP +
    # flatten. This is numerically identical to the previous behaviour where
    # the AdaptiveAvgPool2d(1) + Flatten lived in the projection head — the math
    # (GAP -> flatten) is unchanged, only relocated into the extractor.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-2],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return feature_extractor, num_features


def _timm_backbone(model_name: str):
    """Build a loader for a vanilla timm model (ImageNet weights via timm).

    Uses ``num_classes=0, global_pool='avg'`` so timm returns a flat pooled
    ``(B, num_features)`` vector directly — no head pooling required.
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
    """Build a loader for a timm model whose weights live on the HF Hub.

    MegaDescriptor checkpoints (``BVRA/MegaDescriptor-*``) are plain timm Swin
    models published on the Hugging Face Hub. timm can pull both the
    architecture config and the weights via the ``hf-hub:`` prefix.

    With ``pretrained=False`` timm still needs a concrete architecture, which it
    reads from the repo's ``config.json`` — that lookup hits the network. To
    stay fully offline in unit tests, mock this loader (or use one of the plain
    timm Swin backbones as a stand-in). The factory itself performs no network
    I/O at import time.
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


def _miewid_loader(pretrained: bool):
    """Loader for MiewID (``conservationxlabs/miewid-msv3``).

    MiewID is **not** a stock timm model: the HF repo ships a custom model
    class (an EfficientNetV2-based trunk with a GeM-pooled embedding head) that
    is loaded through ``transformers.AutoModel`` with ``trust_remote_code=True``
    (the repo's remote modelling code), not through timm's ``create_model``. Its
    forward already returns a pooled embedding vector rather than a feature map,
    and its output width / GeM pooling differ from the
    ``num_classes=0, global_pool='avg'`` contract the other wrappers rely on.
    Wiring it cleanly (and offline-safe for CI) needs the remote code plus an
    adapter, which is out of scope for this pass.

    Raises:
        NotImplementedError: always, with loading notes.
    """
    raise NotImplementedError(
        "MiewID backbone is not wired yet. It is a bespoke Hugging Face custom "
        "model (repo: 'conservationxlabs/miewid-msv3', an EfficientNetV2 trunk "
        "with GeM pooling). Load it via "
        "`transformers.AutoModel.from_pretrained('conservationxlabs/miewid-msv3', "
        "trust_remote_code=True)` (or the project's `wildlife-tools` loader), "
        "then wrap its pooled-embedding output to match the "
        "(feature_extractor -> (B, num_features)) contract. The trunk emits a "
        "fixed-width embedding (not an ImageNet `global_pool='avg'` vector), so "
        "an adapter is required. Network/remote-code download must be guarded "
        "out of unit tests."
    )


@dataclass(frozen=True)
class BackboneSpec:
    """Registry entry describing how to build a backbone.

    ``loader(pretrained) -> (feature_extractor, num_features)`` where the
    feature extractor emits a flat ``(B, num_features)`` vector.
    """

    loader: Callable[[bool], tuple[nn.Module, int]]
    # Spatial input size the backbone expects (square). Most CNNs accept any
    # size; transformer backbones are fixed. Informational for callers/tests.
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
    BackboneType.MIEWID_MSV3: BackboneSpec(_miewid_loader),
}


def get_backbone(backbone_type: BackboneType, pretrained: bool = True):
    """
    Selects and instantiates a backbone feature extractor.

    Args:
        backbone_type (BackboneType): The backbone type to use.
        pretrained (bool): Whether to use pre-trained weights.

    Returns:
        A tuple containing:
        - feature_extractor: A module emitting a flat (B, num_features) vector.
        - num_features: The number of output features.
    """
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
