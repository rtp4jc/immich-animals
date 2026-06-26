"""Tests for the backbone registry and the unified flat-feature contract.

All tests use ``pretrained=False`` (random init) or mock the loader so that no
weight download happens in CI.
"""

import pytest
import torch

from animal_id.embedding.backbones import (
    BackboneType,
    get_backbone,
    get_backbone_input_size,
)
from animal_id.embedding.models import EmbeddingNet

# (BackboneType, expected num_features, input_size)
TIMM_CNN_BACKBONES = [
    (BackboneType.CONVNEXTV2_TINY, 768, 224),
    (BackboneType.CONVNEXTV2_NANO, 640, 224),
    (BackboneType.EFFICIENTNETV2_RW_M, 2152, 224),
    (BackboneType.TF_EFFICIENTNETV2_M, 1280, 224),
]

# 224px: at tiny inputs (e.g. 64px) randomly-initialised EfficientNet/MobileNet
# features collapse to ~0, so F.normalize hits its eps floor and the unit-norm
# assertion fails for reasons unrelated to the contract. 224 is cheap enough.
TORCHVISION_BACKBONES = [
    (BackboneType.EFFICIENTNET_B0, 1280, 224),
    (BackboneType.MOBILENET_V3_SMALL, 576, 224),
    (BackboneType.RESNET50, 2048, 224),
]


def _assert_flat_features(feature_extractor, num_features, input_size):
    """Feature extractor must emit a flat (B, num_features) vector."""
    x = torch.randn(2, 3, input_size, input_size)
    feature_extractor.eval()
    with torch.no_grad():
        feats = feature_extractor(x)
    assert feats.shape == (2, num_features), (
        f"expected flat (2, {num_features}), got {tuple(feats.shape)}"
    )


def _assert_embedding_contract(backbone_type, input_size, embedding_dim=512):
    """A forward through EmbeddingNet yields a (B, embedding_dim) L2-normed vec.

    Run in ``train()`` mode so BatchNorm uses batch statistics. In eval mode a
    randomly-initialised deep BN stack (e.g. EfficientNet) drives features below
    F.normalize's eps floor, making the unit-norm assertion fail for reasons
    unrelated to the contract. With real (pretrained) weights this never
    happens; batch-stat BN keeps the signal healthy for random init too.
    """
    torch.manual_seed(0)
    model = EmbeddingNet(
        backbone_type=backbone_type,
        embedding_dim=embedding_dim,
        pretrained=False,
    )
    model.train()
    # Batch of 8 so BatchNorm batch statistics are well-conditioned.
    x = torch.randn(8, 3, input_size, input_size)
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (8, embedding_dim)
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize("backbone_type,num_features,input_size", TORCHVISION_BACKBONES)
def test_torchvision_backbones_flat_contract(backbone_type, num_features, input_size):
    fe, nf = get_backbone(backbone_type, pretrained=False)
    assert nf == num_features
    _assert_flat_features(fe, nf, input_size)


@pytest.mark.parametrize("backbone_type,num_features,input_size", TORCHVISION_BACKBONES)
def test_torchvision_embedding_contract(backbone_type, num_features, input_size):
    _assert_embedding_contract(backbone_type, input_size)


@pytest.mark.parametrize("backbone_type,num_features,input_size", TIMM_CNN_BACKBONES)
def test_timm_cnn_backbones_flat_contract(backbone_type, num_features, input_size):
    fe, nf = get_backbone(backbone_type, pretrained=False)
    assert nf == num_features
    _assert_flat_features(fe, nf, input_size)


@pytest.mark.parametrize("backbone_type,num_features,input_size", TIMM_CNN_BACKBONES)
def test_timm_cnn_embedding_contract(backbone_type, num_features, input_size):
    _assert_embedding_contract(backbone_type, input_size)


@pytest.mark.parametrize(
    "backbone_type,timm_name,num_features,input_size",
    [
        (BackboneType.MEGADESCRIPTOR_T_224, "swin_tiny_patch4_window7_224", 768, 224),
        (
            BackboneType.MEGADESCRIPTOR_L_384,
            "swin_large_patch4_window12_384",
            1536,
            384,
        ),
    ],
)
def test_megadescriptor_swin_contract(
    monkeypatch, backbone_type, timm_name, num_features, input_size
):
    """MegaDescriptor loads via timm ``hf-hub:``; offline we mock create_model
    to build the equivalent plain timm Swin with random weights."""
    import timm

    real_create_model = timm.create_model

    def fake_create_model(model_name, pretrained=False, **kwargs):
        # MegaDescriptor uses the ``hf-hub:BVRA/...`` syntax; substitute the
        # equivalent local Swin architecture and force pretrained=False (offline).
        assert model_name.startswith("hf-hub:BVRA/MegaDescriptor")
        return real_create_model(timm_name, pretrained=False, **kwargs)

    monkeypatch.setattr(timm, "create_model", fake_create_model)

    fe, nf = get_backbone(backbone_type, pretrained=False)
    assert nf == num_features
    assert get_backbone_input_size(backbone_type) == input_size
    _assert_flat_features(fe, nf, input_size)

    # Full EmbeddingNet contract.
    model = EmbeddingNet(backbone_type=backbone_type, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, input_size, input_size)
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (2, 512)
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_miewid_raises_helpful_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        get_backbone(BackboneType.MIEWID_MSV3, pretrained=False)
    msg = str(exc.value)
    assert "conservationxlabs/miewid-msv3" in msg
    assert "trust_remote_code" in msg


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        get_backbone("not-a-backbone")  # type: ignore[arg-type]
