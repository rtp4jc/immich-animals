import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from animal_id.embedding.losses import (
    HEAD_TYPES,
    ArcFaceLoss,
    CosFaceLoss,
    HeadType,
    MarginHead,
    SubCenterArcFace,
    build_head,
)

EMBEDDING_DIM = 32
NUM_CLASSES = 6
BATCH_SIZE = 4


def _dummy_inputs():
    embeddings = F.normalize(torch.randn(BATCH_SIZE, EMBEDDING_DIM), p=2, dim=1)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return embeddings, labels


@pytest.mark.parametrize("head_type", list(HeadType))
def test_head_forward_shape_and_backward(head_type):
    """Each head maps (batch, dim) embeddings + int labels to (batch, classes) logits
    and supports a backward pass."""
    embeddings, labels = _dummy_inputs()
    embeddings.requires_grad_(True)

    head = build_head(head_type, EMBEDDING_DIM, NUM_CLASSES)
    assert isinstance(head, MarginHead)

    logits = head(embeddings, labels)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    loss = nn.CrossEntropyLoss()(logits, labels)
    assert loss.item() > 0

    loss.backward()
    # Gradients should flow to both the head weights and the embeddings.
    assert head.weight.grad is not None
    assert embeddings.grad is not None


@pytest.mark.parametrize(
    "head_type,expected_cls",
    [
        (HeadType.ARCFACE, ArcFaceLoss),
        (HeadType.SUBCENTER_ARCFACE, SubCenterArcFace),
        (HeadType.COSFACE, CosFaceLoss),
    ],
)
def test_build_head_dispatch(head_type, expected_cls):
    """The factory dispatches each HEAD_TYPE to the right class."""
    head = build_head(head_type, EMBEDDING_DIM, NUM_CLASSES)
    assert type(head) is expected_cls
    assert HEAD_TYPES[head_type] is expected_cls


def test_build_head_accepts_string_value():
    """The factory accepts the enum's string value as well as the enum."""
    head = build_head("arcface", EMBEDDING_DIM, NUM_CLASSES)
    assert isinstance(head, ArcFaceLoss)


def test_build_head_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_head("not_a_real_head", EMBEDDING_DIM, NUM_CLASSES)


def test_build_head_drops_none_kwargs():
    """None-valued kwargs fall back to the head's own defaults."""
    head = build_head("cosface", EMBEDDING_DIM, NUM_CLASSES, m=None)
    # CosFace default margin is 0.35.
    assert head.m == pytest.approx(0.35)


def test_subcenter_weight_shape_and_k():
    """Sub-center ArcFace allocates k prototypes per class."""
    k = 3
    head = build_head(HeadType.SUBCENTER_ARCFACE, EMBEDDING_DIM, NUM_CLASSES, k=k)
    assert head.k == k
    assert head.num_centers == k
    assert head.weight.shape == (NUM_CLASSES * k, EMBEDDING_DIM)


def test_arcface_default_numerically_unchanged():
    """The factory-built ArcFace with default config must reproduce the output of
    a direct ArcFaceLoss instantiation (behavior-preserving refactor)."""
    torch.manual_seed(0)
    direct = ArcFaceLoss(EMBEDDING_DIM, NUM_CLASSES)

    torch.manual_seed(0)
    via_factory = build_head(HeadType.ARCFACE, EMBEDDING_DIM, NUM_CLASSES)

    # Same default hyperparameters.
    assert direct.s == via_factory.s == 30.0
    assert direct.m == via_factory.m == 0.50
    assert direct.label_smoothing == via_factory.label_smoothing == 0.1

    # Identical xavier-init weights given the same seed.
    assert torch.allclose(direct.weight, via_factory.weight)

    embeddings, labels = _dummy_inputs()
    assert torch.allclose(direct(embeddings, labels), via_factory(embeddings, labels))


def test_subcenter_reduces_to_arcface_when_k_is_one():
    """With k=1 sub-center ArcFace and ArcFace produce identical logits given the
    same prototype weights."""
    sub = SubCenterArcFace(EMBEDDING_DIM, NUM_CLASSES, k=1)
    arc = ArcFaceLoss(EMBEDDING_DIM, NUM_CLASSES)
    with torch.no_grad():
        arc.weight.copy_(sub.weight)

    embeddings, labels = _dummy_inputs()
    assert torch.allclose(sub(embeddings, labels), arc(embeddings, labels))


def test_cosface_subtracts_cosine_margin():
    """CosFace logits equal s*(cosine - m) at target positions and s*cosine
    elsewhere."""
    head = CosFaceLoss(EMBEDDING_DIM, NUM_CLASSES, s=30.0, m=0.35)
    embeddings, labels = _dummy_inputs()

    cosine = head.cosine_logits(embeddings)
    logits = head(embeddings, labels)

    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    expected = (cosine - one_hot * head.m) * head.s
    assert torch.allclose(logits, expected)


def test_arcface_th_mm_constants():
    """Sanity-check the angular-margin numerical-stability constants are set."""
    head = ArcFaceLoss(EMBEDDING_DIM, NUM_CLASSES, m=0.5)
    assert head.cos_m == pytest.approx(math.cos(0.5))
    assert head.sin_m == pytest.approx(math.sin(0.5))
