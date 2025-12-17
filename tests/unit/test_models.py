import torch
import torch.nn as nn

from animal_id.embedding.backbones import BackboneType
from animal_id.embedding.models import DogEmbeddingModel, EmbeddingNet


def test_model_initialization():
    """Test initializing the model with a specific backbone."""
    model = DogEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        num_classes=10,
        embedding_dim=128,
        pretrained=False,
    )
    assert isinstance(model, DogEmbeddingModel)
    assert isinstance(model.backbone, EmbeddingNet)

    # Check embedding dimension logic
    # MobileNetV3 Small feature dim is 576.
    # Projection head should take 576 -> 128.
    first_linear = None
    for m in model.backbone.projection_head:
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    assert first_linear is not None
    assert first_linear.in_features == 576
    assert first_linear.out_features == 128


def test_forward_pass_training():
    """Test forward pass in training mode (returns ArcFace loss logits)."""
    model = DogEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        num_classes=5,
        embedding_dim=64,
        pretrained=False,
    )

    # Batch of 2 images, 3 channels, 224x224
    inputs = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    # Forward with labels -> ArcFace logits
    logits = model(inputs, labels)

    # Output shape should be (Batch_Size, Num_Classes)
    assert logits.shape == (2, 5)


def test_forward_pass_inference():
    """Test forward pass in inference mode (returns embeddings)."""
    model = DogEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        num_classes=5,
        embedding_dim=64,
        pretrained=False,
    )
    model.eval()

    inputs = torch.randn(2, 3, 224, 224)

    # Forward without labels -> Embeddings
    embeddings = model(inputs)

    # Output shape should be (Batch_Size, Embedding_Dim)
    assert embeddings.shape == (2, 64)

    # Embeddings should be normalized (L2 norm approx 1)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_freeze_unfreeze_backbone():
    """Test freezing and unfreezing the backbone."""
    model = DogEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL, num_classes=5, pretrained=False
    )

    model.freeze_backbone()
    for param in model.backbone.feature_extractor.parameters():
        assert param.requires_grad is False

    model.unfreeze_backbone()
    for param in model.backbone.feature_extractor.parameters():
        assert param.requires_grad is True


def test_get_embeddings_method():
    """Test the get_embeddings helper method."""
    model = DogEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        num_classes=5,
        embedding_dim=32,
        pretrained=False,
    )
    inputs = torch.randn(1, 3, 224, 224)

    embeddings = model.get_embeddings(inputs)
    assert embeddings.shape == (1, 32)
