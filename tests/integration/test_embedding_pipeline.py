import torch
from torch.utils.data import DataLoader

from animal_id.common.datasets import DogIdentityDataset
from animal_id.embedding.backbones import BackboneType
from animal_id.embedding.models import DogEmbeddingModel
from animal_id.embedding.trainer import EmbeddingTrainer


def test_train_embedding(mock_image_dataset, tmp_path):
    """
    Integration test for the embedding training pipeline.
    Runs a single warmup epoch to verify end-to-end execution.
    """
    # 1. Setup Data
    dataset = DogIdentityDataset(
        json_path=mock_image_dataset, img_size=64, is_training=True
    )
    # Create a small loader (batch size 2 to work with 5 items)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Setup Model
    # Use MobileNetV3 for speed, pretrained=False to avoid network calls
    model = DogEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        num_classes=dataset.num_classes,
        embedding_dim=128,  # Small dim for test
        pretrained=False,
    )

    # 3. Setup Trainer
    run_dir = tmp_path / "runs" / "embedding" / "integration_test"
    run_dir.mkdir(parents=True)

    device = torch.device("cpu")
    model.to(device)

    trainer = EmbeddingTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,  # Use same loader for val to save time
        device=device,
        run_dir=run_dir,
    )

    # 4. Run Training
    best_model_path = trainer.train(
        warmup_epochs=1,
        full_epochs=1,
        head_lr=0.001,
        backbone_lr=0.0001,
        full_lr=0.001,
        patience=1,
    )

    # 5. Verify Results
    assert best_model_path is not None
    assert best_model_path.exists()
    assert (run_dir / "training_metrics.json").exists()

    print(f"Embedding integration test passed. Model saved to {best_model_path}")
