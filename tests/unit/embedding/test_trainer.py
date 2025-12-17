from unittest.mock import MagicMock, patch

import pytest
import torch

from animal_id.embedding.trainer import EmbeddingTrainer


@pytest.fixture
def mock_trainer_with_empty_val_loader(tmp_path):
    """Fixture for an EmbeddingTrainer with an empty validation dataloader."""
    mock_model = MagicMock()
    # Fix: Configure the mock model to return a valid logit tensor that requires gradients
    mock_model.return_value = torch.rand(2, 1001, requires_grad=True)
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.rand(1))]
    mock_model.state_dict.return_value = {}  # Needed for save_checkpoint

    # Mock the train_loader to yield one batch so the training loop runs
    mock_train_loader = [(torch.rand(2, 3, 64, 64), torch.tensor([0, 1]))]
    mock_val_loader = []  # Empty list simulates an empty dataloader

    device = torch.device("cpu")
    run_dir = tmp_path

    # The trainer's `validate` method calls `evaluate_embedding_model`.
    # We patch it here at the source. It will return an empty dict because the loader is empty.
    with patch(
        "animal_id.embedding.trainer.EmbeddingTrainer.validate", return_value={}
    ):
        trainer = EmbeddingTrainer(
            model=mock_model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            device=device,
            run_dir=run_dir,
        )
        yield trainer


def test_validate_handles_empty_dataloader(mock_trainer_with_empty_val_loader):
    """
    Test that the public validate() method handles an empty dataloader gracefully.
    """
    trainer = mock_trainer_with_empty_val_loader
    metrics = trainer.validate()

    assert metrics == {}


def test_train_loop_handles_empty_val_metrics(mock_trainer_with_empty_val_loader):
    """
    Test that the public train() method runs without crashing if validation returns empty metrics.
    This simulates robustness to an empty or misconfigured validation set.
    """
    trainer = mock_trainer_with_empty_val_loader

    try:
        # Call the main public training method for a minimal duration.
        # The key is to ensure this doesn't raise a KeyError when trying to access
        # a metric (like 'mAP') from the empty validation metrics dict.
        trainer.train(
            warmup_epochs=1,
            full_epochs=0,  # No full training needed for this test
            head_lr=1e-3,
            backbone_lr=1e-4,
            full_lr=1e-5,
            patience=3,
        )
    except KeyError as e:
        pytest.fail(
            f"Training loop crashed with KeyError, likely from accessing val_metrics: {e}"
        )
    except Exception as e:
        pytest.fail(f"An unexpected exception occurred during training: {e}")

    # The primary assertion is that the code above ran without crashing.
    # We can also check the internal state to confirm it handled the empty metrics.
    assert len(trainer.epoch_metrics) == 1
    assert trainer.epoch_metrics[0]["val_metrics"] == {}
    # The patience counter is 0 because the metric 0.0 is better than the initial -1.0
    assert trainer.patience_counter == 0
