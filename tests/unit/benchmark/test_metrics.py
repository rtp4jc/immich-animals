from unittest.mock import MagicMock

import numpy as np
import torch

from animal_id.benchmark.metrics import (
    calculate_tar_at_far,
    evaluate_embedding_model,
    retrieval_metrics,
)


def test_calculate_tar_at_far_no_positive_pairs():
    """Test TAR calculation when no pairs of the same identity exist."""
    # 5 unique identities, so no positive pairs can be formed
    embeddings = np.random.rand(5, 128).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 4])

    tar, threshold = calculate_tar_at_far(embeddings, labels)

    # If there are no positive pairs, TAR should be 0
    assert tar == 0.0


def test_calculate_tar_at_far_no_negative_pairs():
    """Test TAR calculation when only one identity exists."""
    # All items have the same identity, so no negative pairs can be formed
    embeddings = np.random.rand(5, 128).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0])

    tar, threshold = calculate_tar_at_far(embeddings, labels, far_threshold=0.01)

    # If there are no negative pairs to form a threshold, TAR should be 0, threshold should be 0
    assert tar == 0.0
    assert threshold == 0.0


def test_evaluate_embedding_model_empty_dataloader():
    """Test that evaluation returns an empty dict for an empty dataloader."""
    mock_model = MagicMock()
    empty_dataloader = []  # Empty list simulates an empty dataloader
    device = torch.device("cpu")

    metrics = evaluate_embedding_model(mock_model, empty_dataloader, device)

    assert metrics == {}


def test_retrieval_against_a_gallery_skips_queries_without_a_match():
    embeddings = np.eye(3)[[0, 1, 2, 0, 2]].astype(float)
    labels = np.array(["a", "b", "c", "a", "z"])
    gallery = np.array([False, False, False, True, True])
    mrr, top_k, n = retrieval_metrics(embeddings, labels, gallery=gallery)
    assert (mrr, top_k[1], n) == (1.0, 1.0, 1)


def test_leave_one_out_skips_identities_with_a_single_photo():
    embeddings = np.eye(3)[[0, 0, 1]].astype(float)
    labels = np.array(["a", "a", "b"])  # "b" has nothing to retrieve
    mrr, top_k, n = retrieval_metrics(embeddings, labels)
    assert (mrr, top_k[1], n) == (1.0, 1.0, 2)
