from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def _generate_positive_pairs(
    labels: np.ndarray, labels_to_indices: Dict[Any, List[int]]
) -> List[Tuple[int, int]]:
    """Generate all possible pairs of indices belonging to the same identity."""
    positive_pairs = []
    for _, idxs in labels_to_indices.items():
        if len(idxs) > 1:
            positive_pairs.extend(list(combinations(idxs, 2)))
    return positive_pairs


def _generate_negative_pairs(
    labels: np.ndarray, labels_to_indices: Dict[Any, List[int]], num_target_pairs: int
) -> List[Tuple[int, int]]:
    """
    Generate pairs of indices belonging to different identities.
    Uses exact generation for small datasets and efficient sampling for large ones.
    """
    unique_labels = list(labels_to_indices.keys())
    if len(unique_labels) < 2:
        return []

    # Calculate limits to avoid impossible requests
    n_samples = len(labels)
    # total_possible_pairs = n_samples * (n_samples - 1) // 2

    # Calculate positive pairs count roughly to get max negatives (exact count not strictly needed for cap)
    # But for safety, we can just check against total pairs.
    # Better: calculate exact max negatives if needed, but for now we trust the target is reasonable
    # or we handle the generated count.

    # We'll use the logic from before:
    negative_pairs = set()

    # STRATEGY 1: Small Dataset - Exact Generation
    if n_samples < 500:
        all_possible_negatives = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] != labels[j]:
                    all_possible_negatives.append((i, j))

        # Cap target if we asked for more than exists
        target = min(num_target_pairs, len(all_possible_negatives))

        if len(all_possible_negatives) > target:
            indices = np.random.choice(
                len(all_possible_negatives), target, replace=False
            )
            negative_pairs = [all_possible_negatives[i] for i in indices]
        else:
            negative_pairs = all_possible_negatives

    # STRATEGY 2: Large Dataset - Efficient Label Sampling
    else:
        unique_labels = np.array(unique_labels)
        # Cap target at a reasonable number if it's huge, but usually caller decides.
        # Max negatives is huge, so we just try to fill the quota.

        max_attempts = num_target_pairs * 5
        attempts = 0

        while len(negative_pairs) < num_target_pairs and attempts < max_attempts:
            # 1. Pick two different identities
            l1, l2 = np.random.choice(unique_labels, 2, replace=False)

            # 2. Pick one image from each identity
            idx1 = np.random.choice(labels_to_indices[l1])
            idx2 = np.random.choice(labels_to_indices[l2])

            # 3. Add sorted tuple (handles duplicates)
            negative_pairs.add(tuple(sorted((idx1, idx2))))
            attempts += 1

        negative_pairs = list(negative_pairs)

    return negative_pairs


def _calculate_pair_scores(
    embeddings: torch.Tensor, pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """Calculate cosine similarity scores for a list of index pairs."""
    if not pairs:
        return np.array([])

    scores = np.array(
        [
            F.cosine_similarity(
                embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
            ).item()
            for i, j in pairs
        ]
    )
    return scores


def calculate_tar_at_far(
    embeddings: np.ndarray,
    labels: np.ndarray,
    far_threshold: float = 0.01,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Calculate True Accept Rate at given False Accept Rate.
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert to torch tensors if needed
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    labels = np.array(labels)

    # Pre-compute label to indices map
    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)

    # 1. Generate Positive Pairs
    positive_pairs = _generate_positive_pairs(labels, labels_to_indices)
    if not positive_pairs:
        return 0.0, 0.0

    # 2. Generate Negative Pairs
    # Target 2x negative pairs for stable statistics
    num_target_negatives = len(positive_pairs) * 2
    negative_pairs = _generate_negative_pairs(
        labels, labels_to_indices, num_target_negatives
    )

    if not negative_pairs:
        return 0.0, 0.0

    # 3. Calculate Scores
    pos_scores = _calculate_pair_scores(embeddings, positive_pairs)
    neg_scores = _calculate_pair_scores(embeddings, negative_pairs)

    # 4. Calculate Threshold and TAR
    if len(neg_scores) == 0:
        return 0.0, 0.0

    threshold = np.quantile(neg_scores, 1 - far_threshold)
    tar = np.sum(pos_scores > threshold) / len(pos_scores)

    return tar, threshold


def _calculate_map(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Mean Average Precision (mAP) for the embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)
    aps = []

    for i in range(len(labels)):
        # Get similarities for this query
        query_sims = similarities[i]
        # Create relevance labels (same identity = relevant)
        relevance = (labels == labels[i]).astype(int)
        relevance[i] = 0  # Exclude self

        if np.sum(relevance) > 0:  # Only if there are relevant items
            ap = average_precision_score(relevance, query_sims)
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


def evaluate_embedding_model(model, dataloader, device) -> Dict[str, float]:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model.get_embeddings(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())

    if not all_embeddings:
        return {}

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    # Calculate metrics
    tar_1, threshold_1 = calculate_tar_at_far(all_embeddings, all_labels, 0.01)
    tar_001, threshold_001 = calculate_tar_at_far(all_embeddings, all_labels, 0.001)

    # Calculate mAP
    map_score = _calculate_map(all_embeddings, all_labels)

    return {
        "mAP": map_score,
        "TAR@FAR=1%": tar_1,
        "TAR@FAR=0.1%": tar_001,
        "threshold_1%": threshold_1,
        "threshold_0.1%": threshold_001,
        "num_identities": len(np.unique(all_labels)),
    }
