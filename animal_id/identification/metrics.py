"""Cluster-quality metrics comparing DBSCAN assignments to ground-truth identities."""

from collections import Counter

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)


def cluster_quality(
    true_labels: list,
    pred_labels: "np.ndarray | list",
) -> dict:
    """Score predicted clusters against ground-truth labels.

    ``pred_labels`` uses ``-1`` for noise/unassigned. sklearn metrics
    (homogeneity, completeness, v_measure, ARI) see the raw labels including
    noise; purity is computed over non-noise points only (noise = images the
    system declined to assign, so it shouldn't penalise formed clusters), and is
    0.0 if every point is noise. Returns a dict of those metrics plus
    num_clusters, num_true_identities, and noise_rate.
    """
    pred_arr = np.asarray(pred_labels)
    n_total = len(true_labels)

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_arr
    )
    ari = adjusted_rand_score(true_labels, pred_arr)

    unique_pred = set(pred_arr.tolist())
    num_clusters = len(unique_pred - {-1})
    num_true_identities = len(set(true_labels))
    noise_rate = float(np.sum(pred_arr == -1)) / n_total if n_total > 0 else 0.0

    # Purity over non-noise points.
    non_noise_mask = pred_arr != -1
    non_noise_pred = pred_arr[non_noise_mask]
    non_noise_true = [t for t, m in zip(true_labels, non_noise_mask) if m]

    if len(non_noise_pred) == 0:
        purity = 0.0
    else:
        majority_sum = 0
        for cluster_id in set(non_noise_pred.tolist()):
            members_true = [
                t for t, p in zip(non_noise_true, non_noise_pred) if p == cluster_id
            ]
            majority_sum += Counter(members_true).most_common(1)[0][1]
        purity = majority_sum / len(non_noise_pred)

    return {
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "adjusted_rand_index": float(ari),
        "num_clusters": num_clusters,
        "num_true_identities": num_true_identities,
        "noise_rate": noise_rate,
        "purity": purity,
    }
