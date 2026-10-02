"""Cluster embeddings the way Immich's face recognition job assigns faces to people.

``eps`` maps to Immich's ``maxDistance`` and ``min_samples`` to ``minFaces``; label
``-1`` is unassigned. Plain DBSCAN chains every connected neighbour into one
cluster, which merged dense cat embeddings that Immich keeps apart.
"""

import numpy as np

from animal_id.common.embeddings import normalize_embeddings


def cluster(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 3,
) -> np.ndarray:
    """Immich's handleRecognizeFaces over ``(N, D)`` embeddings, in input order."""
    normed = normalize_embeddings(embeddings)
    distance = 1 - normed @ normed.T
    person = np.full(len(normed), -1)
    next_person = 0
    deferred = []
    # Immich defers faces without minFaces neighbours to a second pass, where
    # they may join an existing person but never start one.
    for second_pass, queue in ((False, range(len(normed))), (True, deferred)):
        for i in queue:
            row = distance[i]
            near = np.flatnonzero(row <= eps)
            near = near[np.argsort(row[near], kind="stable")][:min_samples]
            if min_samples > 1 and len(near) <= 1:
                continue  # only itself
            core = len(near) >= min_samples
            if not core and not second_pass:
                deferred.append(i)
                continue
            match = next((person[j] for j in near if person[j] >= 0), -1)
            if match < 0:
                # Immich then looks for the closest face already assigned to a person.
                assigned = np.flatnonzero((person >= 0) & (row <= eps))
                if len(assigned):
                    match = person[assigned[np.argmin(row[assigned])]]
            if match < 0 and core:
                match, next_person = next_person, next_person + 1
            person[i] = match
    return person
