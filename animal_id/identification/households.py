"""Simulated photo libraries clustered the way Immich assigns faces, scored per own pet.

homes = sample_homes(ids, halves(ids)[1], n_homes=300, seed=2)
simulate(embeddings, ids, homes, eps=0.375)["pure"]
"""

from collections import Counter, defaultdict

import numpy as np

from .clusterer import cluster

MIN_FACES = 3

# Metric definitions; per-pet ones are pooled over every own pet of every home.
# A "person" is a cluster; unassigned photos (-1) belong to no person.
# pure: "in one correct person", the release-notes headline. The person holding
#   most of the pet's photos is >= 90% that pet and holds >= 2 of its photos.
# clean: pure, and that person holds >= 80% of the pet's photos.
# missed: none of the pet's photos is in a person.
# split: the pet is the majority of 2+ persons.
# merged: a person holding >= 2 of the pet's photos also holds >= 2 photos of
#   another own pet of the same home.
# clustered: share of the pet's photos that are in a person.
# absorbed: share of stranger photos in a person whose majority is an own pet.
# precision/recall/f1: BCubed over own-pet photos, an unassigned photo being
#   its own singleton; averaged over homes.
PET_METRICS = ("pure", "clean", "missed", "split", "merged", "clustered")


def halves(ids: np.ndarray, seed: int = 0) -> tuple[set, set]:
    """Pets split into two disjoint halves: choose Max Distance on one, report the other."""
    pets = np.array(sorted(set(ids)))
    np.random.default_rng(seed).shuffle(pets)
    return set(pets[: len(pets) // 2]), set(pets[len(pets) // 2 :])


def sample_homes(
    ids: np.ndarray,
    pets: set,
    n_homes: int,
    seed: int,
    breeds: dict[str, str] | None = None,
) -> list[tuple[np.ndarray, set]]:
    """(photo indices, own pets) per home: 1-4 own pets with 3-15 photos, 20-200 strangers with 1-2.

    With ``breeds``, 30% of multi-pet homes own pets of a single breed, the hardest case.
    """
    photos = defaultdict(list)
    for i, pet in enumerate(ids):
        if pet in pets:
            photos[pet].append(i)
    eligible = sorted(p for p, v in photos.items() if len(v) >= 3)
    by_breed = defaultdict(list)
    for p in eligible:
        if breeds and breeds.get(p, "") not in ("", "Mixed"):
            by_breed[breeds[p]].append(p)
    homes = []
    for h in range(n_homes):
        rng = np.random.default_rng(seed * 100003 + h)
        n_own = int(rng.integers(1, 5))
        own = None
        if n_own >= 2 and rng.random() < 0.3:
            breed = sorted(b for b, v in by_breed.items() if len(v) >= n_own)
            if breed:
                breed = breed[rng.integers(len(breed))]
                own = list(rng.choice(by_breed[breed], n_own, replace=False))
        if own is None:
            own = list(rng.choice(eligible, n_own, replace=False))
        chosen = []
        for p in own:
            k = min(int(rng.integers(3, 16)), len(photos[p]))
            chosen += list(rng.choice(photos[p], k, replace=False))
        strangers = sorted(set(photos) - set(own))
        n_strangers = min(int(rng.integers(20, 201)), len(strangers))
        for p in rng.choice(strangers, n_strangers, replace=False):
            k = min(1 if rng.random() < 0.6 else 2, len(photos[p]))
            chosen += list(rng.choice(photos[p], k, replace=False))
        homes.append((np.array(chosen), set(own)))
    return homes


def score_home(person: np.ndarray, labels: np.ndarray, own: set) -> dict:
    """Per-pet outcomes, stranger absorption and own-pet BCubed for one clustered home."""
    persons = defaultdict(Counter)
    for p, label in zip(person, labels, strict=True):
        if p >= 0:
            persons[p][label] += 1
    majority = {p: c.most_common(1)[0][0] for p, c in persons.items()}
    total = Counter(labels)
    pets = {}
    for pet in own:
        held = {p: c[pet] for p, c in persons.items() if c[pet]}
        top = max(held, key=held.get, default=None)
        pure = (
            top is not None
            and held[top] >= 2
            and held[top] / persons[top].total() >= 0.9
        )
        pets[pet] = {
            "pure": pure,
            "clean": pure and held[top] / total[pet] >= 0.8,
            "missed": not held,
            "split": sum(majority[p] == pet for p in held) >= 2,
            "merged": any(
                n >= 2
                and any(o != pet and o in own and k >= 2 for o, k in persons[p].items())
                for p, n in held.items()
            ),
            "clustered": sum(held.values()) / total[pet],
        }
    precision, recall = [], []
    for p, label in zip(person, labels, strict=True):
        if label in own:
            same, size = (persons[p][label], persons[p].total()) if p >= 0 else (1, 1)
            precision.append(same / size)
            recall.append(same / total[label])
    stranger = np.array([label not in own for label in labels])
    absorbed = sum(p >= 0 and majority[p] in own for p in person[stranger])
    return {
        "pets": pets,
        "absorbed": int(absorbed),
        "strangers": int(stranger.sum()),
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
    }


def simulate(embeddings: np.ndarray, ids: np.ndarray, homes: list, eps: float) -> dict:
    """Clusters each home in a fixed random upload order and pools the metrics above."""
    scores = []
    for h, (photos, own) in enumerate(homes):
        order = np.random.default_rng(h).permutation(len(photos))
        person = np.empty(len(photos), int)
        person[order] = cluster(embeddings[photos[order]], eps, MIN_FACES)
        scores.append(score_home(person, ids[photos], own))
    pets = [pet for s in scores for pet in s["pets"].values()]
    precision = np.array([s["precision"] for s in scores])
    recall = np.array([s["recall"] for s in scores])
    return {
        **{m: float(np.mean([pet[m] for pet in pets])) for m in PET_METRICS},
        "absorbed": sum(s["absorbed"] for s in scores)
        / sum(s["strangers"] for s in scores),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(np.mean(2 * precision * recall / (precision + recall))),
        "pets": len(pets),
    }
