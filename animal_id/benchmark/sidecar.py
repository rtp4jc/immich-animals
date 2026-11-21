"""Tables for scripts/evaluate_sidecar.py, from per-photo sidecar detections."""

from collections import defaultdict

import numpy as np

from animal_id.identification import cluster, cluster_quality

SCORE_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7]
EPS_SWEEP = [0.25, 0.3, 0.35, 0.4, 0.45]


def _hits(detections: list[dict], threshold: float) -> list[dict]:
    return [d for d in detections if d["score"] >= threshold]


def _group(rec: dict) -> str:
    # Unnamed cat photos are labelled "negative" but still contain a cat.
    return f"{rec['species']}/{rec['source']}" if rec["species"] else "negative"


def score_sweep(records: list[dict], preds: dict) -> None:
    """Recall on animal photos and false-positive rate on animal-free ones, per minScore."""
    groups = defaultdict(list)
    for rec in records:
        if rec["path"] in preds:
            groups[_group(rec)].append(preds[rec["path"]])

    header = f"{'minScore':>9}  " + "  ".join(f"{k[:18]:>18}" for k in sorted(groups))
    print("\nDetection: share of photos with at least one detection")
    print(header)
    print("-" * len(header))
    for threshold in SCORE_SWEEP:
        cells = []
        for key in sorted(groups):
            rows = groups[key]
            fired = sum(1 for d in rows if _hits(d, threshold))
            per_photo = np.mean([len(_hits(d, threshold)) for d in rows])
            cells.append(f"{fired / len(rows):>10.3f} ({per_photo:>4.2f})")
        print(f"{threshold:>9.2f}  " + "  ".join(cells))
    print("  (share of photos, and mean detections per photo)")
    for key in sorted(groups):
        print(f"  {key}: {len(groups[key])} photos")


def negative_breakdown(records: list[dict], preds: dict, threshold: float) -> None:
    """Which kinds of animal-free photo the detector actually fires on."""
    by_category = defaultdict(lambda: [0, 0])
    for rec in records:
        if rec["species"] or rec["path"] not in preds:
            continue
        name = rec["category"].replace("Category:Quality images of ", "")
        name = name.replace("Category:Featured pictures of ", "")
        by_category[name][1] += 1
        if _hits(preds[rec["path"]], threshold):
            by_category[name][0] += 1
    print(f"\nFalse positives by subject at minScore {threshold}")
    for name, (fired, total) in sorted(
        by_category.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)
    ):
        print(f"  {name:<28} {fired:>3}/{total:<4} {fired / max(total, 1):>6.3f}")


def eps_sweep(
    records: list[dict], preds: dict, threshold: float, species: str, source: str
) -> None:
    """Cluster the best detection per identity photo, scored against the labels."""
    vectors, labels, kinds = [], [], []
    for rec in records:
        if rec["label"] == "negative" or rec["path"] not in preds:
            continue
        if species != "all" and rec["species"] != species:
            continue
        if source != "all" and rec["source"] != source:
            continue
        hits = _hits(preds[rec["path"]], threshold)
        if not hits:
            continue  # a miss is already counted by the detection sweep
        vectors.append(max(hits, key=lambda d: d["score"])["embedding"])
        labels.append(rec["label"])
        kinds.append(rec["species"])
    if not vectors:
        return

    embeddings = np.asarray(vectors, dtype=np.float32)
    header = (
        f"{'eps':>6}  {'clusters':>8}  {'true_ids':>8}  {'homog':>7}  "
        f"{'compl':>7}  {'v_meas':>7}  {'purity':>7}  {'noise':>7}  {'mixed':>5}"
    )
    print(
        f"\nClustering [{species}/{source}]: {len(labels)} crops, minScore {threshold}"
    )
    print(header)
    print("-" * len(header))
    for eps in EPS_SWEEP:
        pred = cluster(embeddings, eps=eps, min_samples=3)
        m = cluster_quality(labels, pred)
        members = defaultdict(set)
        for c, kind in zip(pred, kinds):
            members[c].add(kind)
        mixed = sum(len(v) > 1 for c, v in members.items() if c != -1)
        print(
            f"{eps:>6.2f}  {m['num_clusters']:>8d}  {m['num_true_identities']:>8d}  "
            f"{m['homogeneity']:>7.4f}  {m['completeness']:>7.4f}  "
            f"{m['v_measure']:>7.4f}  {m['purity']:>7.4f}  {m['noise_rate']:>7.4f}  "
            f"{mixed:>5d}"
        )


def changed_detections(
    records: list[dict], preds: dict, baseline: dict, threshold: float
) -> tuple[dict[str, list[tuple[str, int, int]]], float]:
    """Photos whose detection count changed from baseline, per group, and the largest score change on the rest."""
    changed = defaultdict(list)
    drift = 0.0
    for rec in records:
        path = rec["path"]
        if path not in preds or path not in baseline:
            continue
        now = sorted(d["score"] for d in _hits(preds[path], threshold))
        before = sorted(d["score"] for d in _hits(baseline[path], threshold))
        if len(now) != len(before):
            changed[_group(rec)].append((path, len(before), len(now)))
        elif now:
            drift = max(drift, float(np.max(np.abs(np.subtract(now, before)))))
    return dict(changed), drift


def print_changes(
    records: list[dict], preds: dict, baseline: dict, threshold: float, name: str
) -> None:
    changed, drift = changed_detections(records, preds, baseline, threshold)
    total = sum(len(rows) for rows in changed.values())
    print(f"\nChanged vs baseline '{name}' at minScore {threshold}: {total} photos")
    for key in sorted(changed):
        print(f"  {key}:")
        for path, before, now in changed[key]:
            print(f"    {before} -> {now}  {path}")
    print(f"  largest score change on the other photos: {drift:.2g}")
