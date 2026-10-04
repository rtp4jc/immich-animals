"""Every model-card number for shipped embedders: held-out retrieval and simulated households.

uv run python scripts/evaluate_release.py --embedder models/onnx/0.2.0/release models/onnx/0.3.0/release
Metric definitions live in animal_id/identification/households.py.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from animal_id.benchmark.metrics import calculate_tar_at_far, retrieval_metrics
from animal_id.common.constants import DATA_DIR, PROJECT_ROOT
from animal_id.common.embeddings import normalize_embeddings
from animal_id.data import dedupe
from animal_id.data.images import crop
from animal_id.data.sample import Source
from animal_id.data.sources import cat_individuals, dogreid
from animal_id.identification import households
from animal_id.pipeline.onnx_models import ONNXEmbedding

CACHE = PROJECT_ROOT / "outputs" / "evaluate_release"
SWEEP = (0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45)
HOUSEHOLD_COLUMNS = (*households.PET_METRICS, "absorbed", "f1")


def held_out(species: str) -> tuple[list, np.ndarray | None]:
    """The species' benchmark samples and, for DogReID, its gallery mask."""
    if species == "dog":
        frames = dogreid.benchmark(DATA_DIR)
        return [s for s, _ in frames], np.array([r == "gallery" for _, r in frames])
    # The cats are shot in bursts, which makes every embedder look near-perfect.
    cats = cat_individuals.benchmark(DATA_DIR)
    return dedupe.drop_bursts(cats, (Source.CAT_INDIVIDUALS,)), None


def embed(embedder: Path, species: str, samples: list) -> np.ndarray:
    """The ONNX embedder on each sample's padded box, cached per model file."""
    with open(embedder / "embedding.onnx", "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()[:12]
    cache = CACHE / digest / f"{species}.npz"
    paths = np.array([s.path for s in samples])
    if cache.exists():
        with np.load(cache) as cached:
            if np.array_equal(cached["paths"], paths):
                return cached["embeddings"]
    model = ONNXEmbedding(str(embedder / "embedding.onnx"))
    batches = []
    for start in range(0, len(samples), 32):
        crops = []
        for sample in samples[start : start + 32]:
            with Image.open(DATA_DIR / sample.path) as image:
                crops.append(np.asarray(crop(image, sample.boxes[0].xyxy)))
        batches.append(model.predict_batch(crops))
    embeddings = normalize_embeddings(np.concatenate(batches))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, embeddings=embeddings, paths=paths)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", type=Path, nargs="+", required=True)
    parser.add_argument("--dog-eps", type=float, default=0.375)
    parser.add_argument("--cat-eps", type=float, default=0.35)
    parser.add_argument("--sweep", action="store_true", help=f"households at {SWEEP}")
    args = parser.parse_args()

    samples, gallery = {}, {}
    for s in ("dog", "cat"):
        samples[s], gallery[s] = held_out(s)
    ids = {s: np.array([x.boxes[0].identity for x in samples[s]]) for s in samples}
    # Half 1 at seed 2 are the homes the 0.3.0 model card was first published on.
    breeds = {"dog": dogreid.breeds(DATA_DIR), "cat": None}
    homes = {
        s: households.sample_homes(
            ids[s], households.halves(ids[s])[1], 300, 2, breeds[s]
        )
        for s in samples
    }
    print(
        f"dog frames {len(ids['dog'])}, cats {len(set(ids['cat']))} with {len(ids['cat'])} photos"
    )

    print(
        "\n| Embedder | DogReID top-1 | Top-5 | MRR | TAR@FAR=1% | Cat top-1 | Cat TAR@FAR=1% |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    embeddings = {}
    for embedder in args.embedder:
        e = {s: embed(embedder, s, samples[s]) for s in samples}
        embeddings[embedder] = e
        dog_mrr, dog_top, _ = retrieval_metrics(
            e["dog"], ids["dog"], gallery=gallery["dog"]
        )
        _, cat_top, _ = retrieval_metrics(e["cat"], ids["cat"], gallery=gallery["cat"])
        tar = {s: calculate_tar_at_far(e[s], ids[s], 0.01)[0] for s in samples}
        print(
            f"| {embedder} | {dog_top[1]:.3f} | {dog_top[5]:.3f} | {dog_mrr:.3f} "
            f"| {tar['dog']:.3f} | {cat_top[1]:.3f} | {tar['cat']:.3f} |"
        )

    print(f"\n| Embedder | Species | Max Distance | {' | '.join(HOUSEHOLD_COLUMNS)} |")
    print("| --- " * (3 + len(HOUSEHOLD_COLUMNS)) + "|")
    for embedder, e in embeddings.items():
        for species, shipped in (("dog", args.dog_eps), ("cat", args.cat_eps)):
            for eps in SWEEP if args.sweep else (shipped,):
                r = households.simulate(e[species], ids[species], homes[species], eps)
                cells = " | ".join(f"{r[c]:.1%}" for c in HOUSEHOLD_COLUMNS[:-1])
                print(f"| {embedder} | {species} | {eps} | {cells} | {r['f1']:.3f} |")


if __name__ == "__main__":
    main()
