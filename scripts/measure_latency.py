"""Measure the cost axis for every backbone, on one instrument.

The ablation's cost gate compares candidates against each other, so every
number in it must come from the same measurement. ``run_ablation.py`` records
cost per training run, which means rows trained before ONNX-Runtime timing
existed carry only a torch number -- not comparable with a later ORT one.

Latency is a property of the architecture, not the learned weights, so this
measures untrained models and needs no GPU or training:

    uv run python scripts/measure_latency.py

Writes ``outputs/ablation/latency.csv`` (one row per backbone), which
``summarize_ablation.py`` prefers over the per-run columns.
"""

import argparse
import copy
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from animal_id.common.logging_config import setup_logging
from animal_id.embedding.backbones import (
    BackboneType,
    get_backbone_input_size,
    get_backbone_license,
)
from animal_id.embedding.config import TRAINING_CONFIG
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.models import AnimalEmbeddingModel

setup_logging(__name__, logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
LATENCY_CSV = OUTPUT_DIR / "latency.csv"

FIELDS = [
    "backbone",
    "license",
    "img_size",
    "params_total_M",
    "torch_ms",
    "ort_ms",
    "onnx_mb",
    "onnx_ok",
]


def _time_torch(model, img_size, iters):
    dummy = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(3):
            model.get_embeddings(dummy)
        start = time.perf_counter()
        for _ in range(iters):
            model.get_embeddings(dummy)
        return (time.perf_counter() - start) / iters * 1000.0


def _time_ort(onnx_path, img_size, iters):
    """Time the exported graph under the provider Immich's ML worker uses."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
    for _ in range(3):
        session.run(None, {name: dummy})
    start = time.perf_counter()
    for _ in range(iters):
        session.run(None, {name: dummy})
    return (time.perf_counter() - start) / iters * 1000.0


def measure(backbone: BackboneType, iters: int) -> dict:
    img_size = get_backbone_input_size(backbone)
    model = AnimalEmbeddingModel(
        backbone_type=backbone,
        num_classes=705,  # Head size does not affect the embedding path.
        embedding_dim=TRAINING_CONFIG.embedding_dim,
        pretrained=False,
    ).eval()

    row = {
        "backbone": backbone.value,
        "license": get_backbone_license(backbone).value,
        "img_size": img_size,
        "params_total_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "torch_ms": round(_time_torch(model, img_size, iters), 1),
        "ort_ms": "",
        "onnx_mb": "",
        "onnx_ok": False,
    }

    onnx_path = OUTPUT_DIR / f"_latency_{backbone.value}.onnx"
    try:
        export_embedding_onnx(copy.deepcopy(model), onnx_path, img_size=img_size)
        row["onnx_ok"] = True
        row["onnx_mb"] = round(onnx_path.stat().st_size / 1e6, 1)
        row["ort_ms"] = round(_time_ort(onnx_path, img_size, iters), 1)
    except Exception as exc:  # noqa: BLE001 - a failed export is a recorded result
        print(f"[{backbone.value}] export/ORT failed: {type(exc).__name__}: {exc}")
    finally:
        onnx_path.unlink(missing_ok=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        nargs="+",
        default=[b.value for b in BackboneType],
        choices=[b.value for b in BackboneType],
        metavar="BACKBONE",
    )
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.backbone:
        print(f"--- {name} ---")
        row = measure(BackboneType(name), args.iters)
        print(
            f"    torch {row['torch_ms']}ms  ort {row['ort_ms']}ms  "
            f"{row['params_total_M']}M  onnx {row['onnx_mb']}MB"
        )
        rows.append(row)

    # Merge, never truncate: a partial re-measure that dropped rows would make
    # summarize_ablation fall back from ORT to torch timings, and the torch
    # ratio passes the latency gate where the ORT one fails it.
    merged = {}
    if LATENCY_CSV.exists():
        with open(LATENCY_CSV, newline="") as f:
            merged = {r["backbone"]: r for r in csv.DictReader(f)}
    merged.update({row["backbone"]: row for row in rows})
    with open(LATENCY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged.values())
    print(f"\nWrote {LATENCY_CSV}")


if __name__ == "__main__":
    main()
