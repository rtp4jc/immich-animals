"""Stage D: prove the trained finalists are deployable.

Stage B measures accuracy on the *PyTorch* model. What Immich runs is an ONNX
graph under ONNX Runtime, so the deploy gate is a separate question: does the
exported artifact still produce the same 512-d L2-normalised vector?

For each finalist this exports its best seed's checkpoint, then checks:
  - output is (1, 512) and L2-normalised (Immich's embedding contract)
  - ONNX Runtime output matches PyTorch within tolerance
  - ORT CPU latency and .onnx size on the real artifact, not a random-init proxy

    uv run python scripts/stage_d_validate.py
    uv run python scripts/stage_d_validate.py --backbone convnext_tiny

Writes ``outputs/ablation/stage_d.csv``.
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from animal_id.common.constants import DATA_DIR
from animal_id.common.datasets import IdentityDataset
from animal_id.common.logging_config import setup_logging
from animal_id.embedding.backbones import (
    BackboneType,
    get_backbone_input_size,
    get_backbone_license,
)
from animal_id.embedding.config import TRAINING_CONFIG
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.losses import HeadType
from animal_id.embedding.models import AnimalEmbeddingModel

setup_logging(__name__, logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
STAGE_D_CSV = OUTPUT_DIR / "stage_d.csv"
RUNS_DIR = PROJECT_ROOT / "runs"
PARITY_TOL = 1e-4

FIELDS = [
    "backbone",
    "head",
    "seed",
    "license",
    "mrr",
    "embedding_dim",
    "l2_norm",
    "parity_max_abs_diff",
    "parity_ok",
    "contract_ok",
    "ort_ms",
    "onnx_mb",
    "run_dir",
    "error",
]


CANONICAL_SEED = "42"


def best_cells(rows, backbones=None, heads=None):
    """One representative cell per (backbone, head) finalist.

    Prefers the canonical seed over the highest-scoring one: max-over-seeds on
    the split you report is an optimistic bias.
    """
    best = {}
    for row in rows:
        if row.get("mode") != "finetune" or (row.get("status") or "ok") != "ok":
            continue
        if backbones and row["backbone"] not in backbones:
            continue
        if heads and row["head"] not in heads:
            continue
        try:
            mrr = float(row["mrr"])
        except (TypeError, ValueError):
            continue
        key = (row["backbone"], row["head"])
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = row
        elif row["seed"] == CANONICAL_SEED:
            best[key] = row
        elif incumbent["seed"] != CANONICAL_SEED and mrr > float(incumbent["mrr"]):
            best[key] = row
    return best


def validate(row, num_classes) -> dict:
    backbone = BackboneType(row["backbone"])
    head = HeadType(row["head"])
    img_size = get_backbone_input_size(backbone)
    run_dir = RUNS_DIR / f"{row['timestamp']}_{backbone.value}_finetune_s{row['seed']}"

    out = {
        "backbone": backbone.value,
        "head": head.value,
        "seed": row["seed"],
        "license": get_backbone_license(backbone).value,
        "mrr": row["mrr"],
        "run_dir": run_dir.name,
        "error": "",
    }

    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        out["error"] = f"missing checkpoint {checkpoint.name}"
        out["contract_ok"] = False
        return out

    model = AnimalEmbeddingModel(
        backbone_type=backbone,
        num_classes=num_classes,
        embedding_dim=TRAINING_CONFIG.embedding_dim,
        pretrained=False,
        head_type=head,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    onnx_path = OUTPUT_DIR / f"stage_d_{backbone.value}_{head.value}.onnx"
    export_embedding_onnx(model, onnx_path, img_size=img_size)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    onnx_out = session.run(None, {name: dummy.numpy()})[0]

    diff = float(np.abs(torch_out - onnx_out).max())
    norm = float(np.linalg.norm(onnx_out[0]))
    dim = int(onnx_out.shape[1])

    for _ in range(3):
        session.run(None, {name: dummy.numpy()})
    start = time.perf_counter()
    for _ in range(30):
        session.run(None, {name: dummy.numpy()})
    ort_ms = (time.perf_counter() - start) / 30 * 1000.0

    out.update(
        {
            "embedding_dim": dim,
            "l2_norm": round(norm, 6),
            "parity_max_abs_diff": f"{diff:.2e}",
            "parity_ok": diff < PARITY_TOL,
            "contract_ok": (
                dim == TRAINING_CONFIG.embedding_dim
                and abs(norm - 1.0) < 1e-4
                and diff < PARITY_TOL
            ),
            "ort_ms": round(ort_ms, 1),
            "onnx_mb": round(onnx_path.stat().st_size / 1e6, 1),
        }
    )
    onnx_path.unlink(missing_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", nargs="+", default=None)
    parser.add_argument("--head", nargs="+", default=None)
    args = parser.parse_args()

    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    finalists = best_cells(rows, args.backbone, args.head)
    if not finalists:
        print("No fine-tuned cells to validate.")
        return

    num_classes = IdentityDataset(
        json_path=DATA_DIR / "identity_train.json", img_size=224, is_training=False
    ).num_classes

    results = []
    for (backbone, head), row in sorted(finalists.items()):
        print(f"--- {backbone} / {head} (seed {row['seed']}) ---")
        try:
            result = validate(row, num_classes)
        except Exception as exc:  # noqa: BLE001 - a failed export is the result
            result = {
                "backbone": backbone,
                "head": head,
                "seed": row["seed"],
                "contract_ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        mark = "PASS" if result.get("contract_ok") else "FAIL"
        print(
            f"    {mark}  dim={result.get('embedding_dim')} "
            f"norm={result.get('l2_norm')} diff={result.get('parity_max_abs_diff')} "
            f"ort={result.get('ort_ms')}ms {result.get('error')}"
        )
        results.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Merge rather than truncate, so a partial re-run cannot drop finalists.
    existing = {}
    if STAGE_D_CSV.exists():
        with open(STAGE_D_CSV, newline="") as f:
            existing = {(r["backbone"], r["head"]): r for r in csv.DictReader(f)}
    existing.update({(r["backbone"], r["head"]): r for r in results})
    results = list(existing.values())

    with open(STAGE_D_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k, "") for k in FIELDS})
    print(f"\nWrote {STAGE_D_CSV}")


if __name__ == "__main__":
    main()
