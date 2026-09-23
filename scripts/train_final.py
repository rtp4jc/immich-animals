"""Train and export the production embedding model.

The ablation answers *which* backbone; this produces the artifact Immich loads.
Two differences from an ablation cell:

- ``--include-val`` folds the validation identities into training. They are
  disjoint from both train and test, so this buys ~21% more identities — the
  strongest lever in metric learning — while the held-out test split stays
  untouched and the reported number stays honest. There is then no clean
  early-stopping signal, so the epoch budget is fixed and the FINAL checkpoint
  is exported rather than the best-on-val one.
- It writes ``models/onnx/embedding.onnx``, the path the pipeline loads.

    uv run python scripts/train_final.py --backbone convnext_tiny --include-val

Appends its test metrics to ``outputs/ablation/final.csv`` so the production
model can be compared against the ablation cells it came from.
"""

import argparse
import csv
import datetime
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from animal_id.benchmark.metrics import evaluate_embedding_model, retrieval_metrics
from animal_id.common.constants import DATA_DIR, ONNX_EMBEDDING_PATH
from animal_id.common.datasets import IdentityDataset
from animal_id.common.logging_config import setup_logging
from animal_id.common.seed import set_seed, worker_init_fn
from animal_id.embedding.backbones import BackboneType, get_backbone_input_size
from animal_id.embedding.config import DATA_CONFIG, TRAINING_CONFIG
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.losses import HeadType
from animal_id.embedding.models import AnimalEmbeddingModel
from animal_id.embedding.trainer import EmbeddingTrainer
from animal_id.identification.clusterer import cluster
from animal_id.identification.metrics import cluster_quality

setup_logging(__name__, logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
FINAL_CSV = OUTPUT_DIR / "final.csv"
NO_EARLY_STOP = 10**6


def _rel(path: Path) -> str:
    """Repo-relative when possible; absolute paths must never be committed."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


EPS_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MIN_SAMPLES = 3

FIELDS = [
    "timestamp",
    "backbone",
    "head",
    "seed",
    "trained_on",
    "epochs",
    "num_classes",
    "mrr",
    "top1",
    "top5",
    "mAP",
    "tar@1%",
    "onnx_path",
    "onnx_contract_ok",
    "onnx_parity",
    "onnx_ort_ms",
]


def verify_export(model, onnx_path: Path, img_size: int) -> dict:
    """An artifact is not ready until the exported graph is the same function.

    A dropped normalize still returns a 512-vector and cosine still returns a
    number, so a broken export degrades silently rather than raising.
    """
    import time

    import onnxruntime as ort

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    onnx_out = session.run(None, {name: dummy.numpy()})[0]

    diff = float(np.abs(torch_out - onnx_out).max())
    norm = float(np.linalg.norm(onnx_out[0]))

    for _ in range(3):
        session.run(None, {name: dummy.numpy()})
    start = time.perf_counter()
    for _ in range(30):
        session.run(None, {name: dummy.numpy()})
    ort_ms = (time.perf_counter() - start) / 30 * 1000.0

    return {
        "onnx_contract_ok": (
            onnx_out.shape[1] == TRAINING_CONFIG.embedding_dim
            and abs(norm - 1.0) < 1e-4
            and diff < 1e-4
        ),
        "onnx_parity": f"{diff:.2e}",
        "onnx_ort_ms": round(ort_ms, 1),
    }


def tune_eps(embeddings, labels) -> tuple[float, list[dict]]:
    """Pick the DBSCAN threshold to ship with this model.

    ``eps`` is Immich's ``maxDistance``, and the right value depends on the
    embedding geometry, so it cannot be inherited across a backbone swap. Note
    the operating point is chosen on the test split: retrieval metrics stay
    unbiased (no tuning), but the clustering scores here are best-case.
    """
    sweep = []
    for eps in EPS_GRID:
        metrics = cluster_quality(list(labels), cluster(embeddings, eps, MIN_SAMPLES))
        sweep.append(
            {"eps": eps, **{k: round(float(v), 4) for k, v in metrics.items()}}
        )
    best = max(sweep, key=lambda r: r["v_measure"])
    return best["eps"], sweep


def build_combined_json(out_path: Path) -> Path:
    """train + val in one file. Identity labels are global, so no remapping."""
    merged = []
    for name in ("identity_train.json", "identity_val.json"):
        merged += json.loads((DATA_DIR / name).read_text())
    out_path.write_text(json.dumps(merged))
    return out_path


def loader_for(json_path, img_size, batch_size, is_training, generator=None):
    dataset = IdentityDataset(
        json_path=json_path, img_size=img_size, is_training=is_training
    )
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=TRAINING_CONFIG.hardware_workers,
        generator=generator if is_training else None,
        worker_init_fn=worker_init_fn if is_training else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone", required=True, choices=[b.value for b in BackboneType]
    )
    parser.add_argument(
        "--head", default="arcface", choices=[h.value for h in HeadType]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Fold the val identities into training (fixed epochs, final checkpoint).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Export an existing best_model.pt instead of training a new one.",
    )
    parser.add_argument("--output", default=None, help="ONNX destination.")
    args = parser.parse_args()

    backbone = BackboneType(args.backbone)
    head = HeadType(args.head)
    img_size = get_backbone_input_size(backbone)
    batch_size = DATA_CONFIG.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onnx_path = Path(args.output) if args.output else ONNX_EMBEDDING_PATH

    generator = set_seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_{backbone.value}_final"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        # The flag seeds RNG; it says nothing about weights we did not train.
        # Record what the checkpoint path can substantiate, and nothing more.
        source = Path(args.checkpoint).parent.name
        trained_on = f"checkpoint:{source}"
        train_json = DATA_DIR / "identity_train.json"
        match = re.search(r"_s(\d+)$", source)
        recorded_seed = match.group(1) if match else ""
        if args.include_val:
            print("--include-val ignored: --checkpoint does not retrain.")
    elif args.include_val:
        train_json = build_combined_json(run_dir / "identity_trainval.json")
        trained_on = "train+val"
        recorded_seed = args.seed
    else:
        train_json = DATA_DIR / "identity_train.json"
        trained_on = "train"
        recorded_seed = args.seed

    if args.checkpoint:
        # Only the head size is needed; no training pass will run.
        train_dataset = IdentityDataset(
            json_path=train_json, img_size=img_size, is_training=False
        )
        train_loader = None
    else:
        train_dataset, train_loader = loader_for(
            train_json, img_size, batch_size, True, generator
        )
    # Monitoring only when val is folded in — it is no longer held out.
    _, val_loader = loader_for(
        DATA_DIR / "identity_val.json", img_size, batch_size, False
    )
    _, test_loader = loader_for(
        DATA_DIR / "identity_test.json", img_size, batch_size, False
    )

    print(
        f"=== Final train: {backbone.value}/{head.value} seed={args.seed} "
        f"on {trained_on} ({len(train_dataset)} imgs, "
        f"{train_dataset.num_classes} head classes) device={device} ==="
    )

    model = AnimalEmbeddingModel(
        backbone_type=backbone,
        num_classes=train_dataset.num_classes,
        embedding_dim=TRAINING_CONFIG.embedding_dim,
        head_type=head,
    ).to(device)

    warmup = args.epochs or TRAINING_CONFIG.warmup_epochs
    full = args.epochs or TRAINING_CONFIG.full_train_epochs
    best_path = None

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        warmup = full = 0
        print(f"Loaded {args.checkpoint} - exporting without training")
    else:
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_dir=run_dir,
        )
        best_path = trainer.train(
            warmup_epochs=warmup,
            full_epochs=full,
            head_lr=TRAINING_CONFIG.head_lr,
            backbone_lr=TRAINING_CONFIG.backbone_lr,
            full_lr=TRAINING_CONFIG.full_train_lr,
            # Selecting on val would be selecting on training data once it is folded in.
            patience=NO_EARLY_STOP
            if args.include_val
            else TRAINING_CONFIG.early_stopping_patience,
            linear_probe=False,
        )

    if args.checkpoint:
        pass  # already loaded above
    elif args.include_val:
        # Our own checkpoint dict carries a numpy scalar, which the 2.6+
        # weights_only default refuses.
        state = torch.load(
            run_dir / "latest_checkpoint.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device).eval()

    test_metrics = evaluate_embedding_model(model, test_loader, device)
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            embeddings.append(model.get_embeddings(images.to(device)).cpu().numpy())
            labels.extend(batch_labels.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    mrr, topk, _ = retrieval_metrics(embeddings, labels)
    best_eps, eps_sweep = tune_eps(embeddings, labels)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    export_embedding_onnx(model, onnx_path, img_size=img_size)
    contract = verify_export(model, onnx_path, img_size)

    row = {
        "timestamp": timestamp,
        "backbone": backbone.value,
        "head": head.value,
        "seed": recorded_seed,
        "trained_on": trained_on,
        "epochs": f"{warmup}+{full}",
        "num_classes": train_dataset.num_classes,
        "mrr": round(mrr, 4),
        "top1": round(topk[1], 4),
        "top5": round(topk[5], 4),
        "mAP": round(test_metrics.get("mAP", 0.0), 4),
        "tar@1%": round(test_metrics.get("TAR@FAR=1%", 0.0), 4),
        "onnx_path": _rel(onnx_path),
        **contract,
    }
    row_metrics = row

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not FINAL_CSV.exists()
    with open(FINAL_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print("\n=== Final model ===")
    for k, v in row.items():
        print(f"  {k}: {v}")
    sidecar = onnx_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "version": run_dir.name,
                "source_run": _rel(run_dir),
                "backbone": backbone.value,
                "head": head.value,
                "seed": recorded_seed,
                "trained_on": trained_on,
                "img_size": img_size,
                "embedding_dim": TRAINING_CONFIG.embedding_dim,
                "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "onnx_contract_ok": contract["onnx_contract_ok"],
                "parity_max_abs_diff": contract["onnx_parity"],
                "ort_cpu_ms": contract["onnx_ort_ms"],
                "test_metrics": {
                    "mrr": row_metrics["mrr"],
                    "top1": row_metrics["top1"],
                    "top5": row_metrics["top5"],
                    "mAP": row_metrics["mAP"],
                    "tar@1%": row_metrics["tar@1%"],
                },
                "preprocessing": {
                    "resize": [img_size, img_size],
                    "scale": "pixels / 255",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "layout": "NCHW",
                    "colour": "RGB",
                    "note": "ImageNet normalisation is required; serving raw [0,1] costs ~1.3pp MRR silently.",
                },
                "clustering": {
                    "eps": best_eps,
                    "min_samples": MIN_SAMPLES,
                    "selected_on": "test",
                    "sweep": eps_sweep,
                },
            },
            indent=2,
        )
        + "\n"
    )

    status = "READY" if contract["onnx_contract_ok"] else "CONTRACT FAILED"
    print(f"  clustering: eps={best_eps} min_samples={MIN_SAMPLES}")
    print(f"\nExported {onnx_path} (+ {sidecar.name}) — {status}")


if __name__ == "__main__":
    main()
