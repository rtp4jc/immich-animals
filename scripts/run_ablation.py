"""Run embedding-backbone ablation cells and record their results.

This is the harness for the backbone ablation described in
``.planning/6-25-2026-embedding-backbone-ablation/plan.md``. One invocation =
one or more (backbone, head, mode, seed) cells: it trains each embedding model,
evaluates it on the **held-out test split**, measures cost (params, CPU latency,
ONNX exportability), and appends a row to a shared results table.

Two modes:
- ``probe``    — linear probe: freeze the backbone trunk, train only the
                 projection + margin head. Fair, cheap feature-quality ranking
                 (no per-backbone LR to tune). Used for the broad Stage A sweep.
- ``finetune`` — full two-phase training (warmup head, then unfreeze + fine-tune
                 with differential LR). Used for the Stage B finalists.

Examples (run from the repo root, where ``data/`` lives):

    # Stage A: linear-probe a single candidate
    uv run python scripts/run_ablation.py --backbone convnextv2_tiny --mode probe

    # Stage A: sweep several backbones in one command
    uv run python scripts/run_ablation.py \
        --backbone convnextv2_tiny efficientnet_b3 mobilenetv3_large \
        --mode probe

    # Stage B: full fine-tune of the finalists over three seeds
    uv run python scripts/run_ablation.py --backbone convnextv2_tiny \
        --mode finetune --seed 42 1 2

    # Quick smoke test (1 epoch per phase) to confirm the cell runs
    uv run python scripts/run_ablation.py --backbone resnet50 --epochs 1 --smoke

Results are appended to ``outputs/ablation/results.csv`` (+ a regenerated
``results.md`` table). Cells already recorded there are skipped (``--force``
re-runs them) and a cell that raises is recorded as failed and does not stop the
sweep, so an unattended multi-day queue can be resumed by re-running the command.
``--dry-run`` prints that plan — which cells would run, which are already
recorded — without training anything.
"""

import argparse
import copy
import datetime
import logging
import time
import traceback
from pathlib import Path

import ablation_results
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from animal_id.benchmark.metrics import evaluate_embedding_model, retrieval_metrics
from animal_id.common.constants import DATA_DIR
from animal_id.common.datasets import IdentityDataset
from animal_id.common.logging_config import setup_logging
from animal_id.common.seed import set_seed, worker_init_fn
from animal_id.embedding.backbones import BackboneType, get_backbone_input_size
from animal_id.embedding.config import DATA_CONFIG, TRAINING_CONFIG
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.losses import HeadType
from animal_id.embedding.models import AnimalEmbeddingModel
from animal_id.embedding.trainer import EmbeddingTrainer

setup_logging(__name__, logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
RESULTS_MD = OUTPUT_DIR / "results.md"


def measure_cpu_latency(model, img_size, num_iters=20):
    """Mean single-image CPU inference latency (ms) for the embedding path."""
    latency_model = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(3):  # warmup
            latency_model.get_embeddings(dummy_input)
        start = time.perf_counter()
        for _ in range(num_iters):
            latency_model.get_embeddings(dummy_input)
        elapsed = time.perf_counter() - start
    return (elapsed / num_iters) * 1000.0


def measure_ort_latency(onnx_path, img_size, num_iters=20):
    """Same, through ONNX Runtime: that is what Immich's CPU ML worker runs."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
    for _ in range(3):  # warmup
        session.run(None, {input_name: dummy_input})
    start = time.perf_counter()
    for _ in range(num_iters):
        session.run(None, {input_name: dummy_input})
    elapsed = time.perf_counter() - start
    return (elapsed / num_iters) * 1000.0


def check_onnx_export(model, img_size):
    """Export to ONNX, then time it under ORT. Returns (export_ok, ort_ms)."""
    export_model = copy.deepcopy(model).to("cpu").eval()
    tmp_path = PROJECT_ROOT / "outputs" / "ablation_onnx_check.onnx"
    try:
        export_embedding_onnx(export_model, tmp_path, img_size=img_size)
        return True, round(measure_ort_latency(tmp_path, img_size), 1)
    except Exception as exc:  # noqa: BLE001 - we want any export failure recorded
        print(f"[onnx] export failed: {type(exc).__name__}: {exc}")
        return False, ""
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_loader(json_name, img_size, batch_size, is_training, generator=None):
    dataset = IdentityDataset(
        json_path=DATA_DIR / json_name,
        img_size=img_size,
        is_training=is_training,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=TRAINING_CONFIG.hardware_workers,
        generator=generator if is_training else None,
        worker_init_fn=worker_init_fn if is_training else None,
    )
    return dataset, loader


def run_cell(cell, backbone, head, args, epochs, device):
    """Train + evaluate one cell; returns the full results row."""
    img_size = cell["img_size"]
    batch_size = DATA_CONFIG.batch_size
    generator = set_seed(cell["seed"])

    train_dataset, train_loader = build_loader(
        DATA_CONFIG.train_json_path.split("/")[-1],
        img_size,
        batch_size,
        is_training=True,
        generator=generator,
    )
    _, val_loader = build_loader(
        DATA_CONFIG.val_json_path.split("/")[-1],
        img_size,
        batch_size,
        is_training=False,
    )
    _, test_loader = build_loader(
        "identity_test.json", img_size, batch_size, is_training=False
    )

    model = AnimalEmbeddingModel(
        backbone_type=backbone,
        num_classes=train_dataset.num_classes,
        embedding_dim=TRAINING_CONFIG.embedding_dim,
        head_type=head,
    ).to(device)

    run_dir = (
        PROJECT_ROOT
        / "runs"
        / f"{cell['timestamp']}_{backbone.value}_{args.mode}_s{cell['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = EmbeddingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
    )

    if args.mode == "probe":
        best_model_path = trainer.train(
            warmup_epochs=epochs or TRAINING_CONFIG.full_train_epochs,
            full_epochs=0,
            head_lr=TRAINING_CONFIG.head_lr,
            backbone_lr=TRAINING_CONFIG.backbone_lr,
            full_lr=TRAINING_CONFIG.full_train_lr,
            patience=TRAINING_CONFIG.early_stopping_patience,
            linear_probe=True,
        )
    else:
        best_model_path = trainer.train(
            warmup_epochs=epochs or TRAINING_CONFIG.warmup_epochs,
            full_epochs=epochs or TRAINING_CONFIG.full_train_epochs,
            head_lr=TRAINING_CONFIG.head_lr,
            backbone_lr=TRAINING_CONFIG.backbone_lr,
            full_lr=TRAINING_CONFIG.full_train_lr,
            patience=TRAINING_CONFIG.early_stopping_patience,
            linear_probe=False,
        )

    # --- Evaluate on the held-out TEST split ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device).eval()

    test_metrics = evaluate_embedding_model(model, test_loader, device)

    # Leave-one-out Top-k / MRR from the raw test embeddings.
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            embeddings.append(model.get_embeddings(images.to(device)).cpu().numpy())
            labels.extend(batch_labels.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    mrr, top_k_accuracy, num_queries = retrieval_metrics(embeddings, labels)

    # --- Cost axis ---
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    cpu_ms = measure_cpu_latency(model, img_size)
    if args.no_onnx:
        onnx_ok, ort_ms = "skipped", ""
    else:
        onnx_ok, ort_ms = check_onnx_export(model, img_size)

    print(f"Run artifacts in {run_dir}")
    return {
        **cell,
        "n_test_queries": num_queries,
        "mrr": round(mrr, 4),
        "top1": round(top_k_accuracy[1], 4),
        "top5": round(top_k_accuracy[5], 4),
        "mAP": round(test_metrics.get("mAP", 0.0), 4),
        "tar@1%": round(test_metrics.get("TAR@FAR=1%", 0.0), 4),
        "tar@0.1%": round(test_metrics.get("TAR@FAR=0.1%", 0.0), 4),
        "params_total_M": round(total_params, 2),
        "params_trainable_M": round(trainable_params, 2),
        "cpu_ms": round(cpu_ms, 1),
        "ort_cpu_ms": ort_ms,
        "onnx_ok": onnx_ok,
        "status": ablation_results.STATUS_OK,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        required=True,
        nargs="+",
        choices=[b.value for b in BackboneType],
        metavar="BACKBONE",
        help="One or more backbones to evaluate in sequence. "
        f"Choices: {[b.value for b in BackboneType]}",
    )
    parser.add_argument(
        "--head",
        default="arcface",
        choices=[h.value for h in HeadType],
        help="Margin head (default: arcface).",
    )
    parser.add_argument(
        "--mode",
        default="probe",
        choices=["probe", "finetune"],
        help="probe = frozen-trunk linear probe (Stage A); "
        "finetune = full two-phase training (Stage B).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[42],
        help="One or more seeds; cells run as seeds x backbones (default: 42).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch budget (probe: total; finetune: warmup epochs).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input size (default: the backbone's native size).",
    )
    parser.add_argument("--tag", default="", help="Free-text note recorded in the row.")
    parser.add_argument(
        "--no-onnx", action="store_true", help="Skip the ONNX-export check."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells already recorded in results.csv instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the cells that would run or be skipped, then exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run: 1 epoch unless --epochs given (for plumbing checks).",
    )
    args = parser.parse_args()

    head = HeadType(args.head)
    epochs = args.epochs or (1 if args.smoke else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbones = [BackboneType(b) for b in args.backbone]
    cells = [(seed, backbone) for seed in args.seed for backbone in backbones]
    if len(cells) > 1:
        print(f"=== Sweep: {len(cells)} cells, mode={args.mode}, device={device} ===")

    recorded = ablation_results.load_rows(RESULTS_CSV)
    counts = {"done": 0, "failed": 0, "skipped": 0, "planned": 0}

    for i, (seed, backbone) in enumerate(cells):
        img_size = args.img_size or get_backbone_input_size(backbone)
        label = (
            f"backbone={backbone.value} head={head.value} mode={args.mode} "
            f"seed={seed} img_size={img_size}"
        )
        prior = ablation_results.find_recorded(
            recorded,
            backbone=backbone.value,
            head=head.value,
            mode=args.mode,
            seed=seed,
            img_size=args.img_size,
            epochs=epochs,
        )
        if prior is not None and not args.force:
            status = ablation_results.row_status(prior)
            verb = "WOULD SKIP" if args.dry_run else "SKIP"
            print(
                f"=== [{i + 1}/{len(cells)}] {verb} {label}: already recorded "
                f"({status}, {prior.get('timestamp')}); --force re-runs it ==="
            )
            counts["skipped"] += 1
            continue

        if args.dry_run:
            print(f"=== [{i + 1}/{len(cells)}] WOULD RUN {label} ===")
            counts["planned"] += 1
            continue

        print(f"=== [{i + 1}/{len(cells)}] Ablation: {label} device={device} ===")
        cell = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "backbone": backbone.value,
            "head": head.value,
            "mode": args.mode,
            "seed": seed,
            "epochs": epochs or "default",
            "img_size": img_size,
            "tag": args.tag,
        }
        try:
            row = run_cell(cell, backbone, head, args, epochs, device)
            counts["done"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad cell must not end the sweep
            traceback.print_exc()
            row = ablation_results.failure_row(cell, exc)
            counts["failed"] += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # an OOM here must not poison the next cell

        ablation_results.append_row(RESULTS_CSV, row)
        recorded.append(row)
        ablation_results.write_markdown(RESULTS_MD, recorded)

        print(f"\n=== Result ({row['status']}) ===")
        for k, v in row.items():
            print(f"  {k}: {v}")
        print(f"\nAppended to {RESULTS_CSV}")

    if args.dry_run:
        print(
            f"\n=== Dry run: {counts['planned']} of {len(cells)} cells would run, "
            f"{counts['skipped']} already recorded. Nothing was trained. ==="
        )
    else:
        print(
            f"\n=== Sweep summary: {counts['done']} run, {counts['failed']} failed, "
            f"{counts['skipped']} skipped ==="
        )


if __name__ == "__main__":
    main()
