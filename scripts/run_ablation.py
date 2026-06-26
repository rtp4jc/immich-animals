"""Run a single embedding-backbone ablation cell and record its results.

This is the harness for the backbone ablation described in
``.planning/6-25-2026-embedding-backbone-ablation/plan.md``. One invocation =
one (backbone, head, mode, seed) cell: it trains the embedding model, evaluates
it on the **held-out test split**, measures cost (params, CPU latency, ONNX
exportability), and appends a row to a shared results table.

Two modes:
- ``probe``    — linear probe: freeze the backbone trunk, train only the
                 projection + margin head. Fair, cheap feature-quality ranking
                 (no per-backbone LR to tune). Used for the broad Stage A sweep.
- ``finetune`` — full two-phase training (warmup head, then unfreeze + fine-tune
                 with differential LR). Used for the Stage B finalists.

Examples (run from the repo root, where ``data/`` lives):

    # Stage A: linear-probe a candidate
    uv run python scripts/run_ablation.py --backbone convnextv2_tiny --mode probe

    # Stage B: full fine-tune of a finalist, one of several seeds
    uv run python scripts/run_ablation.py --backbone convnextv2_tiny \
        --mode finetune --seed 1

    # Quick smoke test (1 epoch, subsampled) to confirm the cell runs
    uv run python scripts/run_ablation.py --backbone resnet50 --epochs 1 --smoke

Results are appended to ``outputs/ablation/results.csv`` (+ a regenerated
``results.md`` table). The script never overwrites prior rows, so it is safe to
run repeatedly and resume an interrupted sweep.
"""

import argparse
import copy
import csv
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from animal_id.benchmark.metrics import evaluate_embedding_model
from animal_id.common.constants import DATA_DIR
from animal_id.common.datasets import IdentityDataset
from animal_id.common.seed import set_seed, worker_init_fn
from animal_id.embedding.backbones import BackboneType, get_backbone_input_size
from animal_id.embedding.config import DATA_CONFIG, TRAINING_CONFIG
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.losses import HeadType
from animal_id.embedding.models import AnimalEmbeddingModel
from animal_id.embedding.trainer import EmbeddingTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
RESULTS_MD = OUTPUT_DIR / "results.md"

CSV_FIELDS = [
    "timestamp",
    "backbone",
    "head",
    "mode",
    "seed",
    "epochs",
    "img_size",
    "n_test_queries",
    "mrr",
    "top1",
    "top5",
    "mAP",
    "tar@1%",
    "tar@0.1%",
    "params_total_M",
    "params_trainable_M",
    "cpu_ms",
    "onnx_ok",
    "tag",
]


def retrieval_metrics(embeddings: np.ndarray, labels: np.ndarray, ks=(1, 5)):
    """Leave-one-out cosine retrieval over the gallery (= the test embeddings).

    Returns (mrr, {k: top_k_accuracy}, n_evaluated_queries). Queries with no
    same-identity gallery item are skipped (standard open-set practice).
    """
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)
    ranked_labels = labels[order]
    matches = ranked_labels == labels[:, None]

    has_pos = matches.any(axis=1)
    matches = matches[has_pos]
    if matches.shape[0] == 0:
        return 0.0, {k: 0.0 for k in ks}, 0

    first_rank = matches.argmax(axis=1) + 1
    mrr = float(np.mean(1.0 / first_rank))
    topk = {k: float(np.mean(matches[:, :k].any(axis=1))) for k in ks}
    return mrr, topk, int(has_pos.sum())


def measure_cpu_latency(model, img_size, iters=20):
    """Mean single-image CPU inference latency (ms) for the embedding path."""
    lat_model = copy.deepcopy(model).to("cpu").eval()
    x = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(3):  # warmup
            lat_model.get_embeddings(x)
        start = time.perf_counter()
        for _ in range(iters):
            lat_model.get_embeddings(x)
        elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def check_onnx_export(model, img_size):
    """Try exporting to ONNX; return True if it succeeds, else False."""
    export_model = copy.deepcopy(model).to("cpu").eval()
    tmp_path = PROJECT_ROOT / "outputs" / "ablation_onnx_check.onnx"
    try:
        export_embedding_onnx(export_model, tmp_path, img_size=img_size)
        return True
    except Exception as exc:  # noqa: BLE001 - we want any export failure recorded
        print(f"[onnx] export failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_results_row(row: dict):
    """Append one row to results.csv and regenerate the markdown table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    _regenerate_markdown()


def _regenerate_markdown():
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    # Sort: finetune before probe, then by MRR descending.
    rows.sort(key=lambda r: (r["mode"] != "finetune", -float(r["mrr"] or 0)))
    lines = [
        "# Ablation results",
        "",
        "_Auto-generated by `scripts/run_ablation.py`. Do not edit by hand._",
        "",
        "| backbone | head | mode | seed | MRR | Top-1 | Top-5 | mAP | "
        "TAR@1% | params(M) | CPU ms | ONNX | tag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {backbone} | {head} | {mode} | {seed} | {mrr} | {top1} | {top5} "
            "| {mAP} | {tar1} | {params} | {cpu} | {onnx} | {tag} |".format(
                backbone=r["backbone"],
                head=r["head"],
                mode=r["mode"],
                seed=r["seed"],
                mrr=_fmt(r["mrr"]),
                top1=_fmt(r["top1"]),
                top5=_fmt(r["top5"]),
                mAP=_fmt(r["mAP"]),
                tar1=_fmt(r["tar@1%"]),
                params=r["params_total_M"],
                cpu=r["cpu_ms"],
                onnx="✅" if r["onnx_ok"] == "True" else "❌",
                tag=r["tag"],
            )
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def _fmt(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


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
        num_workers=TRAINING_CONFIG["HARDWARE_WORKERS"],
        generator=generator if is_training else None,
        worker_init_fn=worker_init_fn if is_training else None,
    )
    return dataset, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        required=True,
        choices=[b.value for b in BackboneType],
        help="Backbone to evaluate.",
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
    parser.add_argument("--seed", type=int, default=42)
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
        "--smoke",
        action="store_true",
        help="Quick run: 1 epoch unless --epochs given (for plumbing checks).",
    )
    args = parser.parse_args()

    backbone = BackboneType(args.backbone)
    head = HeadType(args.head)
    img_size = args.img_size or get_backbone_input_size(backbone)
    batch_size = DATA_CONFIG["BATCH_SIZE"]
    epochs = args.epochs or (1 if args.smoke else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"=== Ablation: backbone={backbone.value} head={head.value} "
        f"mode={args.mode} seed={args.seed} img_size={img_size} device={device} ==="
    )

    g = set_seed(args.seed)

    train_dataset, train_loader = build_loader(
        DATA_CONFIG["TRAIN_JSON_PATH"].split("/")[-1],
        img_size,
        batch_size,
        is_training=True,
        generator=g,
    )
    _, val_loader = build_loader(
        DATA_CONFIG["VAL_JSON_PATH"].split("/")[-1],
        img_size,
        batch_size,
        is_training=False,
    )
    test_dataset, test_loader = build_loader(
        "identity_test.json", img_size, batch_size, is_training=False
    )

    model = AnimalEmbeddingModel(
        backbone_type=backbone,
        num_classes=train_dataset.num_classes,
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
        head_type=head,
    ).to(device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        PROJECT_ROOT / "runs" / f"{timestamp}_{backbone.value}_{args.mode}_s{args.seed}"
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
        probe_epochs = epochs or TRAINING_CONFIG["FULL_TRAIN_EPOCHS"]
        best_model_path = trainer.train(
            warmup_epochs=probe_epochs,
            full_epochs=0,
            head_lr=TRAINING_CONFIG["HEAD_LR"],
            backbone_lr=TRAINING_CONFIG["BACKBONE_LR"],
            full_lr=TRAINING_CONFIG["FULL_TRAIN_LR"],
            patience=TRAINING_CONFIG["EARLY_STOPPING_PATIENCE"],
            linear_probe=True,
        )
    else:
        best_model_path = trainer.train(
            warmup_epochs=epochs or TRAINING_CONFIG["WARMUP_EPOCHS"],
            full_epochs=TRAINING_CONFIG["FULL_TRAIN_EPOCHS"],
            head_lr=TRAINING_CONFIG["HEAD_LR"],
            backbone_lr=TRAINING_CONFIG["BACKBONE_LR"],
            full_lr=TRAINING_CONFIG["FULL_TRAIN_LR"],
            patience=TRAINING_CONFIG["EARLY_STOPPING_PATIENCE"],
            linear_probe=False,
        )

    # --- Evaluate on the held-out TEST split ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device).eval()

    test_metrics = evaluate_embedding_model(model, test_loader, device)

    # Leave-one-out Top-k / MRR from the raw test embeddings.
    embeddings, labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            embeddings.append(model.get_embeddings(images.to(device)).cpu().numpy())
            labels.extend(lbls.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    mrr, topk, n_queries = retrieval_metrics(embeddings, labels)

    # --- Cost axis ---
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    cpu_ms = measure_cpu_latency(model, img_size)
    onnx_ok = False if args.no_onnx else check_onnx_export(model, img_size)

    row = {
        "timestamp": timestamp,
        "backbone": backbone.value,
        "head": head.value,
        "mode": args.mode,
        "seed": args.seed,
        "epochs": epochs or "default",
        "img_size": img_size,
        "n_test_queries": n_queries,
        "mrr": round(mrr, 4),
        "top1": round(topk[1], 4),
        "top5": round(topk[5], 4),
        "mAP": round(test_metrics.get("mAP", 0.0), 4),
        "tar@1%": round(test_metrics.get("TAR@FAR=1%", 0.0), 4),
        "tar@0.1%": round(test_metrics.get("TAR@FAR=0.1%", 0.0), 4),
        "params_total_M": round(total_params, 2),
        "params_trainable_M": round(trainable_params, 2),
        "cpu_ms": round(cpu_ms, 1),
        "onnx_ok": onnx_ok if not args.no_onnx else "skipped",
        "tag": args.tag,
    }
    write_results_row(row)

    print("\n=== Result ===")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print(f"\nAppended to {RESULTS_CSV}")
    print(f"Run artifacts in {run_dir}")


if __name__ == "__main__":
    main()
