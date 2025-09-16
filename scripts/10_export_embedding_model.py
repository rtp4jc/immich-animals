import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from animal_id.common.constants import MODELS_DIR, ONNX_EMBEDDING_PATH
from animal_id.common.utils import find_latest_timestamped_run
from animal_id.embedding.config import DATA_CONFIG, DEFAULT_BACKBONE, TRAINING_CONFIG
from animal_id.embedding.models import DogEmbeddingModel
from animal_id.embedding.backbones import BackboneType


def main(args):
    """
    Loads the trained PyTorch model and exports it to the ONNX format.
    """
    print("--- Embedding Model ONNX Exporter ---")
    device = torch.device("cpu")  # ONNX export should be done on CPU

    # Try to find model in latest run directory first
    latest_run = find_latest_timestamped_run()
    model_path = None

    if latest_run:
        model_path = latest_run / "best_model.pt"
        if not model_path.exists():
            model_path = None

    # Fall back to old location if not found
    if model_path is None:
        model_path = MODELS_DIR / "dog_embedding_best.pt"
        if not model_path.exists():
            print(f"[ERROR] No trained model found.")
            print(
                f"Checked: runs/*/best_model.pt and {MODELS_DIR / 'dog_embedding_best.pt'}"
            )
            print("Please run training first.")
            sys.exit(1)

    print(f"Using model: {model_path}")

    # Instantiate the model with the same architecture used for training
    print(f"Instantiating model with backbone: {args.backbone.value}")
    model = DogEmbeddingModel(
        backbone_type=args.backbone,
        num_classes=1001,  # This doesn't matter for ONNX export (only embeddings)
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Create a wrapper that only outputs embeddings for ONNX export
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.get_embeddings(x)

    export_model = EmbeddingWrapper(model)

    dummy_input = torch.randn(
        1, 3, DATA_CONFIG["IMG_SIZE"], DATA_CONFIG["IMG_SIZE"], device=device
    )

    # Ensure output directory exists
    ONNX_EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {ONNX_EMBEDDING_PATH}...")
    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            str(ONNX_EMBEDDING_PATH),
            export_params=True,
            opset_version=12,  # Use a slightly more modern opset
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print("\nExport successful!")
    except Exception as e:
        print(f"\n[ERROR] Error during ONNX export: {e}")
        sys.exit(1)

    # --- Final Verification ---
    if ONNX_EMBEDDING_PATH.exists():
        print(f"Verified: ONNX model created at {ONNX_EMBEDDING_PATH}")
        print(f"File size: {ONNX_EMBEDDING_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("\n[ERROR] Export process completed, but the output file was not found.")
        sys.exit(1)

    print("\n--- Exporter finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained embedding model to ONNX."
    )
    parser.add_argument(
        "--backbone",
        type=BackboneType,
        default=DEFAULT_BACKBONE,
        choices=list(BackboneType),
        help=f"Backbone of the trained model to export. Default: {DEFAULT_BACKBONE.value}",
    )
    args = parser.parse_args()
    main(args)
