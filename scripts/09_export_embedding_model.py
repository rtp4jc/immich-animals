import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dog_id.common.constants import MODELS_DIR, ONNX_EMBEDDING_PATH
from dog_id.embedding.config import DATA_CONFIG, DEFAULT_BACKBONE, TRAINING_CONFIG
from dog_id.embedding.models import EmbeddingNet

def main(args):
    """
    Loads the trained PyTorch model and exports it to the ONNX format.
    """
    print("--- Embedding Model ONNX Exporter ---")
    device = torch.device("cpu")  # ONNX export should be done on CPU

    model_path = MODELS_DIR / "dog_embedding_best.pt"
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}. Please run training first.")
        sys.exit(1)

    # Instantiate the model with the same architecture used for training
    backbone_name = args.backbone
    print(f"Instantiating model with backbone: {backbone_name}")
    model = EmbeddingNet(
        backbone_name=backbone_name,
        embedding_dim=TRAINING_CONFIG["EMBEDDING_DIM"],
        pretrained=False,  # Weights are loaded next, no need to download
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    dummy_input = torch.randn(
        1, 3, DATA_CONFIG["IMG_SIZE"], DATA_CONFIG["IMG_SIZE"], device=device
    )
    
    # Ensure output directory exists
    ONNX_EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {ONNX_EMBEDDING_PATH}...")
    try:
        torch.onnx.export(
            model,
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
    parser = argparse.ArgumentParser(description="Export a trained embedding model to ONNX.")
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        help=f"Backbone of the trained model to export. Default: {DEFAULT_BACKBONE}",
    )
    args = parser.parse_args()
    main(args)
