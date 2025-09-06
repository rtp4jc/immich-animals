"""
Exports a trained PyTorch embedding model to the ONNX format.

What it's for:
This script is the final step in the model production pipeline. It converts a model
from PyTorch's internal format (`.pt`) to ONNX (`.onnx`), a standardized,
interoperable format for machine learning models.

What it does:
1. Instantiates the `EmbeddingNet` model architecture, specifying a backbone.
2. Loads the best trained model weights from the path specified in the config.
3. Creates a dummy input tensor with the correct shape.
4. Calls `torch.onnx.export` to trace the model and save the resulting computational
   graph and weights as an ONNX file.
5. The export is configured with dynamic axes, allowing the final model to process
   variable-sized batches of images.

How to run it:
- This script should be run after a model has been trained.
- The backbone argument should match the one used for training.
- Example:
  `python scripts/08_export_embedding_model.py --backbone efficientnet_b0`

How to interpret the results:
The script will print its progress. A successful run will create a new file:
- `models/dog_embedding.onnx`
This `.onnx` file can then be used with various ONNX-compatible runtimes.
"""
import torch
import os
import sys
import argparse

# Adjust path to import from our new package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dog_id.embedding.models import EmbeddingNet
from dog_id.embedding.config import TRAINING_CONFIG, DATA_CONFIG, DEFAULT_BACKBONE

# --- Configuration ---
ONNX_OUTPUT_PATH = 'models/dog_embedding.onnx'

def main(args):
    """
    Loads the trained PyTorch model and exports it to the ONNX format.
    """
    print(f"--- Exporting model to ONNX format ---")
    device = torch.device("cpu") # ONNX export should be done on CPU

    model_path = TRAINING_CONFIG['MODEL_OUTPUT_PATH']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return

    # Instantiate the model with the same architecture used for training
    print(f"Instantiating model with backbone: {args.backbone}")
    model = EmbeddingNet(
        backbone_name=args.backbone,
        embedding_dim=TRAINING_CONFIG['EMBEDDING_DIM'],
        pretrained=False # Weights are loaded next, no need to download
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the --backbone argument matches the one used for training.")
        return
    
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, DATA_CONFIG['IMG_SIZE'], DATA_CONFIG['IMG_SIZE'], device=device)
    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

    print(f"Exporting to {ONNX_OUTPUT_PATH}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("Export successful!")
        print(f"Model saved to {ONNX_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export a trained embedding model to ONNX.")
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, 
                        help=f"Backbone of the trained model to export. Default: {DEFAULT_BACKBONE}")
    args = parser.parse_args()
    main(args)
