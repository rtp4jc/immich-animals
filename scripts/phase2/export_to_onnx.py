"""
Exports the trained PyTorch embedding model to the ONNX format.

What it's for:
This script is the final step in the model production pipeline. It converts the
model from PyTorch's internal format (`.pt`) to ONNX (`.onnx`), a standardized,
interoperable format for machine learning models.

What it does:
1. Loads the best trained model weights from `models/dog_embedding_best.pt`.
2. Instantiates the `EmbeddingNet` model architecture.
3. Creates a dummy input tensor with the correct shape (`1, 3, 224, 224`).
4. Calls `torch.onnx.export` to trace the model with the dummy input and save the
   resulting computational graph and weights as an ONNX file.
5. The export is configured with dynamic axes, allowing the final model to process
   variable-sized batches of images.

How to run it:
- This script should be run after `training/train_embedding.py` has produced a model.
- Run from the root of the project:
  `python scripts/phase2/export_to_onnx.py`

How to interpret the results:
The script will print its progress. A successful run will create a new file:
- `models/dog_embedding.onnx`
This `.onnx` file can then be used with various ONNX-compatible runtimes (like ONNX
Runtime, TensorRT, etc.) for deployment in different applications.
"""
import torch
import os
import sys

# Adjust the path to where your training module is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.phase2.embedding_model import EmbeddingNet

# --- Configuration ---
MODEL_PATH = 'models/dog_embedding_best.pt'
ONNX_OUTPUT_PATH = 'models/dog_embedding.onnx'
IMG_SIZE = 224
EMBEDDING_DIM = 512

def main():
    """
    Loads the trained PyTorch model and exports it to the ONNX format.
    """
    print(f"--- Exporting model to ONNX format ---")
    device = torch.device("cpu") # ONNX export should be done on CPU

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run training first.")
        return

    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    
    model.to(device)
    model.eval() # Set to evaluation mode

    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

    print(f"Exporting to {ONNX_OUTPUT_PATH}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=11, # A commonly used version
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'}, # Allow for variable batch size
                'output': {0: 'batch_size'}
            }
        )
        print("Export successful!")
        print(f"Model saved to {ONNX_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == '__main__':
    main()