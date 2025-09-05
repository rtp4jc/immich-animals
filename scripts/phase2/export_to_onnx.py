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
