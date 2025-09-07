#!/usr/bin/env python
"""
Inspects the input and output shapes of one or more ONNX models.

This script is a diagnostic tool to confirm the exact I/O specifications of
ONNX models, ensuring that pre-processing and post-processing logic is based
on correct information, not assumptions.
"""

import argparse
import sys
from pathlib import Path

import onnxruntime as ort

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

def inspect_model(name: str, path: Path):
    """Loads a model and prints its input and output details."""
    print(f"--- Inspecting: {name} ---")
    if not path.exists():
        print(f"[ERROR] Model not found at: {path}")
        return

    try:
        session = ort.InferenceSession(str(path))
        
        print("Inputs:")
        for i, input_meta in enumerate(session.get_inputs()):
            print(f"  [{i}] Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
            
        print("\nOutputs:")
        for i, output_meta in enumerate(session.get_outputs()):
            print(f"  [{i}] Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")

    except Exception as e:
        print(f"[ERROR] Could not inspect model: {e}")
    print("-" * (len(name) + 16))
    print()

def main(args):
    """Main function to inspect all provided models."""
    for model_path_str in args.model_paths:
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"[ERROR] Provided model path does not exist: {model_path}")
            continue
        inspect_model(model_path.name, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the input/output specs of ONNX models.")
    parser.add_argument("model_paths", nargs='+', help="One or more paths to ONNX model files.")
    args = parser.parse_args()
    main(args)
