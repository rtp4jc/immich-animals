#!/usr/bin/env python3
"""
run_smoke_tests.py - A simple verification test for the exported ONNX model.

Prompt 8: Minimal verification tests & commit

This script loads the exported ONNX model and runs inference on a few sample
images to ensure the model is valid and produces output.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

# Note: onnxruntime-gpu is recommended for GPU inference.
# If you don't have it, this will fall back to CPU.
# You can install it with: pip install onnxruntime-gpu
import onnxruntime as ort

def main():
    """Main smoke test function."""
    print("=" * 60)
    print("Running smoke tests on exported ONNX model...")
    print("=" * 60)

    # 1. Find the ONNX model
    onnx_model_paths = glob.glob("models/phase1/export/*.onnx")
    if not onnx_model_paths:
        print("Error: No .onnx model found in models/phase1/export/")
        print("Please run scripts/phase1/export_model.py first.")
        exit(1)
    # Find the most recent ONNX file
    onnx_model_path = max(onnx_model_paths, key=os.path.getctime)
    print(f"Found ONNX model: {onnx_model_path}")

    # 2. Find sample images
    sample_dir = Path("outputs/phase1/sample_images")
    sample_images = list(sample_dir.glob("*.jpg"))[:3]
    if len(sample_images) < 3:
        print(f"Error: Not enough sample images found in {sample_dir}. Need at least 3.")
        print("Please run scripts/phase1/create_sample_images.py first.")
        exit(1)
    print(f"Found {len(sample_images)} sample images to test.")

    # 3. Load ONNX model and create inference session
    try:
        session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print("Please ensure 'onnxruntime' or 'onnxruntime-gpu' is installed.")
        exit(1)

    # 4. Get model input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    _, _, input_height, input_width = input_shape
    print(f"Model expects input shape: {input_shape}")

    output_dir = Path("outputs/phase1/inference_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving test inference results to: {output_dir.resolve()}")

    # 5. Run inference on sample images
    for image_path in sample_images:
        print(f"  - Processing {image_path.name}...")
        
        # Preprocess image
        img = cv2.imread(str(image_path))
        img_resized = cv2.resize(img, (input_width, input_height))
        img_transposed = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
        img_normalized = img_transposed.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})[0]
        
        # For a smoke test, we just confirm the output exists and has a reasonable shape.
        # A full post-processing of the output is complex and not needed here.
        print(f"    Output shape: {outputs.shape}")
        assert len(outputs.shape) == 3, f"Output shape {outputs.shape} is not 3-dimensional as expected."
        assert outputs.shape[0] == 1, f"Batch size in output is not 1."

        # Save a simple visualized output to confirm success
        cv2.putText(img, "SMOKE TEST PASSED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        output_image_path = output_dir / image_path.name
        cv2.imwrite(str(output_image_path), img)

    print("\nSmoke tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
