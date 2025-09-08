#!/bin/bash
# Copy ONNX models to immich-clone directory

set -e

echo "Copying ONNX models to immich-clone..."

# Create model directories
mkdir -p immich-clone/machine-learning/models/onnx/dog-detector
mkdir -p immich-clone/machine-learning/models/onnx/dog-keypoint  
mkdir -p immich-clone/machine-learning/models/onnx/dog-embedder

# Copy models with correct names
cp models/onnx/detector.onnx immich-clone/machine-learning/models/onnx/dog-detector/model.onnx
cp models/onnx/keypoint.onnx immich-clone/machine-learning/models/onnx/dog-keypoint/model.onnx
cp models/onnx/embedding.onnx immich-clone/machine-learning/models/onnx/dog-embedder/model.onnx

echo "Models copied successfully!"
