#!/bin/bash
# Usage: ./copy_models.sh <path_to_immich_app>

IMMICH_PATH=${1:-"../immich-app"}

if [ ! -d "$IMMICH_PATH" ]; then
    echo "Error: Immich path '$IMMICH_PATH' does not exist."
    echo "Usage: ./copy_models.sh <path_to_immich_app>"
    exit 1
fi

TARGET_DIR="$IMMICH_PATH/machine-learning/models/onnx"

echo "Copying models to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

if [ ! -f "models/onnx/detector.onnx" ] || [ ! -f "models/onnx/embedding.onnx" ]; then
    echo "Error: Source models not found in models/onnx/"
    exit 1
fi

cp models/onnx/detector.onnx "$TARGET_DIR/detector.onnx"
cp models/onnx/embedding.onnx "$TARGET_DIR/embedding.onnx"

echo "Done! You can now run docker compose in $IMMICH_PATH/docker"
