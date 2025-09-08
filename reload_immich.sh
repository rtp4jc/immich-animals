#!/bin/bash
# Reload immich container with latest code and models

set -e

echo "Reloading Immich ML container..."

cd immich-clone

# Stop and remove existing container
echo "Stopping existing container..."
docker stop immich-ml-dogs 2>/dev/null || true
docker rm immich-ml-dogs 2>/dev/null || true

# Rebuild container
echo "Building new container..."
docker build -f machine-learning/Dockerfile.dogs -t immich-ml-dogs machine-learning/

# Start new container
echo "Starting new container..."
docker run -d --name immich-ml-dogs -p 3003:3003 immich-ml-dogs

# Wait for container to be ready
echo "Waiting for container to be ready..."
sleep 5

# Test health
echo "Testing container health..."
curl -f http://localhost:3003/ping || echo "Warning: Health check failed"

echo "Container reloaded successfully!"
