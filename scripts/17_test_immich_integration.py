#!/usr/bin/env python
"""
Tests the custom dog identification pipeline running in the Immich ML container.
"""

import argparse
import json
import sys
from pathlib import Path

import requests

def main(args):
    """Main test function."""
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found at: {image_path}")
        sys.exit(1)

    # This JSON payload tells the Immich ML service which models to run in which order.
    # We use our custom "dog-identification" task to chain our three models.
    if args.skip_keypoints:
        entries = {
            "dog-identification": {
                "detection": {"modelName": "dog_detector"},
                "recognition": {"modelName": "dog_embedder_direct"},
            }
        }
    else:
        entries = {
            "dog-identification": {
                "detection": {"modelName": "dog_detector"},
                "keypoint": {"modelName": "dog_keypoint"},
                "recognition": {"modelName": "dog_embedder"},
            }
        }

    url = f"http://{args.host}:{args.port}/predict"
    files = {
        "image": (image_path.name, open(image_path, "rb"), "image/jpeg")
    }
    data = {
        "entries": json.dumps(entries)
    }

    print(f"Sending request to {url} with image: {image_path.name}")
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status() # Raise an exception for bad status codes
        
        print("\n--- Response --- ")
        print(f"Status Code: {response.status_code}")
        # Pretty-print JSON response
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Immich dog identification pipeline.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--host", type=str, default="localhost", help="Hostname of the Immich ML container.")
    parser.add_argument("--port", type=int, default=3003, help="Port of the Immich ML container.")
    parser.add_argument("--skip-keypoints", action="store_true", help="Skip keypoint stage in pipeline")
    args = parser.parse_args()
    main(args)