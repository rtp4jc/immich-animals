"""POST images to a running sidecar exactly as Immich does, and check the reply.

Usage: python sidecar/smoke_test.py IMAGE [IMAGE ...] [--url http://localhost:3003]
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

ENTRIES = {
    "facial-recognition": {
        "detection": {"modelName": "buffalo_l", "options": {"minScore": 0.7}},
        "recognition": {"modelName": "buffalo_l"},
    }
}


def check(url: str, path: Path) -> list[dict]:
    response = httpx.post(
        f"{url}/predict",
        data={"entries": json.dumps(ENTRIES)},
        files={"image": path.read_bytes()},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()

    assert set(body) == {"facial-recognition", "imageHeight", "imageWidth"}, body.keys()
    for face in body["facial-recognition"]:
        assert set(face) == {"boundingBox", "embedding", "score"}, face.keys()
        assert set(face["boundingBox"]) == {"x1", "y1", "x2", "y2"}
        # Immich parses this string into a pgvector, so it must be a string.
        assert isinstance(face["embedding"], str), type(face["embedding"])
        assert isinstance(json.loads(face["embedding"])[0], float)
    return body["facial-recognition"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default="http://localhost:3003")
    args = parser.parse_args()

    assert httpx.get(f"{args.url}/ping").text == "pong"

    total = 0
    for path in args.images:
        faces = check(args.url, path)
        total += len(faces)
        scores = ", ".join(f"{f['score']:.2f}" for f in faces)
        print(f"{path.name}: {len(faces)} face(s) [{scores}]")

    print(f"\nOK — {len(args.images)} images, {total} faces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
