#!/usr/bin/env python
"""
Visualizes the Immich-integrated dog identification pipeline with similarity matching.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

def get_embedding(image_path, host="localhost", port=3003):
    """Get embedding for a single image."""
    entries = {
        "dog-identification": {
            "detection": {"modelName": "dog_detector"},
            "keypoint": {"modelName": "dog_keypoint"},
            "recognition": {"modelName": "dog_embedder"},
        }
    }

    url = f"http://{host}:{port}/predict"
    files = {"image": (image_path.name, open(image_path, "rb"), "image/jpeg")}
    data = {"entries": json.dumps(entries)}

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        results = response.json()
        embeddings = results.get("dog-identification", [])
        return np.array(embeddings[0]) if embeddings else None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for {image_path}: {e}")
        return None

def find_similar_images(query_embedding, gallery_embeddings, gallery_paths, top_k=5):
    """Find most similar images using cosine similarity."""
    similarities = cosine_similarity([query_embedding], gallery_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'path': gallery_paths[idx],
            'similarity': similarities[idx],
            'embedding': gallery_embeddings[idx]
        })
    return results

def create_results_grid(query_results):
    """Create a grid showing all queries and their matches."""
    n_queries = len(query_results)
    n_matches = 5
    
    fig, axes = plt.subplots(n_queries, n_matches + 1, figsize=(18, 4 * n_queries))
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for i, (query_path, matches) in enumerate(query_results):
        # Query image
        query_img = cv2.imread(str(query_path))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title(f"Query:\n{query_path.name}")
        axes[i, 0].axis('off')
        
        # Matches
        for j, match in enumerate(matches):
            match_img = cv2.imread(str(match['path']))
            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            axes[i, j+1].imshow(match_img)
            axes[i, j+1].set_title(f"Sim: {match['similarity']:.3f}\n{match['path'].name}")
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize Immich dog identification pipeline with similarity matching")
    parser.add_argument("--queries", nargs="*", help="Query image paths (default: first 3 from validation set)")
    parser.add_argument("--host", default="localhost", help="Immich ML host")
    parser.add_argument("--port", type=int, default=3003, help="Immich ML port")
    parser.add_argument("--save", help="Save visualizations to directory")
    parser.add_argument("--gallery-size", type=int, default=100, help="Number of gallery images to use")
    args = parser.parse_args()

    # Load validation data
    val_file = Path("data/identity_val.json")
    if not val_file.exists():
        print(f"[ERROR] Validation file not found: {val_file}")
        sys.exit(1)
    
    with open(val_file) as f:
        val_data = json.load(f)

    # Get gallery images
    gallery_paths = [Path(item["file_path"]) for item in val_data[:args.gallery_size]]
    print(f"Building gallery with {len(gallery_paths)} images...")

    # Generate gallery embeddings
    gallery_embeddings = []
    valid_gallery_paths = []
    
    for path in tqdm(gallery_paths, desc="Building gallery"):
        if not path.exists():
            continue
            
        embedding = get_embedding(path, args.host, args.port)
        if embedding is not None:
            gallery_embeddings.append(embedding)
            valid_gallery_paths.append(path)
    
    if not gallery_embeddings:
        print("[ERROR] No valid gallery embeddings generated")
        sys.exit(1)
    
    gallery_embeddings = np.array(gallery_embeddings)
    print(f"Gallery ready: {len(gallery_embeddings)} embeddings")

    # Get query images
    if args.queries:
        query_paths = [Path(p) for p in args.queries]
    else:
        # Use 10 random images from validation set
        random.seed(42)  # For reproducible results
        query_paths = [Path(item["file_path"]) for item in random.sample(val_data, 10)]
        print(f"Selected 10 random query images")

    # Collect all query results
    query_results = []
    failed_detections = []

    # Process each query
    for query_path in tqdm(query_paths, desc="Processing queries"):
        if not query_path.exists():
            print(f"[ERROR] Query image not found: {query_path}")
            continue
        
        # Get query embedding
        query_embedding = get_embedding(query_path, args.host, args.port)
        if query_embedding is None:
            failed_detections.append(query_path.name)
            continue

        # Find similar images
        matches = find_similar_images(query_embedding, gallery_embeddings, valid_gallery_paths)
        query_results.append((query_path, matches))

    # Report failed detections
    if failed_detections:
        print(f"\nImages with no dog detection: {', '.join(failed_detections)}")

    # Create and save single grid visualization
    if query_results:
        fig = create_results_grid(query_results)
        
        if args.save:
            save_dir = Path(args.save)
            save_dir.mkdir(exist_ok=True)
            output_path = save_dir / "immich_pipeline_results.png"
        else:
            output_path = Path("outputs/immich_pipeline_results.png")
            output_path.parent.mkdir(exist_ok=True)
            
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nResults saved to: {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()
