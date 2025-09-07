import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dog_id.common.constants import (
    DATA_DIR,
    ONNX_DETECTOR_PATH,
    ONNX_EMBEDDING_PATH,
    ONNX_KEYPOINT_PATH,
)

# --- Visualization Helpers ---

def create_text_image(text: str, width: int, height: int) -> np.ndarray:
    """Creates an image with text for labeling."""
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    # Shorten filename if too long
    if len(text) > 30:
        text = "..." + text[-27:]
    cv2.putText(img, text, (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def resize_with_aspect_ratio(image: np.ndarray, height: int) -> np.ndarray:
    """Resizes an image to a specific height while maintaining aspect ratio."""
    if image is None:
        return np.full((height, height, 3), (0, 0, 255), dtype=np.uint8) # Return red square on error
    h, w = image.shape[:2]
    if h == 0:
        return np.full((height, height, 3), (0, 0, 255), dtype=np.uint8)
    ratio = height / h
    new_w = int(w * ratio)
    return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)

# --- Pre-processing and Post-processing Helpers ---

def preprocess_detector(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Prepares an image for the YOLO detector model, returning the resized image and original shape."""
    original_shape = image.shape[:2]
    # YOLOv8 and v11 models exported with NMS expect a resize, not letterbox padding.
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    
    img_fp = resized.astype(np.float32) / 255.0
    img_fp = np.transpose(img_fp, (2, 0, 1))
    return np.expand_dims(img_fp, axis=0), original_shape

def preprocess_embedding(image: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Prepares an image for the embedding model."""
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    img_fp = resized.astype(np.float32) / 255.0
    img_fp = np.transpose(img_fp, (2, 0, 1))
    return np.expand_dims(img_fp, axis=0)


class Pipeline:
    """Encapsulates the full inference pipeline."""

    def __init__(self):
        print("Initializing inference pipeline...")
        self.detector = ort.InferenceSession(str(ONNX_DETECTOR_PATH))
        self.keypoint = ort.InferenceSession(str(ONNX_KEYPOINT_PATH))
        self.embedder = ort.InferenceSession(str(ONNX_EMBEDDING_PATH))
        self.detector_input_size = self.detector.get_inputs()[0].shape[2:]
        self.keypoint_input_size = self.keypoint.get_inputs()[0].shape[2:]
        self.embedding_input_size = self.embedder.get_inputs()[0].shape[2:]
        print("ONNX models loaded.")

    def run(self, image_path: Path) -> list[np.ndarray]:
        """Runs the full pipeline on a single image and returns a list of embeddings."""
        try:
            image = cv2.imread(str(image_path))
            if image is None: return []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            return []

        # --- Stage 1: Detection ---
        detector_input, original_shape = preprocess_detector(image, self.detector_input_size)
        detections = self.detector.run(None, {self.detector.get_inputs()[0].name: detector_input})[0][0]

        if len(detections) == 0:
            return []

        embeddings = []
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            if conf < 0.5: continue

            # Scale box to original image size
            h, w = original_shape
            x1, x2 = int(x1 * w / self.detector_input_size[1]), int(x2 * w / self.detector_input_size[1])
            y1, y2 = int(y1 * h / self.detector_input_size[0]), int(y2 * h / self.detector_input_size[0])

            padding = int((x2 - x1) * 0.1)
            crop_x1, crop_y1 = max(0, x1 - padding), max(0, y1 - padding)
            crop_x2, crop_y2 = min(w, x2 + padding), min(h, y2 + padding)
            detector_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

            if detector_crop.size == 0: continue

            # --- Stage 2: Keypoint Estimation ---
            keypoint_input, crop_shape = preprocess_detector(detector_crop, self.keypoint_input_size)
            keypoint_dets = self.keypoint.run(None, {self.keypoint.get_inputs()[0].name: keypoint_input})[0][0]

            final_crop = detector_crop
            if len(keypoint_dets) > 0:
                best_kp_det = keypoint_dets[0] # Assume highest score is first
                keypoints = best_kp_det[6:].reshape((4, 3))
                kp_conf = keypoints[:, 2]

                if np.all(kp_conf > 0.3):
                    # Scale keypoints to detector_crop size
                    ch, cw = crop_shape
                    keypoints[:, 0] = keypoints[:, 0] * cw / self.keypoint_input_size[1]
                    keypoints[:, 1] = keypoints[:, 1] * ch / self.keypoint_input_size[0]

                    kx1, ky1 = np.min(keypoints[:, :2], axis=0).astype(int)
                    kx2, ky2 = np.max(keypoints[:, :2], axis=0).astype(int)
                    kp_padding = int((kx2 - kx1) * 0.2)
                    final_crop_x1, final_crop_y1 = max(0, kx1 - kp_padding), max(0, ky1 - kp_padding)
                    final_crop_x2, final_crop_y2 = min(cw, kx2 + kp_padding), min(ch, ky2 + kp_padding)
                    final_crop = detector_crop[final_crop_y1:final_crop_y2, final_crop_x1:final_crop_x2]

            if final_crop.size == 0: continue

            # --- Stage 3: Embedding ---
            embedding_input = preprocess_embedding(final_crop, self.embedding_input_size)
            embedding = self.embedder.run(None, {self.embedder.get_inputs()[0].name: embedding_input})[0][0]
            embeddings.append(embedding)

        return embeddings

def create_results_visualization(results: dict, similarity_matrix: np.ndarray, output_path: Path, num_queries: int = 10):
    """Creates a single image visualizing the similarity results."""
    print(f"\nCreating visualization at: {output_path}")
    filenames = list(results.keys())
    all_paths = [item['path'] for item in results.values()]
    all_dog_ids = [item['dog_id'] for item in results.values()]
    num_queries = min(num_queries, len(filenames))
    
    # Select queries evenly spaced across the dataset
    query_indices = np.linspace(0, len(filenames) - 1, num_queries, dtype=int)
    
    output_rows = []
    vis_height = 224
    label_height = 30

    for i, query_idx in enumerate(tqdm(query_indices, desc="Creating Visualization")):
        query_filename = filenames[query_idx]
        query_path = all_paths[query_idx]
        query_dog_id = all_dog_ids[query_idx]

        similarities = similarity_matrix[query_idx]
        top_indices = np.argsort(similarities)[::-1][1:6]

        # --- Create Query Image ---
        query_img = cv2.imread(str(query_path))
        query_img = resize_with_aspect_ratio(query_img, vis_height)
        query_label = create_text_image(f"QUERY: ID {query_dog_id}", query_img.shape[1], label_height)
        query_vis = np.vstack([query_img, query_label])

        row_images = [query_vis]

        # --- Create Match Images ---
        for idx in top_indices:
            match_filename = filenames[idx]
            match_path = all_paths[idx]
            match_dog_id = all_dog_ids[idx]
            match_img = cv2.imread(str(match_path))
            match_img = resize_with_aspect_ratio(match_img, vis_height)
            
            # Color code the similarity score based on correctness
            is_correct = query_dog_id == match_dog_id
            color_indicator = "OK" if is_correct else "X"
            label_text = f"ID {match_dog_id} {color_indicator} ({similarities[idx]:.3f})"
            match_label = create_text_image(label_text, match_img.shape[1], label_height)
            match_vis = np.vstack([match_img, match_label])
            row_images.append(match_vis)
        
        # --- Combine row ---
        max_h = max(img.shape[0] for img in row_images)
        row_images = [cv2.copyMakeBorder(img, 0, max_h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) for img in row_images]
        output_rows.append(cv2.hconcat(row_images))

    # --- Combine all rows ---
    max_w = max(row.shape[1] for row in output_rows)
    output_rows = [cv2.copyMakeBorder(row, 0, 0, 0, max_w - row.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for row in output_rows]
    final_image = cv2.vconcat(output_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_image)
    print("Visualization saved.")

def main(args):
    """Main function to run the pipeline and verification."""
    val_json_path = DATA_DIR / "identity_val.json"
    if not val_json_path.exists():
        print(f"[ERROR] Validation JSON not found: {val_json_path}")
        sys.exit(1)

    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    
    image_paths_data = [{"path": PROJECT_ROOT / item["file_path"], "dog_id": item["identity_label"]} for item in val_data]
    random.shuffle(image_paths_data)
    image_paths_data = image_paths_data[:args.num_images]

    print(f"Found {len(val_data)} validation images. Processing {len(image_paths_data)}.")

    pipeline = Pipeline()
    results = {}
    missed_detections = []

    for item in tqdm(image_paths_data, desc="Processing Images"):
        image_path = item["path"]
        dog_id = item["dog_id"]
        
        embeddings = pipeline.run(image_path)
        if embeddings:
            results[image_path.name] = {
                "embedding": embeddings[0], 
                "path": image_path,
                "dog_id": dog_id
            }
        else:
            missed_detections.append((image_path.name, dog_id))

    if not results:
        print("\n[ERROR] No embeddings were generated. Cannot perform analysis.")
        sys.exit(1)

    print(f"\nGenerated {len(results)} embeddings from {len(image_paths_data)} images. Performing similarity analysis...")

    # --- Similarity Analysis ---
    filenames = list(results.keys())
    all_embeddings = np.array([item['embedding'] for item in results.values()])

    similarity_matrix = cosine_similarity(all_embeddings)

    # --- Create Visualization ---
    output_path = PROJECT_ROOT / "outputs/pipeline_verification.jpg"
    create_results_visualization(results, similarity_matrix, output_path, args.num_queries)

    # --- Report Missed Detections ---
    if missed_detections:
        print(f"\n--- {len(missed_detections)} images with no dogs detected ---")
        for filename, dog_id in missed_detections[:10]:  # Show first 10
            print(f"  - {filename} (ID {dog_id})")
        if len(missed_detections) > 10:
            print(f"  ... and {len(missed_detections) - 10} more")

    # --- Calculate accuracy metrics ---
    correct_matches = 0
    total_matches = 0
    
    for i, query_filename in enumerate(filenames):
        query_dog_id = results[query_filename]['dog_id']
        similarities = similarity_matrix[i]
        top_match_idx = np.argsort(similarities)[::-1][1]  # Best match (excluding self)
        match_filename = filenames[top_match_idx]
        match_dog_id = results[match_filename]['dog_id']
        
        if query_dog_id == match_dog_id:
            correct_matches += 1
        total_matches += 1
    
    accuracy = correct_matches / total_matches if total_matches > 0 else 0
    print(f"\nTop-1 Retrieval Accuracy: {correct_matches}/{total_matches} = {accuracy:.3f} ({accuracy*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full dog identification pipeline and verify embeddings.")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images to process from the validation set.")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of query images to show in visualization.")
    args = parser.parse_args()
    main(args)