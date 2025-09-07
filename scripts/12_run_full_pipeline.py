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

def preprocess_detector(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Prepares an image for the YOLO detector model."""
    # [TODO] This is a placeholder and needs to be replaced with the actual
    # YOLOv11 pre-processing logic if it differs.
    h, w, _ = image.shape
    target_h, target_w = input_size
    scale = min(target_h / h, target_w / w)
    unpad_h, unpad_w = int(round(h * scale)), int(round(w * scale))
    pad_t = (target_h - unpad_h) // 2
    pad_b = target_h - unpad_h - pad_t
    pad_l = (target_w - unpad_w) // 2
    pad_r = target_w - unpad_w - pad_l

    resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    img_fp = padded.astype(np.float32) / 255.0
    img_fp = np.transpose(img_fp, (2, 0, 1))
    return np.expand_dims(img_fp, axis=0), scale

def postprocess_detector(output: np.ndarray, scale: float, conf_threshold: float = 0.5, nms_threshold: float = 0.45) -> list[np.ndarray]:
    """Extracts bounding boxes from our custom single-class detector."""
    preds = np.squeeze(output).T
    scores = preds[:, 4]
    preds = preds[scores > conf_threshold]
    scores = scores[scores > conf_threshold]
    if len(preds) == 0:
        return []

    boxes_cx = preds[:, 0]
    boxes_cy = preds[:, 1]
    boxes_w = preds[:, 2]
    boxes_h = preds[:, 3]
    boxes_x1 = boxes_cx - boxes_w / 2
    boxes_y1 = boxes_cy - boxes_h / 2

    indices = cv2.dnn.NMSBoxes(np.column_stack((boxes_x1, boxes_y1, boxes_w, boxes_h)).tolist(), scores.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    final_boxes = []
    for i in indices.flatten():
        x1, y1, w, h = boxes_x1[i], boxes_y1[i], boxes_w[i], boxes_h[i]
        final_boxes.append(np.array([x1 / scale, y1 / scale, (x1 + w) / scale, (y1 + h) / scale]))
    return final_boxes

def postprocess_keypoint(output: np.ndarray, scale: float, conf_threshold: float = 0.5) -> np.ndarray | None:
    """Extracts keypoints from our custom 4-keypoint pose model."""
    preds = np.squeeze(output).T
    scores = preds[:, 4]
    
    # Filter out detections with low confidence
    confident_preds = preds[scores > conf_threshold]
    confident_scores = scores[scores > conf_threshold]

    if len(confident_preds) == 0:
        return None

    # Select the detection with the highest confidence
    best_detection = confident_preds[np.argmax(confident_scores)]
    keypoints = best_detection[5:].reshape((4, 3))

    # Scale keypoints back to the original crop's dimensions
    keypoints[:, 0] /= scale # x
    keypoints[:, 1] /= scale # y
    return keypoints

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
        detector_input_size = self.detector.get_inputs()[0].shape[2:]
        detector_input, scale = preprocess_detector(image, detector_input_size)
        detector_output = self.detector.run(None, {self.detector.get_inputs()[0].name: detector_input})[0]
        bboxes = postprocess_detector(detector_output, scale)

        if not bboxes:
            return []

        embeddings = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            padding = int((x2 - x1) * 0.1)
            crop_x1, crop_y1 = max(0, x1 - padding), max(0, y1 - padding)
            crop_x2, crop_y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
            detector_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

            if detector_crop.size == 0: continue

            # --- Stage 2: Keypoint Estimation ---
            keypoint_input_size = self.keypoint.get_inputs()[0].shape[2:]
            keypoint_input, keypoint_scale = preprocess_detector(detector_crop, keypoint_input_size)
            keypoint_output = self.keypoint.run(None, {self.keypoint.get_inputs()[0].name: keypoint_input})[0]
            keypoints = postprocess_keypoint(keypoint_output, keypoint_scale)

            final_crop = detector_crop
            if keypoints is not None and np.all(keypoints[:, 2] > 0.3):
                # If all keypoints are confident, crop to keypoints
                kx1, ky1 = np.min(keypoints[:, :2], axis=0).astype(int)
                kx2, ky2 = np.max(keypoints[:, :2], axis=0).astype(int)
                kp_padding = int((kx2 - kx1) * 0.2)
                final_crop_x1 = max(0, kx1 - kp_padding)
                final_crop_y1 = max(0, ky1 - kp_padding)
                final_crop_x2 = min(detector_crop.shape[1], kx2 + kp_padding)
                final_crop_y2 = min(detector_crop.shape[0], ky2 + kp_padding)
                final_crop = detector_crop[final_crop_y1:final_crop_y2, final_crop_x1:final_crop_x2]

            if final_crop.size == 0: continue

            # --- Stage 3: Embedding ---
            embedding_input_size = self.embedder.get_inputs()[0].shape[2:]
            embedding_input = preprocess_embedding(final_crop, embedding_input_size)
            embedding = self.embedder.run(None, {self.embedder.get_inputs()[0].name: embedding_input})[0][0]
            embeddings.append(embedding)

        return embeddings

def create_results_visualization(results: dict, similarity_matrix: np.ndarray, output_path: Path, num_queries: int = 10):
    """Creates a single image visualizing the similarity results."""
    print(f"\nCreating visualization at: {output_path}")
    filenames = list(results.keys())
    num_queries = min(num_queries, len(filenames))
    
    # Select queries evenly spaced across the dataset
    query_indices = np.linspace(0, len(filenames) - 1, num_queries, dtype=int)
    
    output_rows = []
    vis_height = 224
    label_height = 30

    for i, query_idx in enumerate(tqdm(query_indices, desc="Creating Visualization")):
        query_filename = filenames[query_idx]
        query_data = results[query_filename]
        query_path = query_data['path']
        query_dog_id = query_data['dog_id']

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
            match_data = results[match_filename]
            match_path = match_data['path']
            match_dog_id = match_data['dog_id']
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
    
    # Process all validation images, not just a subset
    print(f"Found {len(val_data)} validation images. Processing all images...")

    pipeline = Pipeline()
    results = {}
    missed_detections = []

    for item in tqdm(val_data, desc="Processing Images"):
        image_path = PROJECT_ROOT / item["file_path"]
        dog_id = item["identity_label"]
        
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

    print(f"\nGenerated {len(results)} embeddings from {len(val_data)} images. Performing similarity analysis...")

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
    parser.add_argument("--num-queries", type=int, default=10, help="Number of query images to show in visualization.")
    args = parser.parse_args()
    main(args)
