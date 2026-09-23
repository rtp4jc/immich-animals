"""
Centralized visualization utilities for the dog identification system.

Provides consistent visualization functions for:
- Dataset inspection (COCO, YOLO formats)
- Model outputs (bboxes, keypoints, embeddings)
- Training results and metrics
- Pipeline visualizations

All functions default to saving files to outputs/ directory with optional display mode.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def setup_output_dir(output_dir: str | Path) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_or_show_plot(
    save_path: str | Path | None = None, display: bool = False, dpi: int = 150
) -> None:
    """Save plot to file and/or display it."""
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved plot to: {save_path}")

    if display:
        plt.show()
    else:
        plt.close()


# COCO Dataset Visualization
def visualize_coco_annotations(
    coco_json_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path = "outputs/coco_visualization",
    num_samples: int = 5,
    display: bool = False,
) -> None:
    """Visualize COCO format annotations."""
    coco_path = Path(coco_json_path)
    data_root = Path(data_root)
    output_path = setup_output_dir(output_dir)

    if not coco_path.exists():
        logger.warning(f"COCO file not found: {coco_path}")
        return

    with open(coco_path) as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data.get("annotations", [])

    # Group annotations by image
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)

    # Sample images
    sample_images = random.sample(images, min(num_samples, len(images)))

    fig, axes = plt.subplots(1, len(sample_images), figsize=(4 * len(sample_images), 4))
    if len(sample_images) == 1:
        axes = [axes]

    for i, img_info in enumerate(sample_images):
        img_path = data_root / img_info["file_name"]
        if not img_path.exists():
            continue

        # Load and display image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"ID: {img_info['id']}")
        axes[i].axis("off")

        # Draw annotations
        for ann in ann_by_image[img_info["id"]]:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[i].add_patch(rect)

            # Draw keypoints if available
            if "keypoints" in ann:
                kpts = ann["keypoints"]
                for j in range(0, len(kpts), 3):
                    if kpts[j + 2] > 0:  # visible
                        axes[i].plot(kpts[j], kpts[j + 1], "ro", markersize=3)

    plt.tight_layout()
    save_or_show_plot(output_path / "coco_samples.png", display)


# YOLO Dataset Visualization
def visualize_yolo_annotations(
    data_yaml_path: str | Path,
    output_dir: str | Path = "outputs/yolo_visualization",
    num_samples: int = 5,
    display: bool = False,
) -> None:
    """Visualize YOLO format annotations."""
    import yaml

    yaml_path = Path(data_yaml_path)
    output_path = setup_output_dir(output_dir)

    if not yaml_path.exists():
        logger.warning(f"YOLO config file not found: {yaml_path}")
        return

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Read validation images
    val_txt = Path(config["val"])
    if not val_txt.exists():
        logger.warning(f"Validation file not found: {val_txt}")
        return

    with open(val_txt) as f:
        image_paths = [line.strip() for line in f.readlines()]

    # Filter for images with annotations (non-empty label files)
    images_with_labels = []
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        # Convert image path to label path: images/val2017/xxx.jpg -> labels/val2017/xxx.txt
        label_path_str = img_path_str.replace("/images/", "/labels/").replace(
            ".jpg", ".txt"
        )
        label_path = Path(label_path_str)

        if label_path.exists() and label_path.stat().st_size > 0:
            images_with_labels.append(img_path)

    if not images_with_labels:
        logger.info("No images with annotations found")
        return

    # Sample from images with labels
    sample_paths = random.sample(
        images_with_labels, min(num_samples, len(images_with_labels))
    )

    fig, axes = plt.subplots(1, len(sample_paths), figsize=(4 * len(sample_paths), 4))
    if len(sample_paths) == 1:
        axes = [axes]

    for i, img_path in enumerate(sample_paths):
        if not img_path.exists():
            continue

        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        axes[i].imshow(img_rgb)
        axes[i].set_title(img_path.name)
        axes[i].axis("off")

        # Load corresponding label file
        label_path_str = (
            str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt")
        )
        label_path = Path(label_path_str)

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x_c, y_c, width, height = map(float, parts[:5])

                        # Convert normalized to pixel coordinates
                        x1 = int((x_c - width / 2) * w)
                        y1 = int((y_c - height / 2) * h)
                        x2 = int((x_c + width / 2) * w)
                        y2 = int((y_c + height / 2) * h)

                        rect = patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=2,
                            edgecolor="blue",
                            facecolor="none",
                        )
                        axes[i].add_patch(rect)

                        # Draw keypoints if available
                        if len(parts) > 5:
                            kpts = list(map(float, parts[5:]))
                            for j in range(0, len(kpts), 3):
                                if j + 2 < len(kpts) and kpts[j + 2] > 0:
                                    kx = int(kpts[j] * w)
                                    ky = int(kpts[j + 1] * h)
                                    axes[i].plot(kx, ky, "bo", markersize=3)

    plt.tight_layout()
    save_or_show_plot(output_path / "yolo_samples.png", display)


# Detection Results Visualization
def visualize_detection_results(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    output_path: str | Path | None = None,
    display: bool = False,
) -> None:
    """Visualize detection model outputs."""
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title(f"Detections: {Path(image_path).name}")
    ax.axis("off")

    for det in detections:
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        conf = det.get("confidence", 0.0)

        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 5, f"{conf:.2f}", color="red", fontsize=10)

    save_or_show_plot(output_path, display)


# Keypoint Results Visualization
def visualize_keypoint_results(
    image_path: str | Path,
    keypoints: list[dict[str, Any]],
    output_path: str | Path | None = None,
    display: bool = False,
) -> None:
    """Visualize keypoint model outputs."""
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title(f"Keypoints: {Path(image_path).name}")
    ax.axis("off")

    keypoint_names = ["nose", "chin", "left_ear", "right_ear"]
    colors = ["red", "blue", "green", "orange"]

    for kpt_data in keypoints:
        bbox = kpt_data.get("bbox", [])
        if bbox:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor="gray",
                facecolor="none",
            )
            ax.add_patch(rect)

        kpts = kpt_data.get("keypoints", [])
        for i in range(0, len(kpts), 3):
            if i // 3 < len(keypoint_names) and kpts[i + 2] > 0:
                ax.plot(
                    kpts[i],
                    kpts[i + 1],
                    "o",
                    color=colors[i // 3],
                    markersize=8,
                    label=keypoint_names[i // 3] if i == 0 else "",
                )

    if keypoints:
        ax.legend()

    save_or_show_plot(output_path, display)


# Training Metrics Visualization
def visualize_training_metrics(
    metrics_data: dict[str, list[float]],
    output_path: str | Path | None = None,
    display: bool = False,
) -> None:
    """Visualize training metrics and loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    metric_names = list(metrics_data.keys())
    for i, metric in enumerate(metric_names[:4]):
        if i < len(axes):
            axes[i].plot(metrics_data[metric])
            axes[i].set_title(metric)
            axes[i].set_xlabel("Epoch")
            axes[i].grid(True)

    # Hide unused subplots
    for i in range(len(metric_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    save_or_show_plot(output_path, display)


# Embedding Visualization
# Dataset Statistics
def print_dataset_statistics(coco_json_path: str | Path) -> dict[str, Any]:
    """Print and return dataset statistics."""
    with open(coco_json_path) as f:
        data = json.load(f)

    stats = {
        "num_images": len(data["images"]),
        "num_annotations": len(data.get("annotations", [])),
        "num_categories": len(data.get("categories", [])),
        "has_keypoints": any("keypoints" in ann for ann in data.get("annotations", [])),
    }

    logger.info("Dataset Statistics:")
    logger.info(f"  Images: {stats['num_images']}")
    logger.info(f"  Annotations: {stats['num_annotations']}")
    logger.info(f"  Categories: {stats['num_categories']}")
    logger.info(f"  Has Keypoints: {stats['has_keypoints']}")

    return stats
