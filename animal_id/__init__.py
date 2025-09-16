"""
Dog identification system for Immich.

A 3-stage pipeline for detecting and identifying individual dogs in photos:
1. Dog Detector - YOLO11n finds dog bounding boxes
2. Keypoint Estimator - YOLO11n-pose finds 4 face keypoints (eyes, nose, throat)
3. Identity Embedder - ResNet50 + ArcFace loss → 512D embeddings
"""

__version__ = "0.1.0"
