#!/usr/bin/env python
"""
Shared constants for the dog_id project.

This file contains common paths and configuration values that are used across
different scripts (training, export, inference) to ensure consistency.
"""

from pathlib import Path

# --- Core Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# --- Phase 1: Detector and Keypoint Models ---
PHASE1_MODELS_DIR = MODELS_DIR / "phase1"

# Detector
DETECTOR_RUN_NAME = "detector_run"
DETECTOR_PROJECT_DIR = PHASE1_MODELS_DIR

# Keypoint
KEYPOINT_RUN_NAME = "keypoint_run"
KEYPOINT_PROJECT_DIR = PHASE1_MODELS_DIR

# --- ONNX Export ---
ONNX_DIR = MODELS_DIR / "onnx"
ONNX_DETECTOR_PATH = ONNX_DIR / "detector.onnx"
ONNX_KEYPOINT_PATH = ONNX_DIR / "keypoint.onnx"
ONNX_EMBEDDING_PATH = ONNX_DIR / "embedding.onnx"
