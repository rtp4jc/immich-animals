"""
Common utilities for the dog identification system.

Contains shared functionality used across all modules:
- Dataset utilities and constants
- Visualization functions
- Inference pipeline utilities
- Utility functions and helpers
"""

from animal_id.common.seed import set_seed, worker_init_fn

__all__ = ["set_seed", "worker_init_fn"]
