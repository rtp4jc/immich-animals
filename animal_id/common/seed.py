import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch RNGs. Returns a Generator for DataLoader."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seed derived from the main generator seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
