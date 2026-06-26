import os
import random

import numpy as np
import torch

from animal_id.common.seed import set_seed, worker_init_fn


def test_set_seed_returns_generator():
    g = set_seed(42)
    assert isinstance(g, torch.Generator)


def test_set_seed_python_random():
    set_seed(42)
    a = random.random()
    set_seed(42)
    b = random.random()
    assert a == b


def test_set_seed_numpy():
    set_seed(42)
    a = np.random.rand()
    set_seed(42)
    b = np.random.rand()
    assert a == b


def test_set_seed_torch():
    set_seed(42)
    a = torch.rand(1).item()
    set_seed(42)
    b = torch.rand(1).item()
    assert a == b


def test_set_seed_env():
    set_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_different_seeds_differ():
    set_seed(1)
    a = torch.rand(1).item()
    set_seed(2)
    b = torch.rand(1).item()
    assert a != b


def test_worker_init_fn_runs():
    # Should not raise; seeds derived from torch.initial_seed()
    worker_init_fn(0)
    worker_init_fn(3)
