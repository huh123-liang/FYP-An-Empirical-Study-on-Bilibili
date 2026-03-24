"""
utils/seed.py
-------------
Global random seed utility for reproducibility across numpy, torch, and CUDA.
Call set_seed(cfg.training.seed) at the start of every experiment entry point.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and CUDA.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops — may slow down training slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
