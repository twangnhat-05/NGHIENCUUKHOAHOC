"""Reproducibility: set seed cho python/numpy/torch/sklearn."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Set seed cho mọi RNG đã có sẵn trong môi trường."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except ImportError:
        pass
