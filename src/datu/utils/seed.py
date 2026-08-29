from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, env: Any | None = None, deterministic: bool = True) -> np.random.Generator:
    """Seed Python, NumPy, PyTorch, CUDA and an optional Gymnasium environment."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by CUDA >= 10.2 for reproducible CuBLAS GEMM kernels.
        # This must be set before the first CUDA matrix multiplication.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # Older supported torch releases.
            torch.use_deterministic_algorithms(True)
    if env is not None:
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    return np.random.default_rng(seed)
