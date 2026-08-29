from __future__ import annotations

import numpy as np

from datu.datasets.trajectory import TransitionDataset


def episodic_returns(dataset: TransitionDataset) -> np.ndarray:
    return dataset.trajectory_returns.astype(np.float64, copy=True)


def reward_density(dataset: TransitionDataset) -> np.ndarray:
    """Per-trajectory return rate, robust to goal-delimited sparse-reward episodes."""
    return episodic_returns(dataset) / dataset.trajectory_lengths


def normalized_returns(
    returns: np.ndarray,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    low = float(np.min(values) if minimum is None else minimum)
    high = float(np.max(values) if maximum is None else maximum)
    if not high > low:
        return np.zeros_like(values)
    return (values - low) / (high - low)
