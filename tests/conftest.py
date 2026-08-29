from __future__ import annotations

import numpy as np
import pytest

from datu.datasets.trajectory import Trajectory, TransitionDataset


@pytest.fixture
def tiny_dataset() -> TransitionDataset:
    rng = np.random.default_rng(11)
    trajectories = []
    for episode in range(12):
        length = 8 + episode % 3
        if episode < 4:
            x = np.linspace(-1.0, 1.0, length)
            y = np.zeros(length)
            rewards = np.ones(length, dtype=np.float32)
        elif episode < 8:
            x = np.linspace(-0.8, 0.1, length)
            y = 0.02 * rng.normal(size=length)
            rewards = np.full(length, -0.1, dtype=np.float32)
        else:
            theta = np.linspace(0, 2 * np.pi, length)
            x = 0.8 + 0.02 * np.cos(theta)
            y = 0.8 + 0.02 * np.sin(theta)
            rewards = np.full(length, -0.1, dtype=np.float32)
        observations = np.stack([x, y, np.gradient(x)], axis=1).astype(np.float32)
        actions = np.clip(
            np.stack([np.gradient(x), np.gradient(y)], axis=1) * 3.0,
            -1.0,
            1.0,
        ).astype(np.float32)
        next_observations = np.concatenate([observations[1:], observations[-1:]])
        terminations = np.zeros(length, dtype=np.bool_)
        truncations = np.zeros(length, dtype=np.bool_)
        truncations[-1] = True
        trajectories.append(
            Trajectory(
                observations,
                actions,
                rewards,
                next_observations,
                terminations,
                truncations,
                str(episode),
            )
        )
    return TransitionDataset.from_trajectories(trajectories)
