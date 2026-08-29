from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datu.datasets.trajectory import TransitionDataset


@dataclass(frozen=True)
class ObservationNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray, epsilon: float = 1e-3) -> ObservationNormalizer:
        values = np.asarray(observations, dtype=np.float64)
        return cls(
            values.mean(axis=0).astype(np.float32),
            (values.std(axis=0) + epsilon).astype(np.float32),
        )

    def transform(self, observations: np.ndarray) -> np.ndarray:
        return ((np.asarray(observations) - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> ObservationNormalizer:
        return cls(
            np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32)
        )


def normalize_dataset(
    dataset: TransitionDataset, normalizer: ObservationNormalizer
) -> TransitionDataset:
    return TransitionDataset(
        observations=normalizer.transform(dataset.observations),
        actions=dataset.actions,
        rewards=dataset.rewards,
        next_observations=normalizer.transform(dataset.next_observations),
        terminations=dataset.terminations,
        truncations=dataset.truncations,
        trajectory_ids=dataset.trajectory_ids,
        trajectory_offsets=dataset.trajectory_offsets,
        episode_ids=dataset.episode_ids,
        metadata={**dataset.metadata, "observation_normalized": True},
    )


class NormalizedAgent:
    """Evaluation adapter applying training-time observation normalization."""

    def __init__(self, agent: Any, normalizer: ObservationNormalizer | None) -> None:
        self.agent = agent
        self.normalizer = normalizer

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        value = observation if self.normalizer is None else self.normalizer.transform(observation)
        return self.agent.act(value, deterministic=deterministic)
