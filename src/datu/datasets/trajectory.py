from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def flatten_observation(observation: Any, keys: Sequence[str] | None = None) -> np.ndarray:
    """Flatten an array or dictionary observation while preserving its leading time axis."""
    if isinstance(observation, Mapping):
        selected = list(keys) if keys is not None else sorted(observation)
        missing = [key for key in selected if key not in observation]
        if missing:
            raise KeyError(f"Observation keys not present: {missing}")
        parts = [flatten_observation(observation[key]) for key in selected]
        if not parts:
            raise ValueError("Observation mapping is empty")
        if any(part.shape[0] != parts[0].shape[0] for part in parts):
            raise ValueError("Observation fields have incompatible leading dimensions")
        return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    array = np.asarray(observation, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


@dataclass(slots=True)
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    episode_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        self.next_observations = np.asarray(self.next_observations, dtype=np.float32)
        self.terminations = np.asarray(self.terminations, dtype=np.bool_).reshape(-1)
        self.truncations = np.asarray(self.truncations, dtype=np.bool_).reshape(-1)
        lengths = {
            len(self.observations),
            len(self.actions),
            len(self.rewards),
            len(self.next_observations),
            len(self.terminations),
            len(self.truncations),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"Trajectory arrays must share one positive length, got {lengths}")
        if self.observations.ndim != 2 or self.next_observations.ndim != 2:
            raise ValueError("Observations must be rank-2 [time, features]")
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        if self.actions.ndim != 2:
            raise ValueError("Actions must be rank-2 [time, features]")
        if self.observations.shape[1] != self.next_observations.shape[1]:
            raise ValueError("Observation and next-observation dimensions differ")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.actions).all():
            raise ValueError("Trajectory contains non-finite observation/action values")

    def __len__(self) -> int:
        return self.rewards.size

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum(dtype=np.float64))

    @property
    def dones(self) -> np.ndarray:
        return np.logical_or(self.terminations, self.truncations)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.terminations,
            self.truncations,
        ):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        digest.update(str(self.episode_id).encode("utf-8"))
        return digest.hexdigest()


def trajectory_from_episode(
    episode: Any,
    episode_id: str | int,
    observation_keys: Sequence[str] | None = None,
) -> Trajectory:
    """Convert a Minari-like episode object to the canonical representation."""
    observations = flatten_observation(episode.observations, observation_keys)
    actions = np.asarray(episode.actions, dtype=np.float32)
    rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
    horizon = len(actions)
    if len(observations) == horizon + 1:
        current, following = observations[:-1], observations[1:]
    elif len(observations) == horizon:
        current = observations
        following = np.concatenate([observations[1:], observations[-1:]], axis=0)
    else:
        raise ValueError(
            f"Episode observation length {len(observations)} is incompatible with {horizon} actions"
        )

    terminations = getattr(episode, "terminations", None)
    if terminations is None:
        terminations = getattr(episode, "terminated", np.zeros(horizon, dtype=np.bool_))
    truncations = getattr(episode, "truncations", None)
    if truncations is None:
        truncations = getattr(episode, "truncated", np.zeros(horizon, dtype=np.bool_))
    metadata: dict[str, Any] = {}
    infos = getattr(episode, "infos", None)
    if isinstance(infos, Mapping):
        for key, value in infos.items():
            array = np.asarray(value)
            if array.size and array.dtype.kind in "biuf":
                metadata[f"info_{key}_mean"] = float(np.mean(array))
    return Trajectory(
        observations=current,
        actions=actions,
        rewards=rewards,
        next_observations=following,
        terminations=np.asarray(terminations)[:horizon],
        truncations=np.asarray(truncations)[:horizon],
        episode_id=str(episode_id),
        metadata=metadata,
    )


class TransitionDataset:
    """Contiguous transition arrays with trajectory boundaries and optional memory mapping."""

    ARRAY_NAMES = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminations",
        "truncations",
        "trajectory_ids",
        "trajectory_offsets",
    )

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        trajectory_ids: np.ndarray,
        trajectory_offsets: np.ndarray,
        episode_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        self.rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self.next_observations = np.asarray(next_observations, dtype=np.float32)
        self.terminations = np.asarray(terminations, dtype=np.bool_).reshape(-1)
        self.truncations = np.asarray(truncations, dtype=np.bool_).reshape(-1)
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)
        self.trajectory_offsets = np.asarray(trajectory_offsets, dtype=np.int64).reshape(-1)
        self.episode_ids = list(episode_ids or [str(i) for i in range(self.num_trajectories)])
        self.metadata = dict(metadata or {})
        self._fingerprint_cache: str | None = None
        self._validate()

    def _validate(self) -> None:
        size = len(self.rewards)
        arrays = (
            self.observations,
            self.actions,
            self.next_observations,
            self.terminations,
            self.truncations,
            self.trajectory_ids,
        )
        if size == 0 or any(len(array) != size for array in arrays):
            raise ValueError("All transition arrays must share a positive first dimension")
        if self.trajectory_offsets.ndim != 1 or self.trajectory_offsets.size < 2:
            raise ValueError("trajectory_offsets must contain at least [0, N]")
        if self.trajectory_offsets[0] != 0 or self.trajectory_offsets[-1] != size:
            raise ValueError("trajectory_offsets must begin at 0 and end at dataset length")
        if np.any(np.diff(self.trajectory_offsets) <= 0):
            raise ValueError("Every trajectory must contain at least one transition")
        if len(self.episode_ids) != self.num_trajectories:
            raise ValueError("episode_ids length differs from number of trajectory boundaries")
        expected_ids = np.repeat(np.arange(self.num_trajectories), np.diff(self.trajectory_offsets))
        if not np.array_equal(self.trajectory_ids, expected_ids):
            raise ValueError("trajectory_ids do not agree with trajectory_offsets")

    @classmethod
    def from_trajectories(
        cls, trajectories: Iterable[Trajectory], metadata: Mapping[str, Any] | None = None
    ) -> TransitionDataset:
        items = list(trajectories)
        if not items:
            raise ValueError("Cannot build a transition dataset from no trajectories")
        lengths = np.asarray([len(item) for item in items], dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
        ids = np.repeat(np.arange(len(items), dtype=np.int64), lengths)
        return cls(
            observations=np.concatenate([item.observations for item in items]),
            actions=np.concatenate([item.actions for item in items]),
            rewards=np.concatenate([item.rewards for item in items]),
            next_observations=np.concatenate([item.next_observations for item in items]),
            terminations=np.concatenate([item.terminations for item in items]),
            truncations=np.concatenate([item.truncations for item in items]),
            trajectory_ids=ids,
            trajectory_offsets=offsets,
            episode_ids=[item.episode_id for item in items],
            metadata=metadata,
        )

    def __len__(self) -> int:
        return self.rewards.size

    @property
    def num_trajectories(self) -> int:
        return max(0, self.trajectory_offsets.size - 1)

    @property
    def observation_dim(self) -> int:
        return self.observations.shape[1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    @property
    def dones(self) -> np.ndarray:
        return np.logical_or(self.terminations, self.truncations)

    @property
    def trajectory_lengths(self) -> np.ndarray:
        return np.diff(self.trajectory_offsets)

    @property
    def trajectory_returns(self) -> np.ndarray:
        return np.add.reduceat(self.rewards.astype(np.float64), self.trajectory_offsets[:-1])

    @property
    def next_actions(self) -> np.ndarray:
        result = np.empty_like(self.actions)
        for start, stop in zip(self.trajectory_offsets[:-1], self.trajectory_offsets[1:]):
            result[start : stop - 1] = self.actions[start + 1 : stop]
            result[stop - 1] = self.actions[stop - 1]
        return result

    def get_trajectory(self, index: int) -> Trajectory:
        if index < 0:
            index += self.num_trajectories
        if not 0 <= index < self.num_trajectories:
            raise IndexError(index)
        start, stop = self.trajectory_offsets[index : index + 2]
        return Trajectory(
            observations=self.observations[start:stop],
            actions=self.actions[start:stop],
            rewards=self.rewards[start:stop],
            next_observations=self.next_observations[start:stop],
            terminations=self.terminations[start:stop],
            truncations=self.truncations[start:stop],
            episode_id=self.episode_ids[index],
        )

    def trajectories(self) -> Iterable[Trajectory]:
        for index in range(self.num_trajectories):
            yield self.get_trajectory(index)

    def select_trajectories(self, indices: Sequence[int]) -> TransitionDataset:
        clean = [int(index) for index in indices]
        if len(set(clean)) != len(clean):
            raise ValueError("Trajectory selection contains duplicates")
        return TransitionDataset.from_trajectories(
            [self.get_trajectory(index) for index in clean], metadata=self.metadata
        )

    def remove_trajectories(self, indices: Sequence[int]) -> TransitionDataset:
        removed = {int(index) for index in indices}
        kept = [index for index in range(self.num_trajectories) if index not in removed]
        if not kept:
            raise ValueError("Group removal would leave an empty dataset")
        return self.select_trajectories(kept)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = rng.integers(0, len(self), size=batch_size)
        trajectory_ids = self.trajectory_ids[indices]
        trajectory_stops = self.trajectory_offsets[trajectory_ids + 1]
        next_indices = np.minimum(indices + 1, trajectory_stops - 1)
        arrays = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices, None],
            "next_observations": self.next_observations[indices],
            "dones": np.logical_or(self.terminations[indices], self.truncations[indices])[
                :, None
            ].astype(np.float32),
            "next_actions": self.actions[next_indices],
        }
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in arrays.items()
        }

    def fingerprint(self) -> str:
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache
        digest = hashlib.sha256()
        for name in self.ARRAY_NAMES:
            digest.update(np.ascontiguousarray(getattr(self, name)).view(np.uint8))
        digest.update(json.dumps(self.episode_ids, sort_keys=True).encode("utf-8"))
        self._fingerprint_cache = digest.hexdigest()
        return self._fingerprint_cache

    def save_npz(self, path: str | Path, compressed: bool = True) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = np.savez_compressed if compressed else np.savez
        writer(
            destination,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.next_observations,
            terminations=self.terminations,
            truncations=self.truncations,
            trajectory_ids=self.trajectory_ids,
            trajectory_offsets=self.trajectory_offsets,
            episode_ids=np.asarray(self.episode_ids, dtype=np.str_),
            metadata=np.asarray(json.dumps(self.metadata)),
            dataset_fingerprint=np.asarray(self.fingerprint()),
        )
        return destination

    @classmethod
    def load_npz(cls, path: str | Path) -> TransitionDataset:
        with np.load(path, allow_pickle=False) as data:
            metadata_raw = data["metadata"].item() if "metadata" in data else "{}"
            dataset = cls(
                **{name: data[name] for name in cls.ARRAY_NAMES},
                episode_ids=data["episode_ids"].astype(str).tolist(),
                metadata=json.loads(metadata_raw),
            )
            if "dataset_fingerprint" in data:
                dataset._fingerprint_cache = str(data["dataset_fingerprint"].item())
            return dataset

    def save_directory(self, path: str | Path) -> Path:
        """Save separate .npy arrays so large datasets can later be memory-mapped."""
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        for name in self.ARRAY_NAMES:
            np.save(destination / f"{name}.npy", getattr(self, name), allow_pickle=False)
        manifest = {
            "episode_ids": self.episode_ids,
            "metadata": self.metadata,
            "dataset_fingerprint": self.fingerprint(),
        }
        (destination / "dataset.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load_directory(cls, path: str | Path, mmap_mode: str | None = "r") -> TransitionDataset:
        source = Path(path)
        manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
        arrays = {
            name: np.load(source / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False)
            for name in cls.ARRAY_NAMES
        }
        dataset = cls(
            **arrays,
            episode_ids=manifest["episode_ids"],
            metadata=manifest.get("metadata", {}),
        )
        dataset._fingerprint_cache = manifest.get("dataset_fingerprint")
        return dataset
