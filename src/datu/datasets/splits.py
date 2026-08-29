from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from datu.datasets.trajectory import TransitionDataset
from datu.utils.parallelism import suppress_loky_physical_core_probe


def split_by_trajectory(
    dataset: TransitionDataset,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[TransitionDataset, TransitionDataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if dataset.num_trajectories < 2:
        raise ValueError("At least two trajectories are required for a split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(dataset.num_trajectories)
    validation_count = max(1, round(dataset.num_trajectories * validation_fraction))
    validation_indices = order[:validation_count].tolist()
    train_indices = order[validation_count:].tolist()
    return dataset.select_trajectories(train_indices), dataset.select_trajectories(
        validation_indices
    )


def group_trajectories(
    dataset: TransitionDataset,
    n_groups: int,
    strategy: str = "random",
    seed: int = 0,
    features: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each trajectory to a fixed group independently of downstream policy return."""
    if not 1 < n_groups <= dataset.num_trajectories:
        raise ValueError("n_groups must be in [2, num_trajectories]")
    rng = np.random.default_rng(seed)
    if strategy == "random":
        assignment = np.empty(dataset.num_trajectories, dtype=np.int64)
        order = rng.permutation(dataset.num_trajectories)
        assignment[order] = np.arange(dataset.num_trajectories) % n_groups
        return assignment
    if strategy == "structure":
        from sklearn.cluster import KMeans

        if features is None:
            features = np.stack(
                [
                    np.concatenate(
                        [
                            trajectory.observations.mean(axis=0),
                            trajectory.observations.std(axis=0),
                            trajectory.actions.mean(axis=0),
                            trajectory.actions.std(axis=0),
                            np.asarray([len(trajectory)], dtype=np.float32),
                        ]
                    )
                    for trajectory in dataset.trajectories()
                ]
            )
        with suppress_loky_physical_core_probe(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Could not find the number of physical cores.*"
            )
            return KMeans(n_clusters=n_groups, random_state=seed, n_init=10).fit_predict(features)
    raise ValueError(f"Unknown grouping strategy: {strategy}")


def indices_for_group(assignments: Sequence[int], group: int) -> list[int]:
    values = np.asarray(assignments)
    return np.flatnonzero(values == group).astype(int).tolist()
