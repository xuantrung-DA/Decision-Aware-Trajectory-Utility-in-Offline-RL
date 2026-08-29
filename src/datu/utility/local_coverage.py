from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from datu.utility.features import project_for_neighbors


@dataclass
class LocalSupportMetrics:
    radius: np.ndarray
    density: np.ndarray
    action_diversity: np.ndarray
    trajectory_diversity: np.ndarray


class LocalSupportEstimator:
    """Standardized kNN local support model for continuous state-action datasets."""

    def __init__(self, k: int = 50) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        self.state_neighbors: NearestNeighbors | None = None
        self.state_action_neighbors: NearestNeighbors | None = None
        self.observations: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.trajectory_ids: np.ndarray | None = None
        self.state_projector = None
        self.state_action_projector = None

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        trajectory_ids: np.ndarray,
    ) -> LocalSupportEstimator:
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        if not (len(observations) == len(actions) == len(trajectory_ids)) or len(observations) < 2:
            raise ValueError("Local support requires aligned arrays with at least two samples")
        states = self.state_scaler.fit_transform(observations)
        normalized_actions = self.action_scaler.fit_transform(actions)
        state_actions = np.concatenate([states, normalized_actions], axis=1)
        states, self.state_projector = project_for_neighbors(states)
        state_actions, self.state_action_projector = project_for_neighbors(state_actions)
        count = min(self.k + 1, len(states))
        self.state_neighbors = NearestNeighbors(n_neighbors=count, n_jobs=1).fit(states)
        self.state_action_neighbors = NearestNeighbors(n_neighbors=count, n_jobs=1).fit(
            state_actions
        )
        self.observations = observations
        self.actions = actions
        self.trajectory_ids = trajectory_ids
        return self

    def _require_fit(self) -> None:
        if self.state_neighbors is None or self.state_action_neighbors is None:
            raise RuntimeError("LocalSupportEstimator.fit must be called first")

    def point_metrics(self, observations: np.ndarray, actions: np.ndarray) -> LocalSupportMetrics:
        self._require_fit()
        assert self.actions is not None and self.trajectory_ids is not None
        states = self.state_scaler.transform(np.asarray(observations, dtype=np.float64))
        normalized_actions = self.action_scaler.transform(np.asarray(actions, dtype=np.float64))
        state_actions = np.concatenate([states, normalized_actions], axis=1)
        if self.state_projector is not None:
            states = self.state_projector.transform(states)
        if self.state_action_projector is not None:
            state_actions = self.state_action_projector.transform(state_actions)
        sa_distances, _ = self.state_action_neighbors.kneighbors(state_actions)
        _, state_indices = self.state_neighbors.kneighbors(states)
        radius = sa_distances[:, -1]
        dimension = state_actions.shape[1]
        density = 1.0 / np.maximum(radius, 1e-8) ** dimension
        action_diversity = np.asarray(
            [np.linalg.norm(np.std(self.actions[index], axis=0)) for index in state_indices]
        )
        trajectory_diversity = np.asarray(
            [len(np.unique(self.trajectory_ids[index])) / len(index) for index in state_indices]
        )
        return LocalSupportMetrics(radius, density, action_diversity, trajectory_diversity)


def group_support_contribution(
    observations: np.ndarray,
    actions: np.ndarray,
    group_ids: np.ndarray,
    k: int = 50,
    query_batch_size: int = 4096,
    probe_multiplier: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Approximate log kNN-radius increase when each query's own group is removed.

    A single neighbor index is queried in chunks. The larger probe neighborhood is filtered by
    group membership, avoiding one full kNN refit per trajectory group on million-step datasets.
    """
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    groups = np.asarray(group_ids)
    if not (len(observations) == len(actions) == len(groups)):
        raise ValueError("Support contribution arrays must be aligned")
    state_scaler = StandardScaler().fit(observations)
    action_scaler = StandardScaler().fit(actions)
    features = np.concatenate(
        [state_scaler.transform(observations), action_scaler.transform(actions)], axis=1
    )
    features, _ = project_for_neighbors(features)
    if k <= 0 or query_batch_size <= 0 or probe_multiplier < 1:
        raise ValueError("Invalid support-contribution k/batch/probe parameters")
    probe_count = min(max(k * probe_multiplier + 1, k + 2), len(features))
    model = NearestNeighbors(n_neighbors=probe_count, algorithm="kd_tree", n_jobs=1).fit(features)
    contribution = np.zeros(len(features), dtype=np.float64)
    nonself_rank = min(k, probe_count - 1)
    for start in range(0, len(features), query_batch_size):
        stop = min(start + query_batch_size, len(features))
        distances, indices = model.kneighbors(features[start:stop])
        full_radius = distances[:, nonself_rank]
        for local_index, global_index in enumerate(range(start, stop)):
            outside = distances[local_index, groups[indices[local_index]] != groups[global_index]]
            if outside.size == 0:
                continue
            without_radius = outside[min(k, outside.size) - 1]
            contribution[global_index] = max(
                0.0,
                float(
                    np.log(max(without_radius, 1e-8)) - np.log(max(full_radius[local_index], 1e-8))
                ),
            )
        if progress_callback is not None:
            progress_callback(stop, len(features))
    return contribution
