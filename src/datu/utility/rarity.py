from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from datu.datasets.trajectory import TransitionDataset
from datu.utility.features import project_for_neighbors


def normalized_state_action(
    observations: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("Observation/action sample counts differ")
    features = np.concatenate([observations, actions], axis=1)
    scaler = StandardScaler().fit(features)
    return scaler.transform(features), scaler


def knn_rarity(
    observations: np.ndarray,
    actions: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Mean non-self kNN radius in standardized state-action space; larger is rarer."""
    features, _ = normalized_state_action(observations, actions)
    features, _ = project_for_neighbors(features)
    if len(features) < 2:
        raise ValueError("At least two transitions are required for kNN rarity")
    neighbors = min(k + 1, len(features))
    if neighbors < 2:
        raise ValueError("kNN rarity has no non-self neighbor")
    distances, _ = (
        NearestNeighbors(n_neighbors=neighbors, algorithm="kd_tree", n_jobs=1)
        .fit(features)
        .kneighbors()
    )
    return distances[:, 1:].mean(axis=1)


def trajectory_rarity(dataset: TransitionDataset, k: int = 20) -> np.ndarray:
    point_scores = knn_rarity(dataset.observations, dataset.actions, k)
    return np.asarray(
        [
            point_scores[start:stop].mean()
            for start, stop in zip(dataset.trajectory_offsets[:-1], dataset.trajectory_offsets[1:])
        ],
        dtype=np.float64,
    )
