from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from datu.datasets.trajectory import TransitionDataset
from datu.utility.features import project_for_neighbors
from datu.utils.parallelism import suppress_loky_physical_core_probe


def discretized_coverage(
    observations: np.ndarray,
    dims: Sequence[int] = (0, 1),
    bins: int = 20,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float | int]:
    values = np.asarray(observations, dtype=np.float64)[:, list(dims)]
    if values.shape[0] == 0 or bins <= 1:
        raise ValueError("Coverage requires observations and bins > 1")
    if bounds is None:
        low, high = np.min(values, axis=0), np.max(values, axis=0)
    else:
        low, high = (np.asarray(item, dtype=np.float64) for item in bounds)
    span = np.maximum(high - low, 1e-8)
    cells = np.floor((values - low) / span * bins).astype(np.int64)
    cells = np.clip(cells, 0, bins - 1)
    occupied = np.unique(cells, axis=0).shape[0]
    total = bins ** values.shape[1]
    return {
        "occupied_cells": int(occupied),
        "total_cells": int(total),
        "fraction": occupied / total,
    }


def mean_knn_radius(features: np.ndarray, k: int = 20) -> float:
    values = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    values, _ = project_for_neighbors(values)
    if len(values) < 2:
        return 0.0
    count = min(k + 1, len(values))
    distances, _ = (
        NearestNeighbors(n_neighbors=count, algorithm="kd_tree", n_jobs=1).fit(values).kneighbors()
    )
    return float(np.mean(distances[:, 1:]))


def cluster_coverage(features: np.ndarray, clusters: int = 100, seed: int = 0) -> dict[str, float]:
    values = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    count = min(clusters, len(values))
    if count < 1:
        raise ValueError("Cluster coverage requires non-empty features")
    with suppress_loky_physical_core_probe(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
        labels = MiniBatchKMeans(n_clusters=count, random_state=seed, n_init=3).fit_predict(values)
    counts = np.bincount(labels, minlength=count).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return {
        "occupied_clusters": float(np.count_nonzero(counts)),
        "normalized_entropy": float(entropy / np.log(count)) if count > 1 else 0.0,
    }


def sampled_pairwise_diversity(values: np.ndarray, max_samples: int = 2000, seed: int = 0) -> float:
    array = StandardScaler().fit_transform(np.asarray(values, dtype=np.float64))
    if len(array) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(array) > max_samples:
        array = array[rng.choice(len(array), max_samples, replace=False)]
    distances = pairwise_distances(array)
    upper = distances[np.triu_indices(len(array), k=1)]
    return float(np.mean(upper))


def diversity_metrics(dataset: TransitionDataset, seed: int = 0) -> dict[str, float]:
    state_action = np.concatenate([dataset.observations, dataset.actions], axis=1)
    return {
        "state_diversity": sampled_pairwise_diversity(dataset.observations, seed=seed),
        "action_diversity": sampled_pairwise_diversity(dataset.actions, seed=seed),
        "state_action_diversity": sampled_pairwise_diversity(state_action, seed=seed),
    }


def trajectory_coverage_contribution(
    dataset: TransitionDataset,
    dims: Sequence[int] = (0, 1),
    bins: int = 20,
) -> np.ndarray:
    """Fraction of occupied spatial cells uniquely supplied by each trajectory."""
    values = dataset.observations[:, list(dims)]
    low, high = np.min(values, axis=0), np.max(values, axis=0)
    span = np.maximum(high - low, 1e-8)
    cells = np.floor((values - low) / span * bins).astype(np.int64)
    cells = np.clip(cells, 0, bins - 1)
    codes = np.ravel_multi_index(cells.T, (bins,) * cells.shape[1])
    owners: dict[int, set[int]] = {}
    for code, trajectory_id in zip(codes, dataset.trajectory_ids):
        owners.setdefault(int(code), set()).add(int(trajectory_id))
    total_occupied = max(len(owners), 1)
    unique_counts = np.zeros(dataset.num_trajectories, dtype=np.float64)
    for owner_set in owners.values():
        if len(owner_set) == 1:
            unique_counts[next(iter(owner_set))] += 1.0
    return unique_counts / total_occupied
