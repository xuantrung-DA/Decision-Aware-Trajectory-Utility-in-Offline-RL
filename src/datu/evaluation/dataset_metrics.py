from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from datu.datasets.matched_builder import CriticalRegion
from datu.datasets.trajectory import TransitionDataset
from datu.utility.global_coverage import (
    cluster_coverage,
    discretized_coverage,
    diversity_metrics,
    mean_knn_radius,
)


def compute_dataset_metrics(
    dataset: TransitionDataset,
    critical_region: CriticalRegion | None = None,
    spatial_dims: Sequence[int] = (0, 1),
    seed: int = 0,
    max_support_samples: int = 100_000,
) -> dict[str, Any]:
    returns = dataset.trajectory_returns
    lengths = dataset.trajectory_lengths
    state_action = np.concatenate([dataset.observations, dataset.actions], axis=1)
    rng = np.random.default_rng(seed)
    if len(state_action) > max_support_samples:
        support_indices = rng.choice(len(state_action), max_support_samples, replace=False)
        support_sample = state_action[support_indices]
    else:
        support_sample = state_action
    metrics: dict[str, Any] = {
        "num_transitions": len(dataset),
        "num_trajectories": dataset.num_trajectories,
        "return_mean": float(np.mean(returns)),
        "return_median": float(np.median(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "trajectory_length_mean": float(np.mean(lengths)),
        "trajectory_length_median": float(np.median(lengths)),
        "trajectory_length_std": float(np.std(lengths)),
        "trajectory_length_min": int(np.min(lengths)),
        "trajectory_length_max": int(np.max(lengths)),
        "state_action_mean_knn_radius": mean_knn_radius(support_sample),
        "support_metric_sample_size": len(support_sample),
        "dataset_hash": dataset.fingerprint(),
    }
    if dataset.observation_dim > max(spatial_dims):
        metrics["spatial_coverage"] = discretized_coverage(dataset.observations, dims=spatial_dims)
    metrics["cluster_coverage"] = cluster_coverage(support_sample, seed=seed)
    metrics.update(diversity_metrics(dataset, seed=seed))
    if critical_region is not None:
        mask = critical_region.mask(dataset.observations)
        metrics["critical_coverage"] = float(np.mean(mask))
        metrics["critical_transition_count"] = int(np.sum(mask))
        metrics["critical_trajectory_fraction"] = float(
            np.mean(
                [
                    np.any(mask[start:stop])
                    for start, stop in zip(
                        dataset.trajectory_offsets[:-1], dataset.trajectory_offsets[1:]
                    )
                ]
            )
        )
    return metrics
