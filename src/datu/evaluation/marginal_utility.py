from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datu.datasets.splits import group_trajectories, indices_for_group
from datu.datasets.trajectory import TransitionDataset
from datu.evaluation.bootstrap import bootstrap_ci
from datu.utils.io import atomic_json_dump


@dataclass(frozen=True)
class RemovalGroup:
    name: str
    trajectory_indices: tuple[int, ...]


def score_quantile_groups(scores: np.ndarray, quantiles: int = 10) -> list[RemovalGroup]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or quantiles < 2 or len(values) < quantiles:
        raise ValueError("Invalid scores/quantile count")
    order = np.argsort(values)
    partitions = np.array_split(order, quantiles)
    return [
        RemovalGroup(f"quantile_{index:02d}", tuple(part.astype(int).tolist()))
        for index, part in enumerate(partitions)
    ]


def top_bottom_groups(scores: np.ndarray, fraction: float = 0.1) -> list[RemovalGroup]:
    values = np.asarray(scores, dtype=np.float64)
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must lie in (0,0.5)")
    count = max(1, int(np.ceil(len(values) * fraction)))
    order = np.argsort(values)
    return [
        RemovalGroup(f"bottom_{fraction:g}", tuple(order[:count].astype(int).tolist())),
        RemovalGroup(f"top_{fraction:g}", tuple(order[-count:].astype(int).tolist())),
    ]


def fixed_groups(
    dataset: TransitionDataset,
    count: int = 8,
    strategy: str = "structure",
    seed: int = 0,
) -> list[RemovalGroup]:
    assignments = group_trajectories(dataset, count, strategy=strategy, seed=seed)
    return [
        RemovalGroup(f"{strategy}_{group:02d}", tuple(indices_for_group(assignments, group)))
        for group in range(count)
    ]


def random_groups(dataset: TransitionDataset, count: int = 8, seed: int = 0) -> list[RemovalGroup]:
    return fixed_groups(dataset, count=count, strategy="random", seed=seed)


def controlled_groups(labels: Sequence[str]) -> list[RemovalGroup]:
    values = np.asarray(labels)
    return [
        RemovalGroup(str(label), tuple(np.flatnonzero(values == label).astype(int).tolist()))
        for label in np.unique(values)
    ]


def save_group_removals(
    dataset: TransitionDataset,
    groups: Sequence[RemovalGroup],
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for group in groups:
        reduced = dataset.remove_trajectories(group.trajectory_indices)
        path = destination / group.name
        reduced.save_directory(path)
        records.append(
            {
                "group": group.name,
                "removed_trajectory_indices": list(group.trajectory_indices),
                "removed_episodes": [
                    dataset.episode_ids[index] for index in group.trajectory_indices
                ],
                "remaining_transitions": len(reduced),
                "remaining_trajectories": reduced.num_trajectories,
                "remaining_hash": reduced.fingerprint(),
                "path": str(path),
            }
        )
    manifest = {"full_dataset_hash": dataset.fingerprint(), "groups": records}
    atomic_json_dump(manifest, destination / "removal_manifest.json")
    return manifest


def empirical_marginal_utility(
    full_performance: Mapping[tuple[str, int], float],
    removed_performance: Mapping[tuple[str, str, int], float],
) -> list[dict[str, Any]]:
    """Compute U_A(G|D)=J(A,D)-J(A,D\\G) per seed from completed evaluations."""
    results = []
    for (algorithm, group, seed), removed_value in removed_performance.items():
        key = (algorithm, seed)
        if key not in full_performance:
            raise KeyError(f"Missing full-data performance for {key}")
        results.append(
            {
                "algorithm": algorithm,
                "group": group,
                "seed": seed,
                "full_performance": float(full_performance[key]),
                "removed_performance": float(removed_value),
                "marginal_utility": float(full_performance[key] - removed_value),
            }
        )
    return results


def summarize_empirical_marginal_utility(
    records: Sequence[Mapping[str, Any]],
    resamples: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["algorithm"]), str(record["group"]))
        grouped.setdefault(key, []).append(record)
    summaries = []
    for (algorithm, group), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["seed"]))
        values = np.asarray([item["marginal_utility"] for item in ordered], dtype=np.float64)
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        interval = (
            bootstrap_ci(values, resamples=resamples, seed=seed)
            if len(values) >= 2
            else None
        )
        summaries.append(
            {
                "algorithm": algorithm,
                "group": group,
                "n_seeds": len(values),
                "seeds": [int(item["seed"]) for item in ordered],
                "individual_marginal_utilities": values.tolist(),
                "mean_paired_marginal_utility": float(np.mean(values)),
                "std_paired_marginal_utility": standard_deviation,
                "paired_cohen_d": (
                    float(np.mean(values) / standard_deviation)
                    if standard_deviation > 0
                    else None
                ),
                "bootstrap_95_ci": interval,
                "same_positive_direction": int(np.sum(values > 0.0)),
                "small_n_warning": len(values) < 5,
            }
        )
    return summaries
