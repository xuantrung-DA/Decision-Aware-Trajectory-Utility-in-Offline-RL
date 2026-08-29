from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from datu.datasets.trajectory import Trajectory, TransitionDataset
from datu.utils.io import atomic_json_dump


@dataclass(frozen=True)
class CriticalRegion:
    observation_dims: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.observation_dims) == len(self.lower) == len(self.upper)):
            raise ValueError("Critical-region dims/lower/upper lengths must agree")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Every critical-region lower bound must be below its upper bound")

    def mask(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations)[:, self.observation_dims]
        return np.logical_and(
            values >= np.asarray(self.lower), values <= np.asarray(self.upper)
        ).all(axis=1)


def discretized_occupancy(
    observations: np.ndarray,
    dims: Sequence[int] = (0, 1),
    bins: int = 20,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    values = np.asarray(observations)[:, list(dims)]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Occupancy requires at least one selected observation dimension")
    if bounds is None:
        lows, highs = np.min(values, axis=0), np.max(values, axis=0)
    else:
        lows, highs = (np.asarray(item, dtype=np.float64) for item in bounds)
    spans = np.maximum(highs - lows, 1e-6)
    cells = np.floor((values - lows) / spans * bins).astype(np.int64)
    cells = np.clip(cells, 0, bins - 1)
    return float(np.unique(cells, axis=0).shape[0] / (bins ** cells.shape[1]))


def trajectory_structure(trajectory: Trajectory, region: CriticalRegion) -> dict[str, float]:
    critical = region.mask(trajectory.observations)
    spatial = trajectory.observations[:, region.observation_dims]
    rounded_states = np.round(spatial, 1)
    unique_fraction = np.unique(rounded_states, axis=0).shape[0] / len(trajectory)
    action_diversity = float(np.linalg.norm(np.std(trajectory.actions, axis=0)))
    result = {
        "return": trajectory.episode_return,
        "reward_density": trajectory.episode_return / len(trajectory),
        "length": float(len(trajectory)),
        "critical_fraction": float(np.mean(critical)),
        "critical_count": float(np.sum(critical)),
        "action_diversity": action_diversity,
        "redundancy": float(1.0 - unique_fraction),
    }
    for dimension in range(spatial.shape[1]):
        result[f"spatial_mean_{dimension}"] = float(np.mean(spatial[:, dimension]))
        result[f"spatial_std_{dimension}"] = float(np.std(spatial[:, dimension]))
        result[f"spatial_span_{dimension}"] = float(np.ptp(spatial[:, dimension]))
    return result


def dataset_statistics(
    dataset: TransitionDataset,
    region: CriticalRegion,
    coverage_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    returns = dataset.trajectory_returns
    lengths = dataset.trajectory_lengths
    critical_mask = region.mask(dataset.observations)
    return {
        "num_transitions": len(dataset),
        "num_trajectories": int(dataset.num_trajectories),
        "return_mean": float(np.mean(returns)),
        "return_median": float(np.median(returns)),
        "return_std": float(np.std(returns)),
        "return_range": float(np.ptp(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_median": float(np.median(lengths)),
        "length_std": float(np.std(lengths)),
        "global_coverage": discretized_occupancy(dataset.observations, bounds=coverage_bounds),
        "critical_coverage": float(np.mean(critical_mask)),
        "critical_transition_count": int(np.sum(critical_mask)),
        "action_mean": np.mean(dataset.actions, axis=0).tolist(),
        "action_std": np.std(dataset.actions, axis=0).tolist(),
        "dataset_hash": dataset.fingerprint(),
    }


def _relative_error(a: float, b: float, floor: float = 1e-8) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def matching_report(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    errors = {
        "transition_error_fraction": _relative_error(
            float(left["num_transitions"]), float(right["num_transitions"])
        ),
        "trajectory_error_fraction": _relative_error(
            float(left["num_trajectories"]), float(right["num_trajectories"])
        ),
        "mean_return_error_fraction": abs(float(left["return_mean"]) - float(right["return_mean"]))
        / max(float(left.get("return_range", 0.0)), float(right.get("return_range", 0.0)), 1e-8),
        "global_coverage_error_fraction": _relative_error(
            float(left["global_coverage"]), float(right["global_coverage"]), floor=1e-4
        ),
        "median_length_error_fraction": _relative_error(
            float(left["length_median"]), float(right["length_median"]), floor=1.0
        ),
        "critical_coverage_separation": abs(
            float(left["critical_coverage"]) - float(right["critical_coverage"])
        ),
        "action_mean_standardized_error": float(
            np.linalg.norm(np.asarray(left["action_mean"]) - np.asarray(right["action_mean"]))
            / max(
                np.linalg.norm(np.asarray(left["action_std"])),
                np.linalg.norm(np.asarray(right["action_std"])),
                1e-8,
            )
        ),
    }
    checks = {
        "transitions": errors["transition_error_fraction"]
        <= float(tolerances.get("max_transition_error", 0.0)),
        "mean_return": errors["mean_return_error_fraction"]
        <= float(tolerances.get("max_mean_return_error_fraction", 0.02)),
        "global_coverage": errors["global_coverage_error_fraction"]
        <= float(tolerances.get("max_global_coverage_error_fraction", 0.03)),
        "median_length": errors["median_length_error_fraction"]
        <= float(tolerances.get("max_median_length_error_fraction", 0.05)),
        "critical_separation": errors["critical_coverage_separation"]
        >= float(tolerances.get("min_critical_coverage_separation", 0.20)),
        "action_mean": errors["action_mean_standardized_error"]
        <= float(tolerances.get("max_action_mean_standardized_error", float("inf"))),
    }
    return {"errors": errors, "checks": checks, "matched": bool(all(checks.values()))}


class MatchedDatasetBuilder:
    """Build auditable PointMaze conditions by selecting and matching real trajectories."""

    def __init__(self, region: CriticalRegion, seed: int = 0) -> None:
        self.region = region
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _pair_low_return_candidates(
        self,
        trajectories: Sequence[Trajectory],
        added_count: int,
        support_level: float = 1.0,
    ) -> tuple[list[int], list[int]]:
        if not 0.0 <= support_level <= 1.0:
            raise ValueError("support_level must lie in [0,1]")
        features = [trajectory_structure(item, self.region) for item in trajectories]
        returns = np.asarray([item["return"] for item in features])
        # PointMaze episodes are split when a goal is reached, so almost every
        # episode has return exactly one. Reward density distinguishes long,
        # inefficient low-return-rate paths without using online outcomes.
        reward_densities = np.asarray([item["reward_density"] for item in features])
        low_return = reward_densities <= np.median(reward_densities)
        helpful_pool = [
            i for i, item in enumerate(features) if low_return[i] and item["critical_count"] > 0
        ]
        harmful_pool = [
            i for i, item in enumerate(features) if low_return[i] and item["critical_count"] == 0
        ]
        helpful_pool.sort(
            key=lambda i: (features[i]["critical_fraction"], features[i]["action_diversity"]),
            reverse=True,
        )
        harmful_pool.sort(key=lambda i: features[i]["redundancy"], reverse=True)
        pool_limit = max(added_count * 5, added_count)
        helpful_start = round((1.0 - support_level) * max(0, len(helpful_pool) - pool_limit))
        helpful_pool = helpful_pool[helpful_start : helpful_start + pool_limit]
        harmful_pool = harmful_pool[:pool_limit]
        if len(helpful_pool) < added_count or len(harmful_pool) < added_count:
            raise ValueError(
                "Insufficient structurally distinct low-return trajectories: "
                f"helpful={len(helpful_pool)}, harmful={len(harmful_pool)}, needed={added_count}"
            )
        costs = np.empty((len(helpful_pool), len(harmful_pool)), dtype=np.float64)
        return_scale = max(float(np.std(returns)), 1.0)
        length_scale = max(float(np.std([item["length"] for item in features])), 1.0)
        footprint_features = [
            "redundancy",
            "action_diversity",
            *[
                f"spatial_{statistic}_{dimension}"
                for dimension in range(len(self.region.observation_dims))
                for statistic in ("mean", "std", "span")
            ],
        ]
        footprint_scales = {
            name: max(float(np.std([item[name] for item in features])), 1e-6)
            for name in footprint_features
        }
        for row, helpful_index in enumerate(helpful_pool):
            for column, harmful_index in enumerate(harmful_pool):
                primary_cost = (
                    abs(features[helpful_index]["return"] - features[harmful_index]["return"])
                    / return_scale
                    + abs(features[helpful_index]["length"] - features[harmful_index]["length"])
                    / length_scale
                )
                footprint_cost = np.mean(
                    [
                        abs(
                            features[helpful_index][name]
                            - features[harmful_index][name]
                        )
                        / footprint_scales[name]
                        for name in footprint_features
                    ]
                )
                costs[row, column] = primary_cost + 0.5 * footprint_cost
        rows, columns = linear_sum_assignment(costs)
        order = np.argsort(costs[rows, columns])[:added_count]
        return (
            [helpful_pool[int(rows[i])] for i in order],
            [harmful_pool[int(columns[i])] for i in order],
        )

    def build(
        self,
        dataset: TransitionDataset,
        base_count: int,
        added_count: int,
        tolerances: Mapping[str, float],
        support_level: float = 1.0,
    ) -> tuple[dict[str, TransitionDataset], dict[str, Any]]:
        if base_count <= 0 or added_count <= 0:
            raise ValueError("base_count and added_count must be positive")
        trajectories = list(dataset.trajectories())
        structures = [trajectory_structure(item, self.region) for item in trajectories]
        ranked_base = sorted(
            range(len(trajectories)),
            key=lambda index: (structures[index]["return"], structures[index]["critical_count"]),
            reverse=True,
        )
        base_indices = ranked_base[:base_count]
        remaining_indices = [
            index for index in range(len(trajectories)) if index not in set(base_indices)
        ]
        remaining = [trajectories[index] for index in remaining_indices]
        # Three matched cohorts keep the two scientific comparisons independent
        # while giving the reference dataset equally sized, controlled helpful
        # and harmful groups for the marginal-utility interventions.
        cohort_count = 3 * added_count
        helpful_local, harmful_local = self._pair_low_return_candidates(
            remaining, cohort_count, support_level=support_level
        )
        helpful_indices = [remaining_indices[index] for index in helpful_local]
        harmful_indices = [remaining_indices[index] for index in harmful_local]
        critical_helpful = helpful_indices[:added_count]
        reference_harmful = harmful_indices[:added_count]
        critical_low_replacement = harmful_indices[added_count : 2 * added_count]
        helpful_comparison = helpful_indices[2 * added_count : 3 * added_count]
        harmful_comparison = harmful_indices[2 * added_count : 3 * added_count]

        # Every condition has base_count + 2 * added_count trajectories. C1
        # swaps critical_helpful for a return/length-matched harmful cohort;
        # C3 swaps an independent helpful/harmful cohort. The common harmful
        # cohort is also the controlled harmful removal group in the reference.
        conditions = {
            "reference_critical_high": dataset.select_trajectories(
                base_indices + critical_helpful + reference_harmful
            ),
            "critical_low": dataset.select_trajectories(
                base_indices + critical_low_replacement + reference_harmful
            ),
            "helpful_low_return": dataset.select_trajectories(
                base_indices + helpful_comparison + reference_harmful
            ),
            "harmful_low_return": dataset.select_trajectories(
                base_indices + harmful_comparison + reference_harmful
            ),
        }
        condition_hashes = {name: condition.fingerprint() for name, condition in conditions.items()}
        if len(set(condition_hashes.values())) != len(condition_hashes):
            raise RuntimeError(
                "Controlled construction produced duplicate condition hashes; "
                "four scientifically distinct datasets are required"
            )
        coverage_values = dataset.observations[:, :2]
        coverage_bounds = (np.min(coverage_values, axis=0), np.max(coverage_values, axis=0))
        statistics = {
            name: dataset_statistics(condition, self.region, coverage_bounds)
            for name, condition in conditions.items()
        }
        comparisons = {
            "c1_reference_vs_critical_low": matching_report(
                statistics["reference_critical_high"],
                statistics["critical_low"],
                tolerances,
            ),
            "c3_helpful_vs_harmful": matching_report(
                statistics["helpful_low_return"],
                statistics["harmful_low_return"],
                tolerances,
            ),
        }
        manifest = {
            "version": 2,
            "builder": (
                "four-condition real-trajectory structural selection with "
                "return/length bipartite matching"
            ),
            "low_return_definition": (
                "at-or-below-median episode reward density (return/length); PointMaze "
                "episodes are goal-delimited and raw episodic return is almost always one"
            ),
            "seed": self.seed,
            "requested_support_level": support_level,
            "source_hash": dataset.fingerprint(),
            "critical_region": {
                "observation_dims": list(self.region.observation_dims),
                "lower": list(self.region.lower),
                "upper": list(self.region.upper),
                "oracle_use": "construction and controlled evaluation only; never a DATU input",
            },
            "selected_episode_ids": {
                "base": [dataset.episode_ids[i] for i in base_indices],
                "reference_helpful_group": [dataset.episode_ids[i] for i in critical_helpful],
                "reference_harmful_group": [
                    dataset.episode_ids[i] for i in reference_harmful
                ],
                "critical_low_replacement": [
                    dataset.episode_ids[i] for i in critical_low_replacement
                ],
                "helpful_comparison_group": [
                    dataset.episode_ids[i] for i in helpful_comparison
                ],
                "harmful_comparison_group": [
                    dataset.episode_ids[i] for i in harmful_comparison
                ],
            },
            "controlled_group_size": added_count,
            "condition_hashes": condition_hashes,
            "four_unique_condition_hashes": True,
            "statistics": statistics,
            "comparisons": comparisons,
            "all_comparisons_matched": bool(
                all(report["matched"] for report in comparisons.values())
            ),
        }
        return conditions, manifest

    def save(
        self,
        conditions: Mapping[str, TransitionDataset],
        manifest: Mapping[str, Any],
        output_dir: str | Path,
        manifest_path: str | Path,
    ) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, condition in conditions.items():
            condition.save_directory(destination / name)
        atomic_json_dump(manifest, manifest_path)
