from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from datu.datasets.trajectory import TransitionDataset
from datu.evaluation.marginal_utility import RemovalGroup, save_group_removals
from datu.utils.io import atomic_json_dump, canonical_json_hash, sha256_file

CORE_GROUPS = (
    "remove_top_datu",
    "remove_bottom_datu",
    "remove_top_return",
    "remove_bottom_return",
    "remove_helpful_controlled",
    "remove_harmful_controlled",
)


def _artifact_hash(data: Mapping[str, Any]) -> str:
    return canonical_json_hash(
        {key: value for key, value in data.items() if key != "preregistration_hash"}
    )


def load_preregistration(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    expected = data.get("preregistration_hash")
    actual = _artifact_hash(data)
    if not expected or expected != actual:
        raise ValueError(f"Preregistration hash mismatch: {source}")
    if data.get("version") != 1:
        raise ValueError(f"Unsupported preregistration version: {data.get('version')}")
    groups = data.get("intervention_groups", {})
    if set(groups) != set(CORE_GROUPS):
        raise ValueError("Preregistration must contain exactly the six frozen core groups")
    group_sizes = {int(record["trajectory_count"]) for record in groups.values()}
    if len(group_sizes) != 1:
        raise ValueError("All preregistered intervention groups must have equal trajectory count")
    return data


def verify_preregistered_dataset(
    config: Mapping[str, Any], dataset: TransitionDataset
) -> dict[str, Any] | None:
    specification = config.get("preregistration")
    if not specification:
        return None
    path = specification.get("path")
    group = specification.get("group")
    condition = specification.get("condition")
    if group and condition:
        raise ValueError("Preregistration cannot specify both a condition and a removal group")
    required = bool(specification.get("required", False))
    if not path:
        if required:
            raise ValueError("A required preregistration has no path")
        return None
    source = Path(path)
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Required preregistration is absent: {source}")
        return None
    data = load_preregistration(source)
    if condition:
        if condition not in data["condition_hashes"]:
            raise KeyError(f"Condition {condition!r} is not frozen in {source}")
        expected_hash = data["condition_hashes"][condition]
    elif not group:
        expected_hash = data["reference_dataset_hash"]
    else:
        if group not in data["intervention_groups"]:
            raise KeyError(f"Group {group!r} is not frozen in {source}")
        expected_hash = data["intervention_groups"][group]["removal_dataset_hash"]
    actual_hash = dataset.fingerprint()
    if expected_hash != actual_hash:
        raise ValueError(
            f"Dataset hash {actual_hash} does not match preregistered hash "
            f"{expected_hash} for {group or condition or 'reference'}"
        )
    return data


def _extreme_indices(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Frozen ranking scores must be a finite vector")
    if count <= 0 or 2 * count > len(scores):
        raise ValueError("Group size must be positive and permit disjoint top/bottom groups")
    order = np.argsort(scores, kind="stable")
    return order[-count:], order[:count]


def _indices_for_episode_ids(
    dataset: TransitionDataset, episode_ids: list[str], label: str
) -> np.ndarray:
    lookup = {str(episode_id): index for index, episode_id in enumerate(dataset.episode_ids)}
    missing = sorted(set(map(str, episode_ids)).difference(lookup))
    if missing:
        raise ValueError(f"Controlled group {label} has episode IDs absent from reference: {missing}")
    indices = np.asarray([lookup[str(episode_id)] for episode_id in episode_ids], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Controlled group {label} contains duplicate episode IDs")
    return indices


def freeze_core_preregistration(
    reference_dataset: TransitionDataset,
    matched_manifest_path: str | Path,
    datu_components_path: str | Path,
    removal_dir: str | Path,
    output_path: str | Path,
    *,
    group_size: int | None = None,
    ensemble_size: int = 5,
    critic_steps_per_member: int = 100_000,
    knn_k: int = 50,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    manifest_path = Path(matched_manifest_path)
    components_path = Path(datu_components_path)
    matched = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_hash = reference_dataset.fingerprint()
    expected_reference_hash = matched["condition_hashes"]["reference_critical_high"]
    if reference_hash != expected_reference_hash:
        raise ValueError("Reference dataset differs from the frozen matched-dataset manifest")
    if not matched.get("all_comparisons_matched", False):
        raise ValueError("Cannot preregister interventions from an unmatched construction")

    with np.load(components_path, allow_pickle=False) as components:
        datu_scores = np.asarray(components["trajectory_scores"], dtype=np.float64)
    if datu_scores.shape != (reference_dataset.num_trajectories,):
        raise ValueError("DATU trajectory score count differs from reference trajectories")

    frozen_size = int(group_size or matched["controlled_group_size"])
    top_datu, bottom_datu = _extreme_indices(datu_scores, frozen_size)
    # Minari PointMaze closes a trajectory when the goal is reached, so every
    # reference trajectory has raw episodic return 1. Ranking that constant
    # vector would create arbitrary, duplicated interventions. Reward density
    # retains the requested return baseline while measuring path efficiency.
    returns = reference_dataset.trajectory_returns
    reward_density = returns / reference_dataset.trajectory_lengths
    top_return, bottom_return = _extreme_indices(reward_density, frozen_size)
    selected = matched["selected_episode_ids"]
    helpful = _indices_for_episode_ids(
        reference_dataset,
        selected["reference_helpful_group"],
        "reference_helpful_group",
    )
    harmful = _indices_for_episode_ids(
        reference_dataset,
        selected["reference_harmful_group"],
        "reference_harmful_group",
    )
    if len(helpful) != frozen_size or len(harmful) != frozen_size:
        raise ValueError("Controlled group sizes differ from the frozen intervention budget")

    groups = [
        RemovalGroup("remove_top_datu", tuple(map(int, top_datu))),
        RemovalGroup("remove_bottom_datu", tuple(map(int, bottom_datu))),
        RemovalGroup("remove_top_return", tuple(map(int, top_return))),
        RemovalGroup("remove_bottom_return", tuple(map(int, bottom_return))),
        RemovalGroup("remove_helpful_controlled", tuple(map(int, helpful))),
        RemovalGroup("remove_harmful_controlled", tuple(map(int, harmful))),
    ]
    removal_manifest = save_group_removals(reference_dataset, groups, removal_dir)
    removal_records = {record["group"]: record for record in removal_manifest["groups"]}
    group_records: dict[str, Any] = {}
    for group in groups:
        indices = np.asarray(group.trajectory_indices, dtype=np.int64)
        record = removal_records[group.name]
        group_records[group.name] = {
            "trajectory_count": len(indices),
            "trajectory_indices": indices.tolist(),
            "episode_ids": [str(reference_dataset.episode_ids[index]) for index in indices],
            "mean_datu": float(np.mean(datu_scores[indices])),
            "mean_return": float(np.mean(returns[indices])),
            "mean_reward_density": float(np.mean(reward_density[indices])),
            "removal_dataset_hash": record["remaining_hash"],
            "removal_dataset_path": record["path"],
            "remaining_transitions": record["remaining_transitions"],
            "remaining_trajectories": record["remaining_trajectories"],
        }

    data: dict[str, Any] = {
        "version": 1,
        "study": "LAB-REVIEW CORE v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": reference_dataset.metadata.get("dataset_id"),
        "reference_dataset_hash": reference_hash,
        "condition_hashes": matched["condition_hashes"],
        "critical_region": matched["critical_region"],
        "datu_definition": {
            "interpretation": "DATU_ref; one frozen offline reference ensemble for all downstream algorithms",
            "combination": "product",
            "component_normalization": "robust percentile clipping at [5,95] then [0,1] scaling",
            "ensemble_size": int(ensemble_size),
            "critic_steps_per_member": int(critic_steps_per_member),
            "knn_k": int(knn_k),
            "aggregation": aggregation,
            "topk_fraction": float(topk_fraction),
            "seed": int(seed),
            "online_outcome_used": False,
        },
        "evaluation_protocol": {
            "episodes": 20,
            "max_episode_steps": 250,
            "checkpoint_selection": "final",
            "paired_training_seeds": True,
        },
        "utility_validation_protocol": {
            "groups": list(CORE_GROUPS),
            "primary": "pairwise ranking accuracy over non-tied empirical group-utility pairs",
            "prediction_tie_credit": 0.5,
            "secondary": "Spearman correlation; descriptive only because n=6",
            "baselines": ["reward_density", "rarity", "global_coverage"],
            "return_score_definition": (
                "episode return / trajectory length; raw PointMaze episodic return is constant"
            ),
            "component_ablation_acceptance": (
                "full DATU pairwise accuracy exceeds at least two of three component removals"
            ),
        },
        "intervention_budget_trajectories": frozen_size,
        "intervention_groups": group_records,
        "source_artifacts": {
            "matched_manifest": str(manifest_path),
            "matched_manifest_sha256": sha256_file(manifest_path),
            "datu_components": str(components_path),
            "datu_components_sha256": sha256_file(components_path),
            "removal_manifest": str(Path(removal_dir) / "removal_manifest.json"),
        },
    }
    data["preregistration_hash"] = _artifact_hash(data)
    atomic_json_dump(data, output_path)
    return data
