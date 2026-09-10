"""Bounded, post-hoc supplementary controls; never modify the frozen 86-run study."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from datu.datasets.trajectory import TransitionDataset
from datu.evaluation.utility_validation import group_score_means, pairwise_ranking_accuracy
from datu.utility.datu import compute_datu_scores, datu_variants
from datu.utils.io import atomic_json_dump, canonical_json_hash

STUDY = "IJMLC_SUPP_V1"
REFERENCE_HASH = "ab1b0f9006e9b6dd4ee786b88134fddbc5b2802eb62f2a637fb0a507f3767c95"
MATRIX_HASH = "a6ec37a53ef4b60a66520252a44625b50399982e402f6137383ea725310c1168"
GROUP_SEEDS = (2026090901, 2026090902)
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEED = 20000  # fresh, shared panel; no checkpoint selection
EVAL_EPISODES = 100


def fresh_fingerprint(dataset: TransitionDataset) -> str:
    # Loading a processed dataset trusts its cached fingerprint by default.
    dataset._fingerprint_cache = None
    return dataset.fingerprint()


def signature(record: dict[str, Any]) -> str:
    return canonical_json_hash({k: v for k, v in record.items() if k != "signature"})


def load_protocol(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("study") != STUDY or record.get("signature") != signature(record):
        raise ValueError(f"Supplement protocol signature mismatch: {path}")
    return record


def matched_random_indices(
    lengths: np.ndarray, target: np.ndarray, seed: int, *, max_attempts: int = 100_000
) -> tuple[np.ndarray, dict[str, Any]]:
    """Uniform within length strata, conditional on a fixed transition tolerance.

    The FIRST accepted sample is used. Neither DATU values nor policy outcomes
    are inputs. Overlap with the target or another draw is allowed, not optimized.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("Lengths must be positive")
    if (
        len(np.unique(target)) != len(target)
        or np.any(target < 0)
        or np.any(target >= len(lengths))
    ):
        raise ValueError("Invalid target indices")
    edges = np.unique(np.quantile(lengths, np.arange(1, 10) / 10))
    bins = np.searchsorted(edges, lengths, side="right")
    pools = [np.flatnonzero(bins == b) for b in range(len(edges) + 1)]
    counts = np.bincount(bins[target], minlength=len(pools))
    removed_steps = int(lengths[target].sum())
    tolerance = max(1, int(np.floor(removed_steps * 0.01)))
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        draw = np.sort(
            np.concatenate(
                [
                    rng.choice(pool, size=int(count), replace=False)
                    for pool, count in zip(pools, counts)
                    if count
                ]
            )
        )
        actual_steps = int(lengths[draw].sum())
        if abs(actual_steps - removed_steps) <= tolerance:
            return draw, {
                "selection_seed": seed,
                "accepted_attempt": attempt,
                "length_bin_edges": edges.tolist(),
                "length_bin_counts": counts.tolist(),
                "target_removed_transitions": removed_steps,
                "removed_transitions": actual_steps,
                "tolerance_transitions": tolerance,
                "target_overlap_trajectories": len(np.intersect1d(draw, target)),
                "removed_length_mean": float(lengths[draw].mean()),
                "removed_length_std": float(lengths[draw].std()),
            }
    raise RuntimeError("No matched sample found; do not silently relax the frozen tolerance")


def freeze_protocol(dataset: TransitionDataset, original: dict[str, Any]) -> dict[str, Any]:
    if fresh_fingerprint(dataset) != REFERENCE_HASH:
        raise ValueError("Wrong reference dataset")
    target = np.asarray(original["intervention_groups"]["remove_top_datu"]["trajectory_indices"])
    groups = {}
    for number, seed in enumerate(GROUP_SEEDS, 1):
        indices, record = matched_random_indices(dataset.trajectory_lengths, target, seed)
        remaining = dataset.remove_trajectories(indices)
        groups[f"random_matched_{number}"] = {
            **record,
            "trajectory_indices": indices.tolist(),
            "episode_ids": [dataset.episode_ids[i] for i in indices],
            "remaining_transitions": len(remaining),
            "remaining_trajectories": remaining.num_trajectories,
            "remaining_dataset_hash": fresh_fingerprint(remaining),
        }
    record = {
        "study": STUDY,
        "version": 1,
        "original_matrix_signature": MATRIX_HASH,
        "reference_hash": REFERENCE_HASH,
        "groups": groups,
        "original_preregistration_hash": original["preregistration_hash"],
        "design": "two independent length-stratified, transition-matched random removals",
        "selection": "first sample within 1% of REMOVED transitions; match 40 trajectories and length strata",
        "post_hoc": True,
        "original_outcomes_seen_before_design": True,
        "new_control_outcomes_used_for_group_selection": False,
        "training_seeds": list(TRAIN_SEEDS),
        "algorithm": "rebrac",
        "steps": 200_000,
        "evaluation": {
            "episodes": EVAL_EPISODES,
            "seed": EVAL_SEED,
            "horizon": 250,
            "checkpoint": "final",
            "deterministic": True,
        },
        "primary_contrast": "mean_random_retained_success - top_datu_removed_success, paired by training seed",
        "analysis": "average two random groups within each seed; n=3 training seeds, NOT n=6",
        "limitations": [
            "only ReBRAC",
            "two random draws",
            "one dataset",
            "post-hoc supplementary control",
        ],
        "jobs": [
            {"id": f"{STUDY}_{group}_seed{seed}", "group": group, "seed": seed}
            for seed in TRAIN_SEEDS
            for group in groups
        ],
    }
    record["signature"] = signature(record)
    return record


def make_training_config(
    base: dict[str, Any], protocol: dict[str, Any], job: dict[str, Any], root: Path, output: Path
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    group = job["group"]
    # V1's validator only accepts the ORIGINAL six groups. The supplement has
    # its own signed protocol and performs an independent hash check before training.
    config.pop("preregistration", None)
    config["seed"] = job["seed"]
    config["device"] = "cuda"
    config["dataset"]["path"] = str(
        root / "data/processed/matched/pointmaze_medium/reference_critical_high"
    )
    config["dataset"]["remove_trajectory_indices"] = protocol["groups"][group]["trajectory_indices"]
    config["training"]["steps"] = 200_000
    config["logging"]["root"] = str(output / "runs")
    config["logging"]["dataset_metrics_cache"] = str(output / "cache")
    config["logging"]["tensorboard"] = False  # CSV/progress logging retained
    config["experiment"] = {
        "id": STUDY,
        "block": "TRANSITION_MATCHED_RANDOM",
        "variant": group,
        "removed_group": group,
    }
    config["supplement_protocol_signature"] = protocol["signature"]
    return config


def score_ablations(
    dataset: TransitionDataset,
    components: dict[str, np.ndarray],
    original: dict[str, Any],
    outcomes: dict[str, dict[str, float]],
    output: Path,
) -> list[dict[str, Any]]:
    rows = []
    for variant, settings in datu_variants().items():
        _, scores = compute_datu_scores(components, dataset.trajectory_offsets, settings)
        if variant == "datu_full":
            np.testing.assert_allclose(
                scores, components["trajectory_scores"], rtol=1e-10, atol=1e-12
            )
        predicted = group_score_means(scores, original)
        for algorithm, target in outcomes.items():
            metrics = pairwise_ranking_accuracy(predicted, target)
            rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "informative_pairs": metrics["informative_pairs"],
                    "pairwise_accuracy": metrics["pairwise_accuracy"],
                    "spearman": metrics["spearman"],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "component_score_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json_dump(
        {
            "scope": "descriptive score ablation on six existing overlapping groups",
            "not_policy_retraining_ablation": True,
            "rows": rows,
        },
        output / "ablation.json",
    )
    return rows
