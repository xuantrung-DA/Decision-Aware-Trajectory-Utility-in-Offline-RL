from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def group_score_means(
    trajectory_scores: np.ndarray, preregistration: Mapping[str, Any]
) -> dict[str, float]:
    scores = np.asarray(trajectory_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Trajectory scores must be a finite vector")
    result = {}
    for name, record in preregistration["intervention_groups"].items():
        indices = np.asarray(record["trajectory_indices"], dtype=np.int64)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(scores)):
            raise ValueError(f"Invalid preregistered indices for {name}")
        result[name] = float(np.mean(scores[indices]))
    return result


def pairwise_ranking_accuracy(
    predicted: Mapping[str, float], target: Mapping[str, float]
) -> dict[str, Any]:
    names = sorted(set(predicted).intersection(target))
    if len(names) < 3:
        raise ValueError("Pairwise validation requires at least three aligned groups")
    concordant = 0.0
    informative = 0
    ties = 0
    for left, right in combinations(names, 2):
        target_difference = float(target[left] - target[right])
        if np.isclose(target_difference, 0.0):
            continue
        predicted_difference = float(predicted[left] - predicted[right])
        informative += 1
        if np.isclose(predicted_difference, 0.0):
            concordant += 0.5
            ties += 1
        elif np.sign(predicted_difference) == np.sign(target_difference):
            concordant += 1.0
    if informative == 0:
        raise ValueError("All empirical marginal-utility pairs are tied")
    predicted_vector = np.asarray([predicted[name] for name in names], dtype=np.float64)
    target_vector = np.asarray([target[name] for name in names], dtype=np.float64)
    return {
        "groups": names,
        "informative_pairs": informative,
        "prediction_ties": ties,
        "pairwise_accuracy": concordant / informative,
        "spearman": float(spearmanr(predicted_vector, target_vector).statistic),
    }
