from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from datu.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_difference
from datu.evaluation.correlations import (
    classification_metrics,
    correlation_metrics,
    paired_metric_bootstrap,
)
from datu.evaluation.marginal_utility import empirical_marginal_utility, fixed_groups
from datu.evaluation.policy_eval import evaluate_policy
from datu.evaluation.utility_validation import pairwise_ranking_accuracy


class _NeverEndingEnvironment:
    def __init__(self) -> None:
        self.action_space = type(
            "ActionSpace",
            (),
            {
                "low": np.asarray([-1.0], dtype=np.float32),
                "high": np.asarray([1.0], dtype=np.float32),
            },
        )()
        self.position = 0

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, bool]]:
        del seed
        self.position = 0
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, bool]]:
        del action
        self.position += 1
        observation = np.asarray([float(self.position), 0.0], dtype=np.float32)
        return observation, 1.0, False, False, {}


class _ZeroAgent:
    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        del observation, deterministic
        return np.zeros(1, dtype=np.float32)


def test_bootstrap_ci_and_paired_difference() -> None:
    values = np.arange(10, dtype=np.float64)
    interval = bootstrap_ci(values, resamples=300, seed=2)
    assert interval["lower"] < interval["estimate"] < interval["upper"]
    paired = paired_bootstrap_difference(values + 1, values, resamples=300, seed=2)
    assert np.isclose(paired["estimate"], 1.0)


def test_correlations_classification_and_metric_ci() -> None:
    target = np.arange(12, dtype=np.float64)
    predicted = target + np.sin(target) * 0.1
    correlations = correlation_metrics(predicted, target)
    assert correlations["spearman"] > 0.9
    labels = np.array([0] * 6 + [1] * 6)
    scores = np.linspace(0, 1, 12)
    classification = classification_metrics(scores, labels)
    assert classification["auroc"] == 1.0
    interval = paired_metric_bootstrap(
        predicted,
        target,
        lambda x, y: spearmanr(x, y).statistic,
        resamples=200,
        seed=1,
    )
    assert interval["estimate"] > 0.9
    auroc_interval = paired_metric_bootstrap(
        scores,
        labels,
        lambda x, y: roc_auc_score(y, x),
        resamples=200,
        seed=1,
        stratify=labels,
    )
    assert auroc_interval["estimate"] == 1.0


def test_grouped_marginal_utility(tiny_dataset) -> None:
    groups = fixed_groups(tiny_dataset, count=3, strategy="structure", seed=1)
    assert len(groups) == 3
    full = {("iql", 0): 50.0}
    removed = {("iql", group.name, 0): 40.0 + index for index, group in enumerate(groups)}
    result = empirical_marginal_utility(full, removed)
    assert len(result) == 3
    assert result[0]["marginal_utility"] == 10.0


def test_policy_evaluation_enforces_frozen_horizon_and_tracks_region() -> None:
    result = evaluate_policy(
        _ZeroAgent(),
        _NeverEndingEnvironment(),
        episodes=2,
        max_episode_steps=5,
        critical_region={
            "observation_dims": [0],
            "lower": [0.0],
            "upper": [1.0],
        },
    )
    assert result["episode_lengths"] == [5, 5]
    assert result["episodic_returns"] == [5.0, 5.0]
    assert result["horizon_truncations"] == 2
    assert np.isclose(result["critical_visit_fraction_mean"], 0.4)
    assert result["critical_episode_fraction"] == 1.0


def test_pairwise_ranking_accuracy_counts_ties_conservatively() -> None:
    result = pairwise_ranking_accuracy(
        {"a": 3.0, "b": 1.0, "c": 1.0},
        {"a": 3.0, "b": 2.0, "c": 1.0},
    )
    assert result["informative_pairs"] == 3
    assert result["prediction_ties"] == 1
    assert np.isclose(result["pairwise_accuracy"], 2.5 / 3.0)
