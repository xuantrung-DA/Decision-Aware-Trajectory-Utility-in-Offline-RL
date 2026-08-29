from __future__ import annotations

import numpy as np

from datu.datasets.trajectory import TransitionDataset
from datu.utility.datu import DATUConfig, compute_datu_scores, datu_variants
from datu.utility.decision_sensitivity import decision_sensitivity
from datu.utility.global_coverage import (
    discretized_coverage,
    diversity_metrics,
    trajectory_coverage_contribution,
)
from datu.utility.local_coverage import LocalSupportEstimator, group_support_contribution
from datu.utility.rarity import knn_rarity, trajectory_rarity
from datu.utility.return_score import reward_density
from datu.utility.uncertainty import action_ranking_disagreement, q_uncertainty


def test_coverage_rarity_and_diversity(tiny_dataset: TransitionDataset) -> None:
    coverage = discretized_coverage(tiny_dataset.observations, bins=10)
    assert 0.0 < coverage["fraction"] <= 1.0
    rarity = knn_rarity(tiny_dataset.observations, tiny_dataset.actions, k=4)
    assert rarity.shape == (len(tiny_dataset),) and np.all(rarity >= 0)
    assert trajectory_rarity(tiny_dataset, k=4).shape == (tiny_dataset.num_trajectories,)
    contributions = trajectory_coverage_contribution(tiny_dataset, bins=10)
    assert contributions.shape == (tiny_dataset.num_trajectories,)
    assert all(value >= 0 for value in diversity_metrics(tiny_dataset).values())


def test_reward_density_is_return_per_trajectory_length(
    tiny_dataset: TransitionDataset,
) -> None:
    expected = tiny_dataset.trajectory_returns / tiny_dataset.trajectory_lengths
    np.testing.assert_allclose(reward_density(tiny_dataset), expected)


def test_local_support_metrics_and_group_removal(tiny_dataset: TransitionDataset) -> None:
    estimator = LocalSupportEstimator(k=5).fit(
        tiny_dataset.observations, tiny_dataset.actions, tiny_dataset.trajectory_ids
    )
    metrics = estimator.point_metrics(tiny_dataset.observations[:7], tiny_dataset.actions[:7])
    assert metrics.radius.shape == (7,)
    contribution = group_support_contribution(
        tiny_dataset.observations,
        tiny_dataset.actions,
        tiny_dataset.trajectory_ids % 3,
        k=5,
        query_batch_size=16,
    )
    assert contribution.shape == (len(tiny_dataset),)
    assert np.all(contribution >= 0)


def test_uncertainty_sensitivity_and_datu(tiny_dataset: TransitionDataset) -> None:
    rng = np.random.default_rng(2)
    q = rng.normal(size=(5, len(tiny_dataset), 7))
    uncertainty = q_uncertainty(q)["std"].mean(axis=1)
    sensitivity = decision_sensitivity(q)
    disagreement = action_ranking_disagreement(q)
    assert uncertainty.shape == sensitivity.shape == disagreement.shape == (len(tiny_dataset),)
    support = np.linspace(0.0, 1.0, len(tiny_dataset))
    point, trajectory = compute_datu_scores(
        {"sensitivity": sensitivity, "uncertainty": uncertainty, "local_support": support},
        tiny_dataset.trajectory_offsets,
        DATUConfig(),
    )
    assert point.shape == (len(tiny_dataset),)
    assert trajectory.shape == (tiny_dataset.num_trajectories,)
    for variant in datu_variants().values():
        _, scores = compute_datu_scores(
            {"sensitivity": sensitivity, "uncertainty": uncertainty, "local_support": support},
            tiny_dataset.trajectory_offsets,
            variant,
        )
        assert np.isfinite(scores).all()
