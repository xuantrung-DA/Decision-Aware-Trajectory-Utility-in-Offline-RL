from __future__ import annotations

import numpy as np
import pytest

from datu.algorithms import build_agent
from datu.datasets.trajectory import TransitionDataset
from datu.utils.seed import set_seed


@pytest.mark.parametrize("name", ["bc", "iql", "cql", "rebrac"])
def test_agent_update_action_and_checkpoint(
    name: str, tiny_dataset: TransitionDataset, tmp_path
) -> None:
    extras = {"cql_n_actions": 2} if name == "cql" else {}
    config = {"algorithm": {"name": name, "hidden_dims": [16, 16], **extras}}
    agent = build_agent(config, tiny_dataset.observation_dim, tiny_dataset.action_dim, "cpu")
    metrics = agent.train_step(tiny_dataset.sample(8, np.random.default_rng(0), "cpu"))
    assert metrics and all(np.isfinite(value) for value in metrics.values())
    action = agent.act(tiny_dataset.observations[0])
    batch_actions = agent.act(tiny_dataset.observations[:3])
    assert action.shape == (tiny_dataset.action_dim,)
    assert batch_actions.shape == (3, tiny_dataset.action_dim)
    checkpoint = tmp_path / f"{name}.pt"
    agent.save(checkpoint)
    restored = build_agent(config, tiny_dataset.observation_dim, tiny_dataset.action_dim, "cpu")
    restored.load(checkpoint)
    assert np.allclose(action, restored.act(tiny_dataset.observations[0]), atol=1e-6)


def test_reproducible_initialization(tiny_dataset: TransitionDataset) -> None:
    config = {"algorithm": {"name": "bc", "hidden_dims": [8, 8]}}
    set_seed(19)
    first = build_agent(config, tiny_dataset.observation_dim, tiny_dataset.action_dim, "cpu")
    first_action = first.act(tiny_dataset.observations[0])
    set_seed(19)
    second = build_agent(config, tiny_dataset.observation_dim, tiny_dataset.action_dim, "cpu")
    assert np.array_equal(first_action, second.act(tiny_dataset.observations[0]))
