from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import ClassVar

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import _bootstrap  # noqa: F401
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from datu.algorithms import build_agent
from datu.datasets.matched_builder import CriticalRegion, MatchedDatasetBuilder
from datu.datasets.trajectory import Trajectory, TransitionDataset
from datu.evaluation.dataset_metrics import compute_dataset_metrics
from datu.evaluation.plotting import plot_pointmaze_occupancy
from datu.evaluation.policy_eval import evaluate_policy
from datu.utility.datu import DATUConfig, compute_datu_scores
from datu.utility.decision_sensitivity import (
    CandidateActionGenerator,
    decision_sensitivity,
    evaluate_candidate_q,
)
from datu.utility.local_coverage import group_support_contribution
from datu.utility.uncertainty import BootstrappedIQLCriticEnsemble, q_uncertainty
from datu.utils.io import atomic_json_dump
from datu.utils.seed import set_seed


class TinyContinuousEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, horizon: int = 15) -> None:
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.horizon = horizon
        self.state = np.zeros(3, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.state = np.asarray([0.7, -0.7, 0.25], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {"success": False}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.state[:2] = np.clip(self.state[:2] + 0.15 * action, -2.0, 2.0)
        self.state[2] *= 0.9
        self.steps += 1
        success = bool(np.linalg.norm(self.state[:2]) < 0.15)
        reward = float(-np.linalg.norm(self.state[:2]) + success)
        return self.state.copy(), reward, success, self.steps >= self.horizon, {"success": success}


def make_tiny_dataset(seed: int = 0) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    trajectories = []
    horizon = 12
    for episode in range(24):
        if episode < 6:
            x = np.linspace(-1.0, 1.0, horizon)
            y = 0.05 * rng.normal(size=horizon)
            rewards = np.full(horizon, 0.5, dtype=np.float32)
        elif episode < 15:
            x = np.linspace(-0.9, 0.15, horizon)
            y = 0.08 * rng.normal(size=horizon)
            rewards = np.full(horizon, -0.05, dtype=np.float32)
        else:
            theta = np.linspace(0.0, 2.0 * np.pi, horizon)
            x = 0.8 + 0.05 * np.cos(theta)
            y = 0.8 + 0.05 * np.sin(theta)
            rewards = np.full(horizon, -0.05, dtype=np.float32)
        velocity = np.gradient(x)
        observations = np.stack([x, y, velocity], axis=1).astype(np.float32)
        actions = np.stack([np.gradient(x), np.gradient(y)], axis=1)
        actions = np.clip(actions * 5.0 + rng.normal(0.0, 0.02, actions.shape), -1.0, 1.0).astype(
            np.float32
        )
        next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0)
        terminations = np.zeros(horizon, dtype=np.bool_)
        truncations = np.zeros(horizon, dtype=np.bool_)
        truncations[-1] = True
        trajectories.append(
            Trajectory(
                observations,
                actions,
                rewards,
                next_observations,
                terminations,
                truncations,
                f"tiny-{episode:03d}",
            )
        )
    return TransitionDataset.from_trajectories(trajectories, {"source": "built-in smoke fixture"})


def main() -> None:
    seed = 7
    rng = set_seed(seed)
    dataset = make_tiny_dataset(seed)
    output = Path("results/smoke")
    output.mkdir(parents=True, exist_ok=True)
    dataset.save_directory(output / "tiny_dataset")
    reloaded = TransitionDataset.load_directory(output / "tiny_dataset")
    if reloaded.fingerprint() != dataset.fingerprint():
        raise AssertionError("Dataset memory-map round trip changed its hash")

    algorithm_configs = {
        "bc": {"name": "bc", "hidden_dims": [32, 32]},
        "iql": {"name": "iql", "hidden_dims": [32, 32]},
        "cql": {"name": "cql", "hidden_dims": [32, 32], "cql_n_actions": 2},
        "rebrac": {"name": "rebrac", "hidden_dims": [32, 32], "policy_delay": 2},
    }
    rows = []
    trained_agents = {}
    with tempfile.TemporaryDirectory(prefix="datu-smoke-") as temporary:
        checkpoint_root = Path(temporary)
        for name, algorithm_config in algorithm_configs.items():
            agent = build_agent(
                {"algorithm": algorithm_config},
                dataset.observation_dim,
                dataset.action_dim,
                "cpu",
            )
            metrics = {}
            for _ in range(3):
                metrics = agent.train_step(dataset.sample(32, rng, "cpu"))
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"{name} produced non-finite metrics: {metrics}")
            action = agent.act(dataset.observations[0])
            if action.shape != (dataset.action_dim,):
                raise AssertionError(f"{name} returned wrong action shape: {action.shape}")
            checkpoint = checkpoint_root / f"{name}.pt"
            agent.save(checkpoint)
            restored = build_agent(
                {"algorithm": algorithm_config},
                dataset.observation_dim,
                dataset.action_dim,
                "cpu",
            )
            restored.load(checkpoint)
            if not np.allclose(action, restored.act(dataset.observations[0]), atol=1e-6):
                raise AssertionError(f"{name} checkpoint round trip changed deterministic action")
            rows.append({"component": name, "status": "PASS", **metrics})
            trained_agents[name] = agent

    dataset_metrics = compute_dataset_metrics(dataset, seed=seed)
    region = CriticalRegion((0, 1), (-0.2, -0.2), (0.2, 0.2))
    builder = MatchedDatasetBuilder(region, seed)
    conditions, matched_manifest = builder.build(
        dataset,
        base_count=4,
        added_count=3,
        tolerances={
            "max_transition_error": 0.0,
            "max_mean_return_error_fraction": 0.02,
            "max_global_coverage_error_fraction": 1.0,
            "max_median_length_error_fraction": 0.05,
            "min_critical_coverage_separation": 0.01,
        },
    )
    if set(conditions) != {
        "reference_critical_high",
        "critical_low",
        "helpful_low_return",
        "harmful_low_return",
    }:
        raise AssertionError("Matched builder omitted a required condition")
    if len(set(matched_manifest["condition_hashes"].values())) != 4:
        raise AssertionError("Controlled conditions must have four distinct hashes")

    ensemble = BootstrappedIQLCriticEnsemble(
        dataset.observation_dim,
        dataset.action_dim,
        ensemble_size=3,
        hidden_dims=[32, 32],
        device="cpu",
        seed=seed,
    )
    ensemble.fit(dataset, steps=2, batch_size=32)
    generator = CandidateActionGenerator(k=6, perturbations=1).fit(
        dataset.observations, dataset.actions, seed
    )
    candidates = generator.generate(
        dataset.observations, ensemble.policy_actions(dataset.observations)
    )
    candidate_values = evaluate_candidate_q(ensemble.predict, dataset.observations, candidates)
    components = {
        "sensitivity": decision_sensitivity(candidate_values),
        "uncertainty": q_uncertainty(candidate_values)["std"].mean(axis=1),
        "local_support": group_support_contribution(
            dataset.observations,
            dataset.actions,
            dataset.trajectory_ids % 4,
            k=6,
            query_batch_size=64,
        ),
    }
    point_scores, trajectory_scores = compute_datu_scores(
        components, dataset.trajectory_offsets, DATUConfig()
    )
    if not np.isfinite(point_scores).all() or not np.isfinite(trajectory_scores).all():
        raise FloatingPointError("DATU produced non-finite scores")

    env = TinyContinuousEnv()
    try:
        evaluation = evaluate_policy(trained_agents["iql"], env, episodes=1, seed=seed)
    finally:
        env.close()
    figure = plot_pointmaze_occupancy(
        dataset.observations,
        output / "diagnostic.png",
        scores=point_scores,
        title="Smoke DATU diagnostic",
    )
    with (output / "result_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "PASS",
        "algorithms": list(algorithm_configs),
        "dataset_metrics": dataset_metrics,
        "matched_builder": matched_manifest["comparisons"],
        "datu_trajectory_count": len(trajectory_scores),
        "environment_evaluation": evaluation,
        "result_table": str(output / "result_table.csv"),
        "figure": str(figure),
    }
    atomic_json_dump(result, output / "smoke_result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
