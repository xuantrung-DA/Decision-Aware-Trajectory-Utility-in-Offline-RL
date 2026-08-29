from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _flat_single_observation(observation: Any, keys: list[str] | None) -> np.ndarray:
    if isinstance(observation, Mapping):
        selected = keys if keys is not None else sorted(observation)
        return np.concatenate(
            [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in selected]
        )
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _in_critical_region(
    observation: np.ndarray, specification: Mapping[str, Any] | None
) -> bool:
    if not specification:
        return False
    dimensions = np.asarray(specification["observation_dims"], dtype=np.int64)
    lower = np.asarray(specification["lower"], dtype=np.float32)
    upper = np.asarray(specification["upper"], dtype=np.float32)
    if dimensions.ndim != 1 or not (len(dimensions) == len(lower) == len(upper)):
        raise ValueError("Invalid critical-region evaluation specification")
    if np.any(dimensions < 0) or np.any(dimensions >= len(observation)):
        raise ValueError("Critical-region dimensions exceed flattened observation size")
    selected = observation[dimensions]
    return bool(np.all((selected >= lower) & (selected <= upper)))


def evaluate_policy(
    agent: Any,
    env: Any,
    episodes: int = 20,
    seed: int = 10_000,
    observation_keys: list[str] | None = None,
    normalized_score: tuple[float, float] | None = None,
    deterministic: bool = True,
    max_episode_steps: int | None = None,
    critical_region: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_episode_steps is not None and max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive when supplied")
    returns, lengths, successes = [], [], []
    critical_fractions: list[float] = []
    horizon_truncations = 0
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        length = 0
        critical_visits = 0
        success = bool(info.get("success", info.get("is_success", False)))
        while True:
            flat = _flat_single_observation(observation, observation_keys)
            critical_visits += int(_in_critical_region(flat, critical_region))
            action = np.asarray(agent.act(flat, deterministic=deterministic), dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            success = success or bool(info.get("success", info.get("is_success", False)))
            if terminated or truncated:
                break
            if max_episode_steps is not None and length >= max_episode_steps:
                horizon_truncations += 1
                break
        returns.append(total_reward)
        lengths.append(length)
        successes.append(float(success))
        critical_fractions.append(critical_visits / max(length, 1))
    returns_array = np.asarray(returns, dtype=np.float64)
    result: dict[str, Any] = {
        "episodic_returns": returns,
        "episode_lengths": lengths,
        "return_mean": float(np.mean(returns_array)),
        "return_std": float(np.std(returns_array)),
        "success_rate": float(np.mean(successes)),
        "episodes": episodes,
        "evaluation_seed": seed,
        "max_episode_steps": max_episode_steps,
        "horizon_truncations": horizon_truncations,
    }
    if critical_region is not None:
        result.update(
            {
                "critical_visit_fractions": critical_fractions,
                "critical_visit_fraction_mean": float(np.mean(critical_fractions)),
                "critical_episode_fraction": float(
                    np.mean(np.asarray(critical_fractions) > 0.0)
                ),
            }
        )
    if normalized_score is not None:
        low, high = normalized_score
        if high <= low:
            raise ValueError("Normalized-score maximum must exceed minimum")
        result["normalized_return_mean"] = float(
            100.0 * (np.mean(returns_array) - low) / (high - low)
        )
    return result
