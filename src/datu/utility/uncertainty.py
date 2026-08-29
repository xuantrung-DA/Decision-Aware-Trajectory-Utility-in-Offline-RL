from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from datu.algorithms.iql import IQLAgent
from datu.datasets.trajectory import TransitionDataset


class QPredictor(Protocol):
    def q_values(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray: ...


def ensemble_q_values(
    critics: Sequence[QPredictor], observations: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    if len(critics) < 2:
        raise ValueError("Epistemic uncertainty requires at least two critics")
    values = np.stack([critic.q_values(observations, actions) for critic in critics], axis=0)
    if not np.isfinite(values).all():
        raise FloatingPointError("Critic ensemble produced non-finite Q values")
    return values


def q_uncertainty(q_values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(q_values, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Q ensemble array must have shape [ensemble>=2, ...]")
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0, ddof=1),
        "variance": np.var(values, axis=0, ddof=1),
    }


def action_ranking_disagreement(candidate_q_values: np.ndarray) -> np.ndarray:
    """Normalized entropy of each ensemble member's best candidate action."""
    values = np.asarray(candidate_q_values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Expected candidate Q shape [ensemble, states, actions]")
    best = np.argmax(values, axis=2)
    result = np.zeros(values.shape[1], dtype=np.float64)
    for state in range(values.shape[1]):
        counts = np.bincount(best[:, state], minlength=values.shape[2]).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        result[state] = (
            entropy / np.log(min(values.shape[0], values.shape[2])) if entropy > 0 else 0.0
        )
    return result


class BootstrappedIQLCriticEnsemble:
    """Small offline-only reference ensemble using independently bootstrapped IQL fits."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (256, 256),
        device: str = "auto",
        seed: int = 0,
        **iql_kwargs: float,
    ) -> None:
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2")
        self.members = [
            IQLAgent(
                observation_dim,
                action_dim,
                hidden_dims=hidden_dims,
                device=device,
                **iql_kwargs,
            )
            for _ in range(ensemble_size)
        ]
        self.rngs = [np.random.default_rng(seed + 1009 * index) for index in range(ensemble_size)]

    def fit(
        self,
        dataset: TransitionDataset,
        steps: int,
        batch_size: int = 256,
        progress_interval: int | None = None,
    ) -> list[dict[str, float]]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if progress_interval is not None and progress_interval <= 0:
            raise ValueError("progress_interval must be positive when supplied")
        final_metrics = []
        overall_start = time.perf_counter()
        completed_updates = 0
        total_updates = len(self.members) * steps
        best_interval_rate = 0.0
        for member_index, (member, rng) in enumerate(
            zip(self.members, self.rngs), start=1
        ):
            metrics: dict[str, float] = {}
            window_start = time.perf_counter()
            window_updates = 0
            sample_seconds = 0.0
            update_seconds = 0.0
            for step in range(1, steps + 1):
                sample_start = time.perf_counter()
                batch = dataset.sample(batch_size, rng, member.device)
                sample_seconds += time.perf_counter() - sample_start
                update_start = time.perf_counter()
                metrics = member.train_step(batch)
                update_seconds += time.perf_counter() - update_start
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(
                        f"Reference critic {member_index} produced non-finite metrics "
                        f"at step {step}: {metrics}"
                    )
                completed_updates += 1
                window_updates += 1
                report = (
                    progress_interval is not None
                    and (step == 1 or step % progress_interval == 0 or step == steps)
                )
                if report:
                    now = time.perf_counter()
                    interval_rate = window_updates / max(now - window_start, 1e-12)
                    overall_rate = completed_updates / max(now - overall_start, 1e-12)
                    eta_seconds = (total_updates - completed_updates) / max(
                        overall_rate, 1e-12
                    )
                    metric_text = " ".join(
                        f"{key}={value:.5g}" for key, value in sorted(metrics.items())
                    )
                    device = torch.device(member.device)
                    memory_text = ""
                    if device.type == "cuda":
                        index = device.index if device.index is not None else torch.cuda.current_device()
                        allocated = torch.cuda.memory_allocated(index) / (1024**2)
                        reserved = torch.cuda.memory_reserved(index) / (1024**2)
                        total = torch.cuda.get_device_properties(index).total_memory / (1024**2)
                        memory_text = (
                            f" vram={allocated:.0f}/{reserved:.0f}/{total:.0f}MB"
                            "(alloc/reserved/total)"
                        )
                    print(
                        f"[reference-progress] member={member_index}/{len(self.members)} "
                        f"step={step:,}/{steps:,} total={completed_updates:,}/{total_updates:,} "
                        f"rate={interval_rate:.1f}upd/s overall={overall_rate:.1f}upd/s "
                        f"eta={eta_seconds / 60.0:.1f}m "
                        f"sample={1000.0 * sample_seconds / window_updates:.2f}ms "
                        f"update={1000.0 * update_seconds / window_updates:.2f}ms"
                        f"{memory_text} | "
                        f"{metric_text}",
                        flush=True,
                    )
                    if best_interval_rate and interval_rate < 0.5 * best_interval_rate:
                        print(
                            f"[warning] reference throughput dropped to {interval_rate:.1f} "
                            f"upd/s from best={best_interval_rate:.1f}; check GPU contention, "
                            "thermal throttling, or paging",
                            flush=True,
                        )
                    best_interval_rate = max(best_interval_rate, interval_rate)
                    window_start = now
                    window_updates = 0
                    sample_seconds = 0.0
                    update_seconds = 0.0
            final_metrics.append(metrics)
        return final_metrics

    def predict(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return ensemble_q_values(self.members, observations, actions)

    def policy_actions(self, observations: np.ndarray) -> np.ndarray:
        return self.members[0].act(observations, deterministic=True)

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(self.members):
            member.save(destination / f"critic_{index}.pt")

    def load(self, directory: str | Path) -> None:
        source = Path(directory)
        paths = sorted(source.glob("critic_*.pt"))
        if len(paths) != len(self.members):
            raise ValueError(
                f"Expected {len(self.members)} cached critics in {source}, found {len(paths)}"
            )
        for member, path in zip(self.members, paths):
            member.load(path)
