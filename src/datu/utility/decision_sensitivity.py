from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from datu.utility.features import project_for_neighbors


class CandidateActionGenerator:
    """Generate in-support continuous actions from state neighbors and small perturbations."""

    def __init__(self, k: int = 50, perturbations: int = 2, perturbation_std: float = 0.05) -> None:
        if k <= 1 or perturbations < 0 or perturbation_std < 0:
            raise ValueError("Invalid candidate-action generator parameters")
        self.k = k
        self.perturbations = perturbations
        self.perturbation_std = perturbation_std
        self.scaler = StandardScaler()
        self.neighbors: NearestNeighbors | None = None
        self.actions: np.ndarray | None = None
        self.rng = np.random.default_rng(0)
        self.projector = None

    def fit(
        self, observations: np.ndarray, actions: np.ndarray, seed: int = 0
    ) -> CandidateActionGenerator:
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if len(observations) != len(actions) or len(observations) < 2:
            raise ValueError("Candidate generation requires aligned offline samples")
        normalized = self.scaler.fit_transform(observations)
        normalized, self.projector = project_for_neighbors(normalized)
        self.neighbors = NearestNeighbors(
            n_neighbors=min(self.k, len(observations)), algorithm="kd_tree", n_jobs=1
        ).fit(normalized)
        self.actions = actions
        self.rng = np.random.default_rng(seed)
        return self

    def generate(
        self,
        observations: np.ndarray,
        policy_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.neighbors is None or self.actions is None:
            raise RuntimeError("CandidateActionGenerator.fit must be called first")
        states = self.scaler.transform(np.asarray(observations, dtype=np.float64))
        if self.projector is not None:
            states = self.projector.transform(states)
        _, indices = self.neighbors.kneighbors(states)
        candidates = self.actions[indices]
        local_low = candidates.min(axis=1)
        local_high = candidates.max(axis=1)
        extras = []
        if policy_actions is not None:
            policy = np.clip(np.asarray(policy_actions), local_low, local_high)
            extras.append(policy[:, None, :])
        center = np.mean(candidates, axis=1)
        for _ in range(self.perturbations):
            perturbed = np.clip(
                center + self.rng.normal(0.0, self.perturbation_std, center.shape),
                local_low,
                local_high,
            )
            extras.append(perturbed[:, None, :])
        if extras:
            candidates = np.concatenate([candidates, *extras], axis=1)
        return candidates.astype(np.float32)


def evaluate_candidate_q(
    q_predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    observations: np.ndarray,
    candidate_actions: np.ndarray,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    candidates = np.asarray(candidate_actions, dtype=np.float32)
    if candidates.ndim != 3 or len(observations) != len(candidates):
        raise ValueError("Candidate actions must have shape [states, candidates, action_dim]")
    repeated_states = np.repeat(observations, candidates.shape[1], axis=0)
    flat_actions = candidates.reshape(-1, candidates.shape[-1])
    values = np.asarray(q_predictor(repeated_states, flat_actions))
    if values.ndim == 1:
        return values.reshape(len(observations), candidates.shape[1])
    if values.ndim == 2:
        return values.reshape(values.shape[0], len(observations), candidates.shape[1])
    raise ValueError("Q predictor must return [N] or [ensemble,N]")


def decision_sensitivity(
    candidate_q_values: np.ndarray,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
) -> np.ndarray:
    values = np.asarray(candidate_q_values, dtype=np.float64)
    if values.ndim == 3:
        values = np.mean(values, axis=0)
    if values.ndim != 2:
        raise ValueError("Candidate Q values must be [states,actions] or [ensemble,states,actions]")
    high = np.percentile(values, upper_percentile, axis=1)
    low = np.percentile(values, lower_percentile, axis=1)
    return np.maximum(high - low, 0.0)
