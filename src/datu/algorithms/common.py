from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return resolved


def mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0 or any(width <= 0 for width in hidden_dims):
        raise ValueError("All network dimensions must be positive")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        linear = nn.Linear(previous, width)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ReLU()])
        previous = width
    output = nn.Linear(previous, output_dim)
    nn.init.uniform_(output.weight, -3e-3, 3e-3)
    nn.init.zeros_(output.bias)
    layers.append(output)
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.network = mlp(observation_dim, hidden_dims, action_dim, nn.Tanh)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("Gaussian actor requires at least one hidden layer")
        layers: list[nn.Module] = []
        previous = observation_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(previous, action_dim)
        self.log_std = nn.Linear(previous, action_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)

    def distribution(self, observations: torch.Tensor) -> Normal:
        features = self.trunk(observations)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observations).mean)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        correction = torch.log(1.0 - action.square() + 1e-6)
        log_probability = (distribution.log_prob(raw_action) - correction).sum(-1, keepdim=True)
        return action, log_probability

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        clipped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(clipped)
        distribution = self.distribution(observations)
        correction = torch.log(1.0 - clipped.square() + 1e-6)
        return (distribution.log_prob(raw_action) - correction).sum(-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        input_dim = observation_dim + action_dim
        self.q1 = mlp(input_dim, hidden_dims, 1)
        self.q2 = mlp(input_dim, hidden_dims, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([observations, actions], dim=-1)
        return self.q1(features), self.q2(features)

    def minimum(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(observations, actions)
        return torch.minimum(q1, q2)


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.network = mlp(observation_dim, hidden_dims, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.data.lerp_(source_parameter.data, tau)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def batch_to_device(
    batch: Mapping[str, torch.Tensor | np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        result[key] = tensor.to(device=device, dtype=torch.float32)
    required = {"observations", "actions", "rewards", "next_observations", "dones"}
    missing = required.difference(result)
    if missing:
        raise KeyError(f"Training batch is missing fields: {sorted(missing)}")
    return result


class OfflineAgent(ABC):
    observation_dim: int
    action_dim: int
    device: torch.device

    @abstractmethod
    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        raise RuntimeError("Abstract method")

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise RuntimeError("Abstract method")

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise RuntimeError("Abstract method")

    @abstractmethod
    def load(self, path: str | Path) -> None:
        raise RuntimeError("Abstract method")

    def _observation_tensor(self, observation: np.ndarray) -> tuple[torch.Tensor, bool]:
        array = np.asarray(observation, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array[None]
        if array.ndim != 2 or array.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observation shape [N,{self.observation_dim}], got {array.shape}"
            )
        return torch.as_tensor(array, device=self.device), single


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
