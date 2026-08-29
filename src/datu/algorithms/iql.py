from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from datu.algorithms.common import (
    OfflineAgent,
    SquashedGaussianActor,
    TwinQ,
    ValueNetwork,
    batch_to_device,
    load_torch_checkpoint,
    resolve_device,
    soft_update,
)


class IQLAgent(OfflineAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        beta: float = 3.0,
        max_weight: float = 100.0,
        device: str = "auto",
    ) -> None:
        if not 0.0 < expectile < 1.0:
            raise ValueError("expectile must lie strictly between 0 and 1")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = resolve_device(device)
        self.actor = SquashedGaussianActor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQ(observation_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = ValueNetwork(observation_dim, hidden_dims).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.beta = beta
        self.max_weight = max_weight
        self.config = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "value_lr": value_lr,
            "discount": discount,
            "tau": tau,
            "expectile": expectile,
            "beta": beta,
            "max_weight": max_weight,
        }

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        tensors = batch_to_device(batch, self.device)
        observations = tensors["observations"]
        actions = tensors["actions"].clamp(-1.0, 1.0)
        with torch.no_grad():
            target_q = self.target_critic.minimum(observations, actions)
        value = self.value(observations)
        difference = target_q - value
        weights = torch.where(difference > 0.0, self.expectile, 1.0 - self.expectile)
        value_loss = (weights * difference.square()).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            next_value = self.value(tensors["next_observations"])
            target = tensors["rewards"] + self.discount * (1.0 - tensors["dones"]) * next_value
        q1, q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            advantage = self.target_critic.minimum(observations, actions) - self.value(observations)
            advantage_weight = torch.exp(self.beta * advantage).clamp(max=self.max_weight)
        log_probability = self.actor.log_prob(observations, actions)
        actor_loss = -(advantage_weight * log_probability).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        soft_update(self.critic, self.target_critic, self.tau)

        return {
            "value_loss": float(value_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach()),
            "advantage_mean": float(advantage.mean()),
        }

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor, single = self._observation_tensor(observation)
        action = self.actor(tensor) if deterministic else self.actor.sample(tensor)[0]
        result = action.cpu().numpy()
        return result[0] if single else result

    @torch.no_grad()
    def q_values(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return self.target_critic.minimum(obs, act).cpu().numpy().reshape(-1)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "iql",
                "config": self.config,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "value": self.value.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(path, self.device)
        if checkpoint.get("algorithm") != "iql":
            raise ValueError("Checkpoint is not an IQL checkpoint")
        for key, module in (
            ("actor", self.actor),
            ("critic", self.critic),
            ("target_critic", self.target_critic),
            ("value", self.value),
        ):
            module.load_state_dict(checkpoint[key])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
