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
    batch_to_device,
    load_torch_checkpoint,
    resolve_device,
    set_requires_grad,
    soft_update,
)


class CQLAgent(OfflineAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        entropy_alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        cql_alpha: float = 5.0,
        cql_n_actions: int = 10,
        cql_temperature: float = 1.0,
        device: str = "auto",
    ) -> None:
        if cql_n_actions <= 0 or cql_temperature <= 0:
            raise ValueError("CQL action count and temperature must be positive")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = resolve_device(device)
        self.actor = SquashedGaussianActor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQ(observation_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(entropy_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.discount = discount
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_temperature = cql_temperature
        self.config = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "alpha_lr": alpha_lr,
            "discount": discount,
            "tau": tau,
            "entropy_alpha": entropy_alpha,
            "automatic_entropy_tuning": automatic_entropy_tuning,
            "cql_alpha": cql_alpha,
            "cql_n_actions": cql_n_actions,
            "cql_temperature": cql_temperature,
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def _conservative_penalty(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        dataset_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.shape[0]
        count = self.cql_n_actions
        repeated_obs = (
            observations[:, None, :].expand(-1, count, -1).reshape(batch_size * count, -1)
        )
        repeated_next_obs = (
            next_observations[:, None, :].expand(-1, count, -1).reshape(batch_size * count, -1)
        )
        random_actions = torch.empty(
            batch_size * count, self.action_dim, device=self.device
        ).uniform_(-1.0, 1.0)
        with torch.no_grad():
            current_actions, current_log_probability = self.actor.sample(repeated_obs)
            next_actions, next_log_probability = self.actor.sample(repeated_next_obs)
        uniform_log_density = -float(self.action_dim) * np.log(2.0)
        random_q1, random_q2 = self.critic(repeated_obs, random_actions)
        current_q1, current_q2 = self.critic(repeated_obs, current_actions)
        next_q1, next_q2 = self.critic(repeated_obs, next_actions)
        random_q1 = random_q1 - uniform_log_density
        random_q2 = random_q2 - uniform_log_density
        current_q1 = current_q1 - current_log_probability
        current_q2 = current_q2 - current_log_probability
        next_q1 = next_q1 - next_log_probability
        next_q2 = next_q2 - next_log_probability
        data_q1, data_q2 = self.critic(observations, dataset_actions)

        def penalty(*values: torch.Tensor, data_q: torch.Tensor) -> torch.Tensor:
            stacked = torch.cat([value.reshape(batch_size, count, 1) for value in values], dim=1)
            conservative = self.cql_temperature * torch.logsumexp(
                stacked / self.cql_temperature, dim=1
            )
            return (conservative - data_q).mean()

        return (
            penalty(random_q1, current_q1, next_q1, data_q=data_q1),
            penalty(random_q2, current_q2, next_q2, data_q=data_q2),
        )

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        tensors = batch_to_device(batch, self.device)
        observations = tensors["observations"]
        actions = tensors["actions"].clamp(-1.0, 1.0)
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(tensors["next_observations"])
            next_q = self.target_critic.minimum(tensors["next_observations"], next_actions)
            target = tensors["rewards"] + self.discount * (1.0 - tensors["dones"]) * (
                next_q - self.alpha * next_log_probability
            )
        q1, q2 = self.critic(observations, actions)
        bellman_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        cql1, cql2 = self._conservative_penalty(observations, tensors["next_observations"], actions)
        critic_loss = bellman_loss + self.cql_alpha * (cql1 + cql2)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        set_requires_grad(self.critic, False)
        sampled_actions, log_probability = self.actor.sample(observations)
        actor_q = self.critic.minimum(observations, sampled_actions)
        actor_loss = (self.alpha * log_probability - actor_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        set_requires_grad(self.critic, True)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)
        soft_update(self.critic, self.target_critic, self.tau)

        return {
            "critic_loss": float(critic_loss.detach()),
            "bellman_loss": float(bellman_loss.detach()),
            "cql_penalty": float((cql1 + cql2).detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.alpha),
            "alpha_loss": float(alpha_loss.detach()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach()),
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
                "algorithm": "cql",
                "config": self.config,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(path, self.device)
        if checkpoint.get("algorithm") != "cql":
            raise ValueError("Checkpoint is not a CQL checkpoint")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
