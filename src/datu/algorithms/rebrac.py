from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from datu.algorithms.common import (
    DeterministicActor,
    OfflineAgent,
    TwinQ,
    batch_to_device,
    load_torch_checkpoint,
    resolve_device,
    set_requires_grad,
    soft_update,
)


class ReBRACAgent(OfflineAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        actor_bc_coef: float = 1.0,
        critic_bc_coef: float = 0.1,
        policy_delay: int = 2,
        device: str = "auto",
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = resolve_device(device)
        self.actor = DeterministicActor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinQ(observation_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef
        self.policy_delay = policy_delay
        self.total_steps = 0
        self.config = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "discount": discount,
            "tau": tau,
            "policy_noise": policy_noise,
            "noise_clip": noise_clip,
            "actor_bc_coef": actor_bc_coef,
            "critic_bc_coef": critic_bc_coef,
            "policy_delay": policy_delay,
        }

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        tensors = batch_to_device(batch, self.device)
        observations = tensors["observations"]
        actions = tensors["actions"].clamp(-1.0, 1.0)
        next_dataset_actions = tensors.get("next_actions", actions).clamp(-1.0, 1.0)
        with torch.no_grad():
            noise = (
                torch.randn_like(actions)
                .mul(self.policy_noise)
                .clamp(-self.noise_clip, self.noise_clip)
            )
            next_actions = (self.target_actor(tensors["next_observations"]) + noise).clamp(
                -1.0, 1.0
            )
            target_q = self.target_critic.minimum(tensors["next_observations"], next_actions)
            behavior_penalty = (next_actions - next_dataset_actions).square().sum(-1, keepdim=True)
            target = tensors["rewards"] + self.discount * (1.0 - tensors["dones"]) * (
                target_q - self.critic_bc_coef * behavior_penalty
            )
        q1, q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.total_steps += 1
        actor_loss_value = 0.0
        actor_updated = 0.0
        if self.total_steps % self.policy_delay == 0:
            set_requires_grad(self.critic, False)
            policy_actions = self.actor(observations)
            q = self.critic.q1(torch.cat([observations, policy_actions], dim=-1))
            q_scale = q.abs().mean().detach().clamp(min=1e-6)
            bc_loss = F.mse_loss(policy_actions, actions)
            actor_loss = -q.mean() / q_scale + self.actor_bc_coef * bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            set_requires_grad(self.critic, True)
            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)
            actor_loss_value = float(actor_loss.detach())
            actor_updated = 1.0
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": actor_loss_value,
            "actor_updated": actor_updated,
            "behavior_penalty": float(behavior_penalty.mean()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach()),
        }

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor, single = self._observation_tensor(observation)
        action = self.actor(tensor)
        if not deterministic:
            action = (action + 0.1 * torch.randn_like(action)).clamp(-1.0, 1.0)
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
                "algorithm": "rebrac",
                "config": self.config,
                "total_steps": self.total_steps,
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(path, self.device)
        if checkpoint.get("algorithm") != "rebrac":
            raise ValueError("Checkpoint is not a ReBRAC checkpoint")
        for key, module in (
            ("actor", self.actor),
            ("target_actor", self.target_actor),
            ("critic", self.critic),
            ("target_critic", self.target_critic),
        ):
            module.load_state_dict(checkpoint[key])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = int(checkpoint["total_steps"])
