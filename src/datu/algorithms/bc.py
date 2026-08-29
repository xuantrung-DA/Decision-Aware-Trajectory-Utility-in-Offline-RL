from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from datu.algorithms.common import (
    DeterministicActor,
    OfflineAgent,
    batch_to_device,
    load_torch_checkpoint,
    resolve_device,
)


class BCAgent(OfflineAgent):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        device: str = "auto",
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = resolve_device(device)
        self.actor = DeterministicActor(observation_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.config = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        tensors = batch_to_device(batch, self.device)
        prediction = self.actor(tensors["observations"])
        if prediction.shape != tensors["actions"].shape:
            raise ValueError("Actor output and action batch shapes differ")
        loss = F.mse_loss(prediction, tensors["actions"])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"bc_loss": float(loss.detach())}

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        tensor, single = self._observation_tensor(observation)
        action = self.actor(tensor).cpu().numpy()
        return action[0] if single else action

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "bc",
                "config": self.config,
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = load_torch_checkpoint(path, self.device)
        if checkpoint.get("algorithm") != "bc":
            raise ValueError("Checkpoint is not a BC checkpoint")
        self.actor.load_state_dict(checkpoint["actor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
