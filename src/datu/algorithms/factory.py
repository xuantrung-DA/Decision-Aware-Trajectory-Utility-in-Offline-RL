from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from datu.algorithms.bc import BCAgent
from datu.algorithms.cql import CQLAgent
from datu.algorithms.iql import IQLAgent
from datu.algorithms.rebrac import ReBRACAgent

AGENTS = {
    "bc": BCAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "rebrac": ReBRACAgent,
}


def build_agent(
    config: Mapping[str, Any],
    observation_dim: int,
    action_dim: int,
    device: str = "auto",
) -> BCAgent | IQLAgent | CQLAgent | ReBRACAgent:
    algorithm_config = dict(config.get("algorithm", config))
    name = str(algorithm_config.pop("name", "")).lower()
    if name not in AGENTS:
        raise ValueError(f"Unknown algorithm {name!r}; expected one of {sorted(AGENTS)}")
    constructor = AGENTS[name]
    valid = set(inspect.signature(constructor.__init__).parameters) - {"self"}
    unknown = set(algorithm_config) - valid
    if unknown:
        raise ValueError(f"Unknown {name} config keys: {sorted(unknown)}")
    return constructor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=device,
        **algorithm_config,
    )
