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

# The frozen v1 matrix uses an IQL base training config and then overlays the
# target algorithm. Its recursive merge intentionally preserves shared fields,
# but also leaves these IQL-only fields behind for BC, CQL, and ReBRAC. They
# are inert for those constructors and may be discarded without changing the
# target algorithm's registered hyperparameters. Unknown keys outside this
# explicit compatibility set must still fail fast.
INHERITED_IQL_KEYS = {
    "actor_lr",
    "critic_lr",
    "value_lr",
    "discount",
    "tau",
    "expectile",
    "beta",
    "max_weight",
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
    unexpected = unknown - INHERITED_IQL_KEYS
    if unexpected:
        raise ValueError(f"Unknown {name} config keys: {sorted(unexpected)}")
    algorithm_config = {key: value for key, value in algorithm_config.items() if key in valid}
    return constructor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=device,
        **algorithm_config,
    )
