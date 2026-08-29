from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class DATUConfig:
    use_sensitivity: bool = True
    use_uncertainty: bool = True
    use_local_support: bool = True
    combination: str = "product"
    aggregation: str = "topk_mean"
    topk_fraction: float = 0.10
    percentile: float = 90.0
    temporal_discount: float = 0.99


def robust_normalize(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("Cannot normalize non-finite DATU components")
    low, high = np.percentile(array, [lower, upper])
    if high - low <= 1e-12:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def combine_components(components: Mapping[str, np.ndarray], config: DATUConfig) -> np.ndarray:
    enabled = []
    if config.use_sensitivity:
        enabled.append(robust_normalize(components["sensitivity"]))
    if config.use_uncertainty:
        enabled.append(robust_normalize(components["uncertainty"]))
    if config.use_local_support:
        enabled.append(robust_normalize(components["local_support"]))
    if not enabled:
        raise ValueError("At least one DATU component must be enabled")
    stacked = np.stack(enabled)
    if config.combination == "product":
        return np.prod(stacked, axis=0)
    if config.combination == "geometric_mean":
        return np.prod(np.maximum(stacked, 1e-12), axis=0) ** (1.0 / len(enabled))
    if config.combination == "mean":
        return np.mean(stacked, axis=0)
    if config.combination == "sum":
        return np.sum(stacked, axis=0)
    raise ValueError(f"Unknown DATU combination: {config.combination}")


def aggregate_values(values: np.ndarray, config: DATUConfig) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot aggregate an empty trajectory/group")
    if config.aggregation == "mean":
        return float(np.mean(array))
    if config.aggregation == "sum":
        return float(np.sum(array))
    if config.aggregation == "max":
        return float(np.max(array))
    if config.aggregation == "percentile":
        return float(np.percentile(array, config.percentile))
    if config.aggregation == "topk_mean":
        if not 0.0 < config.topk_fraction <= 1.0:
            raise ValueError("topk_fraction must lie in (0,1]")
        count = max(1, int(np.ceil(array.size * config.topk_fraction)))
        return float(np.mean(np.partition(array, -count)[-count:]))
    if config.aggregation == "discounted":
        if not 0.0 < config.temporal_discount <= 1.0:
            raise ValueError("temporal_discount must lie in (0,1]")
        weights = config.temporal_discount ** np.arange(array.size)
        return float(np.sum(weights * array) / np.sum(weights))
    raise ValueError(f"Unknown trajectory aggregation: {config.aggregation}")


def compute_datu_scores(
    components: Mapping[str, np.ndarray],
    trajectory_offsets: np.ndarray,
    config: DATUConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if config is None:
        config = DATUConfig()
    point_scores = combine_components(components, config)
    offsets = np.asarray(trajectory_offsets, dtype=np.int64)
    if offsets[0] != 0 or offsets[-1] != len(point_scores) or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid trajectory offsets for DATU point scores")
    trajectory_scores = np.asarray(
        [aggregate_values(point_scores[start:stop], config) for start, stop in pairwise(offsets)]
    )
    return point_scores, trajectory_scores


def datu_variants() -> dict[str, DATUConfig]:
    return {
        "datu_full": DATUConfig(),
        "datu_no_uncertainty": DATUConfig(use_uncertainty=False),
        "datu_no_sensitivity": DATUConfig(use_sensitivity=False),
        "datu_no_local_support": DATUConfig(use_local_support=False),
        "uncertainty_only": DATUConfig(False, True, False),
        "sensitivity_only": DATUConfig(True, False, False),
        "coverage_only": DATUConfig(False, False, True),
    }
