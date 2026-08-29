from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    method: str = "bca",
) -> dict[str, float | int | str]:
    """One-sample percentile or bias-corrected accelerated bootstrap interval."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("Bootstrap input must be a finite vector with at least two values")
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("Invalid confidence level or too few bootstrap resamples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    bootstrap = np.asarray([statistic(sample) for sample in data[indices]], dtype=np.float64)
    estimate = float(statistic(data))
    alpha = 1.0 - confidence
    if method == "percentile":
        low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    elif method == "bca":
        proportion = np.clip(np.mean(bootstrap < estimate), 1e-6, 1.0 - 1e-6)
        z0 = norm.ppf(proportion)
        jackknife = np.asarray(
            [statistic(np.delete(data, index)) for index in range(data.size)], dtype=np.float64
        )
        mean_jackknife = np.mean(jackknife)
        numerator = np.sum((mean_jackknife - jackknife) ** 3)
        denominator = 6.0 * np.sum((mean_jackknife - jackknife) ** 2) ** 1.5
        acceleration = numerator / denominator if denominator > 0 else 0.0
        probabilities = []
        for tail in (alpha / 2.0, 1.0 - alpha / 2.0):
            z = norm.ppf(tail)
            adjusted = norm.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z)))
            probabilities.append(float(np.clip(adjusted, 0.0, 1.0)))
        low, high = np.quantile(bootstrap, probabilities)
    else:
        raise ValueError("method must be 'percentile' or 'bca'")
    return {
        "estimate": estimate,
        "lower": float(low),
        "upper": float(high),
        "confidence": confidence,
        "resamples": resamples,
        "method": method,
    }


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    **kwargs: Any,
) -> dict[str, float | int | str]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("Paired arrays must have identical shapes")
    return bootstrap_ci(left_array - right_array, **kwargs)
