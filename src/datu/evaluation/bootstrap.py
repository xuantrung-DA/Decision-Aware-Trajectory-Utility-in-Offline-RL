from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    method: str = "bca",
) -> dict[str, float | int | str]:
    """One-sample percentile or bias-corrected accelerated bootstrap interval.

    SciPy is the reference implementation. Delegating BCa construction avoids
    unstable hand-written bias correction when a small, discrete training-seed
    bootstrap distribution contains many ties.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("Bootstrap input must be a finite vector with at least two values")
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("Invalid confidence level or too few bootstrap resamples")
    estimate = float(statistic(data))
    if method not in {"percentile", "bca"}:
        raise ValueError("method must be 'percentile' or 'bca'")

    # SciPy's BCa interval is undefined for a degenerate distribution. The
    # interval for an exactly constant statistic is nevertheless unambiguous.
    if np.all(data == data[0]):
        low = high = estimate
    else:
        result = scipy_bootstrap(
            (data,),
            statistic,
            confidence_level=confidence,
            n_resamples=resamples,
            method="BCa" if method == "bca" else "percentile",
            random_state=np.random.default_rng(seed),
            vectorized=False,
        )
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
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
