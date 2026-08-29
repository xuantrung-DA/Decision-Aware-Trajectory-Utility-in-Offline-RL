from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def project_for_neighbors(
    features: np.ndarray,
    max_exact_dimensions: int = 10,
    projected_dimensions: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, Any | None]:
    """Use a seeded random projection before kNN when continuous features are high-dimensional."""
    values = np.asarray(features, dtype=np.float64)
    if values.shape[1] <= max_exact_dimensions:
        return values, None
    dimensions = min(projected_dimensions, values.shape[1])
    projector = GaussianRandomProjection(n_components=dimensions, random_state=seed)
    return projector.fit_transform(values), projector
