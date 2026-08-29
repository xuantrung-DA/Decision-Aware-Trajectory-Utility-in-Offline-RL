from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def correlation_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("Correlation inputs must be aligned vectors of length >= 3")
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
    }


def classification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != labels.shape or len(np.unique(labels)) != 2:
        raise ValueError("AUROC/AUPRC require aligned scores with both binary classes present")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "prevalence": float(np.mean(labels)),
    }


def paired_metric_bootstrap(
    predicted: np.ndarray,
    target: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    stratify: np.ndarray | None = None,
) -> dict[str, Any]:
    x = np.asarray(predicted)
    y = np.asarray(target)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Metric bootstrap inputs must be aligned vectors")
    rng = np.random.default_rng(seed)
    values = []
    if stratify is None:
        for _ in range(resamples):
            indices = rng.integers(0, len(x), len(x))
            value = metric(x[indices], y[indices])
            if np.isfinite(value):
                values.append(value)
    else:
        strata = np.asarray(stratify)
        unique = np.unique(strata)
        pools = [np.flatnonzero(strata == value) for value in unique]
        for _ in range(resamples):
            indices = np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])
            value = metric(x[indices], y[indices])
            if np.isfinite(value):
                values.append(value)
    if len(values) < resamples // 2:
        raise RuntimeError("Too many invalid bootstrap metric replicates")
    alpha = 1.0 - confidence
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": float(metric(x, y)),
        "lower": float(low),
        "upper": float(high),
        "confidence": confidence,
        "valid_resamples": len(values),
    }
