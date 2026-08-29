from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_pointmaze_occupancy(
    observations: np.ndarray,
    path: str | Path,
    scores: np.ndarray | None = None,
    title: str = "PointMaze dataset occupancy",
) -> Path:
    values = np.asarray(observations)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("PointMaze plotting requires at least two observation dimensions")
    fig, axis = plt.subplots(figsize=(6, 5))
    artist = axis.scatter(values[:, 0], values[:, 1], c=scores, s=4, alpha=0.5, cmap="viridis")
    if scores is not None:
        fig.colorbar(artist, ax=axis, label="score")
    axis.set(title=title, xlabel="x", ylabel="y")
    axis.set_aspect("equal", adjustable="box")
    return _save(fig, path)


def plot_trajectory_overlays(
    trajectories: Sequence[np.ndarray], path: str | Path, title: str = "Trajectories"
) -> Path:
    fig, axis = plt.subplots(figsize=(6, 5))
    for trajectory in trajectories:
        values = np.asarray(trajectory)
        axis.plot(values[:, 0], values[:, 1], alpha=0.5, linewidth=1)
    axis.set(title=title, xlabel="x", ylabel="y")
    axis.set_aspect("equal", adjustable="box")
    return _save(fig, path)


def plot_spatial_landscape(
    observations: np.ndarray,
    scores: np.ndarray,
    path: str | Path,
    title: str,
    label: str,
) -> Path:
    return plot_pointmaze_occupancy(observations, path, scores=scores, title=f"{title} ({label})")


def plot_value_landscape(observations: np.ndarray, values: np.ndarray, path: str | Path) -> Path:
    return plot_spatial_landscape(observations, values, path, "Value landscape", "Q/V")


def plot_uncertainty_landscape(
    observations: np.ndarray, uncertainty: np.ndarray, path: str | Path
) -> Path:
    return plot_spatial_landscape(observations, uncertainty, path, "Uncertainty landscape", "Q std")


def plot_sensitivity_landscape(
    observations: np.ndarray, sensitivity: np.ndarray, path: str | Path
) -> Path:
    return plot_spatial_landscape(
        observations, sensitivity, path, "Decision-sensitivity landscape", "P90-P10 Q"
    )


def plot_datu_landscape(observations: np.ndarray, scores: np.ndarray, path: str | Path) -> Path:
    return plot_spatial_landscape(observations, scores, path, "DATU landscape", "DATU")


def plot_score_distribution(
    score_sets: dict[str, np.ndarray], path: str | Path, title: str = "Utility distributions"
) -> Path:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(list(score_sets.values()), bins=30, label=list(score_sets), alpha=0.6)
    axis.set(title=title, xlabel="score", ylabel="count")
    axis.legend()
    return _save(fig, path)


def plot_score_vs_utility(
    score: np.ndarray,
    utility: np.ndarray,
    path: str | Path,
    label: str = "DATU",
) -> Path:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(score, utility, alpha=0.8)
    axis.set(xlabel=label, ylabel="empirical marginal utility", title="Score vs utility")
    return _save(fig, path)


def plot_critical_coverage_vs_performance(
    coverage: np.ndarray, performance: np.ndarray, path: str | Path
) -> Path:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(coverage, performance, alpha=0.8)
    axis.set(
        xlabel="critical-region coverage",
        ylabel="policy performance",
        title="Critical coverage vs performance",
    )
    return _save(fig, path)


def plot_return_vs_datu(returns: np.ndarray, datu_scores: np.ndarray, path: str | Path) -> Path:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(returns, datu_scores, alpha=0.8)
    axis.set(xlabel="trajectory return", ylabel="DATU", title="Return vs DATU")
    return _save(fig, path)


def plot_performance_curve(
    fractions: np.ndarray,
    method_values: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    fig, axis = plt.subplots(figsize=(6, 4))
    for name, values in method_values.items():
        axis.plot(fractions, values, marker="o", label=name)
    axis.set(xlabel="retained data fraction", ylabel="normalized return")
    axis.legend()
    return _save(fig, path)


def plot_correlation_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    path: str | Path,
    title: str = "Algorithm utility correlation",
) -> Path:
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="correlation")
    return _save(fig, path)
