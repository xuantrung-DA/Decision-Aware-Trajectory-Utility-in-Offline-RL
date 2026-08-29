from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import gymnasium as gym
import gymnasium_robotics
import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from datu.datasets.minari_loader import load_processed_dataset
from datu.utils.io import atomic_json_dump


def _free_cell_graph(maze_map: list[list[Any]]) -> nx.Graph:
    graph = nx.Graph()
    free = [
        (row, column)
        for row, values in enumerate(maze_map)
        for column, value in enumerate(values)
        if value != 1
    ]
    graph.add_nodes_from(free)
    for row, column in free:
        for delta_row, delta_column in ((1, 0), (0, 1)):
            neighbor = (row + delta_row, column + delta_column)
            if neighbor in graph:
                graph.add_edge((row, column), neighbor)
    if not nx.is_connected(graph):
        raise ValueError("PointMaze free-cell graph is unexpectedly disconnected")
    return graph


def _cell_record(
    dataset,
    cell: tuple[int, int],
    center: np.ndarray,
    half_width: float,
    articulation_points: set[tuple[int, int]],
    betweenness: dict[tuple[int, int], float],
    base_count: int,
    required_cohort_count: int,
) -> dict[str, Any]:
    lower = center - half_width
    upper = center + half_width
    positions = dataset.observations[:, :2]
    mask = np.logical_and(positions >= lower, positions <= upper).all(axis=1)
    visit_counts = np.bincount(
        dataset.trajectory_ids,
        weights=mask.astype(np.int64),
        minlength=dataset.num_trajectories,
    )
    visit_fractions = visit_counts / dataset.trajectory_lengths
    returns = dataset.trajectory_returns
    reward_rates = returns / dataset.trajectory_lengths
    low_reward_density = reward_rates <= np.median(reward_rates)

    ranked_base = sorted(
        range(dataset.num_trajectories),
        key=lambda index: (returns[index], visit_counts[index]),
        reverse=True,
    )
    remaining = np.ones(dataset.num_trajectories, dtype=np.bool_)
    remaining[np.asarray(ranked_base[:base_count], dtype=np.int64)] = False
    helpful = remaining & low_reward_density & (visit_counts > 0)
    harmful = remaining & low_reward_density & (visit_counts == 0)
    visitor_fractions = visit_fractions[visit_counts > 0]
    return {
        "row": cell[0],
        "column": cell[1],
        "center": center.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "articulation_point": cell in articulation_points,
        "graph_betweenness": float(betweenness[cell]),
        "transition_count": int(np.sum(mask)),
        "transition_fraction": float(np.mean(mask)),
        "visitor_trajectories": int(np.sum(visit_counts > 0)),
        "visitor_trajectory_fraction": float(np.mean(visit_counts > 0)),
        "visitor_mean_within_trajectory_fraction": (
            float(np.mean(visitor_fractions)) if len(visitor_fractions) else 0.0
        ),
        "visitor_p90_within_trajectory_fraction": (
            float(np.quantile(visitor_fractions, 0.9)) if len(visitor_fractions) else 0.0
        ),
        "eligible_helpful_after_base": int(np.sum(helpful)),
        "eligible_harmful_after_base": int(np.sum(harmful)),
        "cohort_feasible": bool(
            np.sum(helpful) >= required_cohort_count
            and np.sum(harmful) >= required_cohort_count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PointMaze topology and offline visitation before freezing a region"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-id", default="PointMaze_Medium-v3")
    parser.add_argument("--half-width", type=float, default=0.45)
    parser.add_argument("--base-count", type=int, default=100)
    parser.add_argument("--added-count", type=int, default=40)
    parser.add_argument("--plot-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.half_width < 0.5:
        raise ValueError("half-width must isolate one unit maze cell")

    dataset = load_processed_dataset(args.dataset)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id)
    try:
        maze = env.unwrapped.maze
        maze_map = [list(row) for row in maze.maze_map]
        graph = _free_cell_graph(maze_map)
        articulation_points = set(nx.articulation_points(graph))
        betweenness = nx.betweenness_centrality(graph)
        required = 3 * args.added_count
        records = [
            _cell_record(
                dataset,
                cell,
                np.asarray(maze.cell_rowcol_to_xy(cell), dtype=np.float64),
                args.half_width,
                articulation_points,
                betweenness,
                args.base_count,
                required,
            )
            for cell in graph.nodes
        ]
    finally:
        env.close()

    candidates = [
        record
        for record in records
        if record["articulation_point"] and record["cohort_feasible"]
    ]
    if not candidates:
        raise RuntimeError("No topology-defined region can supply the frozen cohort counts")
    selected = max(candidates, key=lambda item: item["graph_betweenness"])
    audit = {
        "version": 1,
        "dataset_hash": dataset.fingerprint(),
        "dataset_transitions": len(dataset),
        "dataset_trajectories": dataset.num_trajectories,
        "selection_inputs": (
            "environment topology and offline visitation only; no trained-policy outcomes"
        ),
        "selection_rule": (
            "highest graph-betweenness articulation cell with at least 3*added_count "
            "eligible visiting and non-visiting low-reward-density trajectories after base"
        ),
        "sparse_reward_note": (
            "Episodes are goal-delimited and almost all return 1; low-return is operationalized "
            "as at-or-below-median reward density (return divided by length)."
        ),
        "half_width": args.half_width,
        "base_count": args.base_count,
        "added_count": args.added_count,
        "required_per_candidate_pool": required,
        "selected_region": selected,
        "cell_audit": sorted(
            records, key=lambda item: item["graph_betweenness"], reverse=True
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(audit, output / "critical_region_audit.json")

    positions = dataset.observations[:, :2]
    rng = np.random.default_rng(args.seed)
    indices = (
        rng.choice(len(positions), args.plot_samples, replace=False)
        if len(positions) > args.plot_samples
        else np.arange(len(positions))
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.hexbin(positions[indices, 0], positions[indices, 1], gridsize=70, mincnt=1)
    for record in records:
        center = np.asarray(record["center"])
        color = "tab:red" if record is selected else "white"
        axis.add_patch(
            Rectangle(
                center - 0.5,
                1.0,
                1.0,
                fill=False,
                linewidth=2.5 if record is selected else 0.6,
                edgecolor=color,
            )
        )
    lower = np.asarray(selected["lower"])
    axis.add_patch(
        Rectangle(
            lower,
            2 * args.half_width,
            2 * args.half_width,
            fill=False,
            linewidth=3,
            edgecolor="tab:red",
            label="selected critical region",
        )
    )
    axis.set(xlabel="x", ylabel="y", title="PointMaze offline occupancy and topology selection")
    axis.set_aspect("equal")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "critical_region_audit.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"selected_region": selected, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
