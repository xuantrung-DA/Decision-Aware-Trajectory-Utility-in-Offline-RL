from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from datu.evaluation.bootstrap import bootstrap_ci
from datu.evaluation.marginal_utility import summarize_empirical_marginal_utility
from datu.utils.io import atomic_json_dump


def flatten_manifest(data: dict[str, Any], path: Path) -> dict[str, Any]:
    evaluation = data.get("evaluation") or {}
    return {
        "run_dir": str(path.parent),
        "experiment_id": data.get("experiment_id"),
        "experiment_block": data.get("experiment_block"),
        "experiment_variant": data.get("experiment_variant"),
        "dataset_condition": data.get("dataset_condition"),
        "removed_group": data.get("removed_group"),
        "algorithm": data.get("algorithm"),
        "seed": data.get("seed"),
        "steps": data.get("steps"),
        "environment": data.get("environment"),
        "dataset_id": data.get("dataset_id"),
        "dataset_hash": data.get("dataset_hash"),
        "runtime_seconds": data.get("runtime_seconds"),
        "return_mean": evaluation.get("return_mean"),
        "normalized_return_mean": evaluation.get("normalized_return_mean"),
        "success_rate": evaluation.get("success_rate"),
        "critical_visit_fraction_mean": evaluation.get("critical_visit_fraction_mean"),
        "critical_episode_fraction": evaluation.get("critical_episode_fraction"),
        "evaluation_horizon": evaluation.get("max_episode_steps"),
    }


def summarize(runs: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    group_columns = [
        "experiment_id",
        "experiment_block",
        "algorithm",
        "dataset_condition",
        "removed_group",
    ]
    for keys, frame in runs.groupby(group_columns, dropna=False):
        for metric in (
            "return_mean",
            "normalized_return_mean",
            "success_rate",
            "critical_visit_fraction_mean",
            "runtime_seconds",
        ):
            values = frame[metric].dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            interval = bootstrap_ci(values, resamples=resamples, seed=seed)
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "ci_lower": interval["lower"],
                    "ci_upper": interval["upper"],
                    "small_n_warning": len(values) < 5,
                }
            )
    return pd.DataFrame(rows)


def _difference_report(
    differences: np.ndarray,
    seeds: list[int],
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "n_seeds": len(values),
        "seeds": seeds,
        "individual_paired_differences": values.tolist(),
        "mean_paired_difference": float(np.mean(values)) if len(values) else None,
        "std_paired_difference": standard_deviation if len(values) else None,
        "paired_cohen_d": (
            float(np.mean(values) / standard_deviation)
            if len(values) and standard_deviation > 0
            else None
        ),
        "positive_direction_count": int(np.sum(values > 0.0)),
        "bootstrap_95_ci": (
            bootstrap_ci(values, resamples=resamples, seed=bootstrap_seed)
            if len(values) >= 2
            else None
        ),
        "small_n_warning": len(values) < 5,
    }


def paired_condition_comparison(
    runs: pd.DataFrame,
    algorithm: str,
    left: str,
    right: str,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    frame = runs[
        (runs["algorithm"] == algorithm)
        & runs["dataset_condition"].isin([left, right])
        & runs[metric].notna()
    ]
    pivot = frame.pivot_table(index="seed", columns="dataset_condition", values=metric)
    if left not in pivot or right not in pivot:
        return _difference_report(np.asarray([]), [], resamples, seed)
    pivot = pivot[[left, right]].dropna().sort_index()
    return _difference_report(
        (pivot[left] - pivot[right]).to_numpy(dtype=float),
        [int(value) for value in pivot.index],
        resamples,
        seed,
    )


def paired_marginal_utility_rows(
    runs: pd.DataFrame, metric: str = "success_rate"
) -> pd.DataFrame:
    full = runs[
        (runs["dataset_condition"] == "reference_critical_high")
        & runs[metric].notna()
    ][["algorithm", "seed", metric]].rename(columns={metric: "full_performance"})
    removed = runs[runs["removed_group"].notna() & runs[metric].notna()][
        ["algorithm", "seed", "removed_group", metric]
    ].rename(columns={metric: "removed_performance", "removed_group": "group"})
    merged = removed.merge(full, on=["algorithm", "seed"], how="left", validate="many_to_one")
    if not merged.empty and merged["full_performance"].isna().any():
        missing = merged[merged["full_performance"].isna()][["algorithm", "seed"]]
        raise ValueError(f"Block B has no paired full reference runs: {missing.to_dict('records')}")
    merged["marginal_utility"] = merged["full_performance"] - merged["removed_performance"]
    return merged[
        [
            "algorithm",
            "group",
            "seed",
            "full_performance",
            "removed_performance",
            "marginal_utility",
        ]
    ]


def _marginal_pair(
    marginal: pd.DataFrame,
    algorithm: str,
    left: str,
    right: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    frame = marginal[
        (marginal["algorithm"] == algorithm) & marginal["group"].isin([left, right])
    ]
    pivot = frame.pivot_table(index="seed", columns="group", values="marginal_utility")
    if left not in pivot or right not in pivot:
        return _difference_report(np.asarray([]), [], resamples, seed)
    pivot = pivot[[left, right]].dropna().sort_index()
    return _difference_report(
        (pivot[left] - pivot[right]).to_numpy(dtype=float),
        [int(value) for value in pivot.index],
        resamples,
        seed,
    )


def _baseline_pairwise_pass(root: Path) -> dict[str, Any]:
    files = list(root.rglob("utility_validation.csv"))
    if not files:
        return {"satisfied": None, "reason": "utility_validation.csv is absent"}
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if not {"method", "pairwise_accuracy"}.issubset(frame):
        return {"satisfied": None, "reason": "required validation columns are absent"}
    means = frame.groupby("method")["pairwise_accuracy"].mean().to_dict()
    baselines = ("reward_density", "rarity", "global_coverage")
    if "datu" not in means or any(name not in means for name in baselines):
        return {"satisfied": None, "method_means": means}
    return {
        "satisfied": bool(all(means["datu"] > means[name] for name in baselines)),
        "method_means": means,
    }


def _component_ablation_pass(root: Path) -> dict[str, Any]:
    files = list(root.rglob("component_ablation.csv"))
    if not files:
        return {"satisfied": None, "reason": "component_ablation.csv is absent"}
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if not {"method", "pairwise_accuracy"}.issubset(frame):
        return {"satisfied": None, "reason": "required ablation columns are absent"}
    means = frame.groupby("method")["pairwise_accuracy"].mean().to_dict()
    full = means.get("datu_full")
    variants = (
        "datu_no_uncertainty",
        "datu_no_sensitivity",
        "datu_no_local_support",
    )
    if full is None or any(name not in means for name in variants):
        return {"satisfied": None, "method_means": means}
    wins = sum(full > means[name] for name in variants)
    return {
        "satisfied": bool(wins >= 2),
        "full_wins": int(wins),
        "required_wins": 2,
        "method_means": means,
    }


def success_criteria(
    root: Path,
    runs: pd.DataFrame,
    marginal: pd.DataFrame,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    c1_iql = paired_condition_comparison(
        runs,
        "iql",
        "reference_critical_high",
        "critical_low",
        "success_rate",
        resamples,
        seed,
    )
    c1_rebrac = paired_condition_comparison(
        runs,
        "rebrac",
        "reference_critical_high",
        "critical_low",
        "success_rate",
        resamples,
        seed,
    )
    c3_iql = paired_condition_comparison(
        runs,
        "iql",
        "helpful_low_return",
        "harmful_low_return",
        "success_rate",
        resamples,
        seed,
    )
    c3_critical = paired_condition_comparison(
        runs,
        "iql",
        "helpful_low_return",
        "harmful_low_return",
        "critical_visit_fraction_mean",
        resamples,
        seed,
    )
    c2_datu = _marginal_pair(
        marginal,
        "iql",
        "remove_top_datu",
        "remove_bottom_datu",
        resamples,
        seed,
    )
    c2_controlled = _marginal_pair(
        marginal,
        "iql",
        "remove_helpful_controlled",
        "remove_harmful_controlled",
        resamples,
        seed,
    )
    baseline = _baseline_pairwise_pass(root)
    ablation = _component_ablation_pass(root)

    c1_effect = c1_iql["paired_cohen_d"]
    c1_complete = c1_iql["n_seeds"] >= 5
    c1_signal = (
        c1_iql["mean_paired_difference"] is not None
        and (
            c1_iql["mean_paired_difference"] >= 0.10
            or (c1_effect is not None and abs(c1_effect) >= 0.8)
        )
        and c1_iql["positive_direction_count"] >= 4
    )
    c3_complete = c3_iql["n_seeds"] >= 5 and c3_critical["n_seeds"] >= 5
    c3_signal = (
        c3_iql["mean_paired_difference"] is not None
        and c3_iql["mean_paired_difference"] >= 0.10
        and c3_iql["positive_direction_count"] >= 4
        and c3_critical["mean_paired_difference"] is not None
        and c3_critical["mean_paired_difference"] > 0.0
    )
    intervention_complete = c2_datu["n_seeds"] >= 3 and c2_controlled["n_seeds"] >= 3
    intervention_pass = (
        c2_datu["mean_paired_difference"] is not None
        and c2_datu["mean_paired_difference"] > 0.0
        and c2_controlled["mean_paired_difference"] is not None
        and c2_controlled["mean_paired_difference"] > 0.0
    )
    c2_complete = intervention_complete and baseline["satisfied"] is not None
    c4_complete = c1_complete and c1_rebrac["n_seeds"] >= 3
    c4_signal = c1_signal and (
        c1_rebrac["mean_paired_difference"] is not None
        and c1_rebrac["mean_paired_difference"] > 0.0
        and c1_rebrac["positive_direction_count"] >= 2
    )
    return {
        "C1_critical_support": {
            "target": "IQL gap >= 10pp or |paired d| >= 0.8, with >=4/5 positive seeds",
            "satisfied": bool(c1_signal) if c1_complete else None,
            "strong_pass": (
                bool(c1_iql["mean_paired_difference"] >= 0.15)
                if c1_complete and c1_iql["mean_paired_difference"] is not None
                else None
            ),
            "iql_primary": c1_iql,
            "rebrac_secondary": c1_rebrac,
        },
        "C2_datu_marginal_utility": {
            "target": "paired intervention ordering plus DATU pairwise accuracy above three baselines",
            "satisfied": bool(intervention_pass and baseline["satisfied"]) if c2_complete else None,
            "intervention_ordering_pass": bool(intervention_pass) if intervention_complete else None,
            "top_vs_bottom_datu": c2_datu,
            "helpful_vs_harmful_removal": c2_controlled,
            "baseline_pairwise_accuracy": baseline,
            "correlation_role": "secondary descriptive metric only; n=6 intervention groups",
        },
        "C3_helpful_vs_harmful": {
            "target": "IQL success gap >=10pp, >=4/5 positive seeds, and positive critical visitation gap",
            "satisfied": bool(c3_signal) if c3_complete else None,
            "success": c3_iql,
            "critical_visitation": c3_critical,
        },
        "C4_cross_algorithm": {
            "target": "strong IQL C1 effect plus same-direction ReBRAC effect",
            "satisfied": bool(c4_signal) if c4_complete else None,
            "iql": c1_iql,
            "rebrac": c1_rebrac,
            "bc_and_cql_role": "mechanism diagnostics; no same-magnitude requirement",
        },
        "DATU_component_ablation": {
            "target": "full DATU pairwise accuracy exceeds at least two of three removals",
            **ablation,
        },
        "note": (
            "null means required preregistered outputs are incomplete. Three-seed results show "
            "individual points and direction; they are not overclaimed as precise significance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate LAB-REVIEW CORE v1 with paired-seed claim checks"
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/aggregated")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.results_root)
    records = [
        flatten_manifest(json.loads(path.read_text(encoding="utf-8")), path)
        for path in root.rglob("run_manifest.json")
    ]
    if not records:
        raise FileNotFoundError(f"No run_manifest.json files found under {root}")
    runs = pd.DataFrame(records)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "runs.csv", index=False)
    summary = summarize(runs, args.resamples, args.seed)
    summary.to_csv(output / "summary.csv", index=False)
    marginal = paired_marginal_utility_rows(runs)
    marginal.to_csv(output / "paired_marginal_utility.csv", index=False)
    marginal_records = marginal.to_dict("records")
    atomic_json_dump(
        summarize_empirical_marginal_utility(
            marginal_records, resamples=args.resamples, seed=args.seed
        ),
        output / "paired_marginal_utility_summary.json",
    )
    criteria = success_criteria(root, runs, marginal, args.resamples, args.seed)
    atomic_json_dump(criteria, output / "success_criteria.json")
    print(
        json.dumps(
            {
                "runs": len(runs),
                "summary_rows": len(summary),
                "paired_marginal_rows": len(marginal),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
