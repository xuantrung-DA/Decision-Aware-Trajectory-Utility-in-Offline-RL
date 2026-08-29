from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from datu.datasets.minari_loader import load_processed_dataset
from datu.evaluation.utility_validation import group_score_means, pairwise_ranking_accuracy
from datu.experiments.preregistration import load_preregistration
from datu.utility.global_coverage import trajectory_coverage_contribution
from datu.utility.rarity import trajectory_rarity
from datu.utility.return_score import reward_density


def _component_scores(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        return np.asarray(artifact["trajectory_scores"], dtype=np.float64)


def _targets(frame: pd.DataFrame, algorithm: str) -> dict[str, float]:
    selected = frame[frame["algorithm"] == algorithm]
    values = selected.groupby("group")["marginal_utility"].mean().to_dict()
    return {str(name): float(value) for name, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen utility scores against six paired removal interventions"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--marginal-utility", required=True)
    parser.add_argument("--datu-components", required=True)
    parser.add_argument("--component-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rarity-k", type=int, default=20)
    args = parser.parse_args()

    dataset = load_processed_dataset(args.dataset_path)
    preregistration = load_preregistration(args.preregistration)
    if dataset.fingerprint() != preregistration["reference_dataset_hash"]:
        raise ValueError("Active reference dataset differs from preregistration")
    marginal = pd.read_csv(args.marginal_utility)
    required = {"algorithm", "group", "marginal_utility"}
    if not required.issubset(marginal):
        raise ValueError(f"Marginal-utility table is missing {sorted(required.difference(marginal))}")

    method_scores = {
        "datu": _component_scores(Path(args.datu_components)),
        "reward_density": reward_density(dataset),
        "rarity": trajectory_rarity(dataset, k=args.rarity_k),
        "global_coverage": trajectory_coverage_contribution(dataset),
    }
    rows = []
    for algorithm in sorted(marginal["algorithm"].dropna().unique()):
        target = _targets(marginal, str(algorithm))
        for method, scores in method_scores.items():
            predicted = group_score_means(scores, preregistration)
            rows.append(
                {
                    "algorithm": str(algorithm),
                    "method": method,
                    **pairwise_ranking_accuracy(predicted, target),
                }
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "utility_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ablation_rows = []
    if args.component_root:
        component_root = Path(args.component_root)
        variants = (
            "datu_full",
            "datu_no_uncertainty",
            "datu_no_sensitivity",
            "datu_no_local_support",
        )
        for variant in variants:
            scores = _component_scores(
                component_root / f"component_{variant}" / "datu_components.npz"
            )
            predicted = group_score_means(scores, preregistration)
            for algorithm in sorted(marginal["algorithm"].dropna().unique()):
                ablation_rows.append(
                    {
                        "algorithm": str(algorithm),
                        "method": variant,
                        **pairwise_ranking_accuracy(
                            predicted, _targets(marginal, str(algorithm))
                        ),
                    }
                )
        with (output / "component_ablation.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
            writer.writeheader()
            writer.writerows(ablation_rows)
    print(
        json.dumps(
            {
                "baseline_rows": len(rows),
                "component_ablation_rows": len(ablation_rows),
                "output_dir": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
