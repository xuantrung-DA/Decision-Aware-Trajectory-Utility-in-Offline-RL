from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from datu.datasets.minari_loader import load_processed_dataset
from datu.experiments.preregistration import freeze_core_preregistration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze LAB-REVIEW CORE v1 DATU protocol and six intervention groups"
    )
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--matched-manifest", required=True)
    parser.add_argument("--datu-components", required=True)
    parser.add_argument("--removal-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--critic-steps", type=int, default=100_000)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--aggregation", default="topk_mean")
    parser.add_argument("--topk-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset = load_processed_dataset(args.reference_dataset)
    data = freeze_core_preregistration(
        dataset,
        args.matched_manifest,
        args.datu_components,
        args.removal_dir,
        args.output,
        group_size=args.group_size,
        ensemble_size=args.ensemble_size,
        critic_steps_per_member=args.critic_steps,
        knn_k=args.knn_k,
        aggregation=args.aggregation,
        topk_fraction=args.topk_fraction,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "preregistration_hash": data["preregistration_hash"],
                "groups": list(data["intervention_groups"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
