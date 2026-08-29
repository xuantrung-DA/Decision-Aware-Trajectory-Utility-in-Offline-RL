from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from datu.datasets.matched_builder import CriticalRegion
from datu.evaluation.dataset_metrics import compute_dataset_metrics
from datu.training import load_dataset_from_config
from datu.utils.config import load_config
from datu.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute auditable fixed-dataset metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    dataset, _ = load_dataset_from_config(config)
    region_config = config["dataset"].get("critical_region")
    region = None
    if region_config:
        region = CriticalRegion(
            tuple(region_config["observation_dims"]),
            tuple(region_config["lower"]),
            tuple(region_config["upper"]),
        )
    metrics = compute_dataset_metrics(dataset, region, seed=int(config.get("seed", 0)))
    atomic_json_dump(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
