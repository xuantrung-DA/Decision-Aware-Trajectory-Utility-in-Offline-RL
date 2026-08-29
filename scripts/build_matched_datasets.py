from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from datu.datasets.matched_builder import CriticalRegion, MatchedDatasetBuilder
from datu.datasets.minari_loader import load_processed_dataset
from datu.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit controlled matched dataset conditions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    dataset = load_processed_dataset(config["input"])
    region_config = config["critical_region"]
    if not bool(region_config.get("frozen", False)):
        raise RuntimeError(
            "critical_region is still a placeholder. Inspect the PointMaze dataset, "
            "freeze the bounds in configs/matched/pointmaze.yaml, and set frozen=true."
        )
    region = CriticalRegion(
        tuple(region_config["observation_dims"]),
        tuple(region_config["lower"]),
        tuple(region_config["upper"]),
    )
    builder = MatchedDatasetBuilder(region, int(config.get("seed", 0)))
    matching = config["matching"]
    conditions, manifest = builder.build(
        dataset,
        int(matching["base_trajectories"]),
        int(matching["added_trajectories"]),
        matching,
        float(matching.get("support_level", 1.0)),
    )
    manifest_path = Path(config["manifest_dir"]) / "matched_pointmaze.json"
    builder.save(conditions, manifest, config["output_dir"], manifest_path)
    print(json.dumps(manifest["comparisons"], indent=2))
    print(f"manifest: {manifest_path}")
    if not manifest["all_comparisons_matched"]:
        message = (
            "WARNING: requested matching tolerances were not all satisfied; inspect the manifest."
        )
        if args.require_match:
            raise RuntimeError(message)
        print(message)


if __name__ == "__main__":
    main()
