from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from datu.evaluation.dataset_metrics import compute_dataset_metrics
from datu.training import load_dataset_from_config
from datu.utils.config import load_config
from datu.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a locally downloaded Minari dataset to mmap-ready arrays"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--npz",
        action="store_true",
        help="Write one compressed NPZ instead of mmap-ready .npy files",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    dataset, _ = load_dataset_from_config(config)
    destination = Path(args.output)
    if args.npz:
        dataset.save_npz(destination)
        manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    else:
        dataset.save_directory(destination)
        manifest_path = destination / "preparation_manifest.json"
    manifest = {
        "source_dataset_id": config["dataset"].get("id"),
        "output": str(destination),
        "dataset_hash": dataset.fingerprint(),
        "metrics": compute_dataset_metrics(dataset, seed=int(config.get("seed", 0))),
    }
    atomic_json_dump(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
