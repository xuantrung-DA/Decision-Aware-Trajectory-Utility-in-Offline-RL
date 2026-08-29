from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from datu.training import run_training
from datu.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Offline RL agent from a fixed dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument(
        "overrides", nargs="*", help="YAML overrides such as seed=1 training.steps=500000"
    )
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    destination, result = run_training(config, args.run_dir, evaluate=not args.no_evaluate)
    print(json.dumps({"run_dir": str(destination), "result": result}, indent=2, default=str))


if __name__ == "__main__":
    main()
