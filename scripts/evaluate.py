from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from datu.evaluation.policy_eval import evaluate_policy
from datu.training import load_agent_for_evaluation, load_dataset_from_config, make_environment
from datu.utils.config import load_config
from datu.utils.io import atomic_json_dump
from datu.utils.normalization import NormalizedAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a predefined/final Offline RL checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    agent, normalizer = load_agent_for_evaluation(config, args.checkpoint)
    _, minari_handle = load_dataset_from_config(config)
    env = make_environment(config["dataset"], minari_handle)
    try:
        evaluation_config = config.get("evaluation", {})
        metrics = evaluate_policy(
            NormalizedAgent(agent, normalizer),
            env,
            episodes=args.episodes or int(evaluation_config.get("episodes", 20)),
            seed=int(evaluation_config.get("seed", 10_000)),
            observation_keys=config["dataset"].get("observation_keys"),
            normalized_score=(
                tuple(evaluation_config["normalized_score"])
                if evaluation_config.get("normalized_score") is not None
                else None
            ),
            max_episode_steps=(
                int(evaluation_config["max_episode_steps"])
                if evaluation_config.get("max_episode_steps") is not None
                else None
            ),
            critical_region=config["dataset"].get("critical_region"),
        )
    finally:
        env.close()
    atomic_json_dump(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
