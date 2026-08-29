from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from datu.experiments.preregistration import load_preregistration
from datu.utils.config import load_config
from datu.utils.io import atomic_json_dump

ROOT = Path(__file__).resolve().parents[1]


def one_factor_jobs(
    config: dict[str, Any], training_config: str, output_root: Path
) -> list[dict[str, Any]]:
    ablation = config["ablation"]
    frozen = ablation["frozen_reference"]
    jobs = []

    def add(name: str, arguments: list[str]) -> None:
        jobs.append(
            {
                "name": name,
                "command": [
                    sys.executable,
                    str(ROOT / "scripts/compute_datu.py"),
                    "--config",
                    training_config,
                    "--output-dir",
                    str(output_root / name),
                    "--ensemble-cache",
                    str(ROOT / "results/reference_critics/pointmaze_core_E5"),
                    "--critic-steps",
                    str(frozen["critic_steps_per_member"]),
                    "--ensemble-size",
                    str(frozen["ensemble_size"]),
                    "--knn-k",
                    str(frozen["knn_k"]),
                    "--aggregation",
                    str(frozen["aggregation"]),
                    *arguments,
                    f"datu.topk_fraction={frozen['topk_fraction']}",
                    f"dataset.path={ablation['reference_dataset']}",
                ],
                "status": "planned",
            }
        )

    for variant in ablation["variants"]:
        switches = {
            "datu_full": [],
            "datu_no_uncertainty": ["datu.use_uncertainty=false"],
            "datu_no_sensitivity": ["datu.use_sensitivity=false"],
            "datu_no_local_support": ["datu.use_local_support=false"],
            "uncertainty_only": ["datu.variant=uncertainty_only"],
            "sensitivity_only": ["datu.variant=sensitivity_only"],
            "coverage_only": ["datu.variant=coverage_only"],
        }
        add(f"component_{variant}", switches[variant])
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan one-factor-at-a-time DATU ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-config", default="configs/experiments/pointmaze_iql.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    root = (
        Path(args.output_root)
        if args.output_root
        else ROOT / "results/ablations" / str(config["ablation"]["id"])
    )
    jobs = one_factor_jobs(config, args.training_config, root)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    manifest = root / "jobs.json"
    atomic_json_dump(
        {"ablation": config["ablation"], "job_count": len(jobs), "jobs": jobs}, manifest
    )
    print(json.dumps({"job_count": len(jobs), "manifest": str(manifest)}, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to run executable DATU-analysis jobs sequentially.")
        return
    preregistration = load_preregistration(config["ablation"]["preregistration"])
    frozen = config["ablation"]["frozen_reference"]
    registered = preregistration["datu_definition"]
    for key in ("ensemble_size", "critic_steps_per_member", "knn_k", "aggregation", "topk_fraction"):
        if registered[key] != frozen[key]:
            raise ValueError(f"Ablation setting {key} differs from preregistration")
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job['name']}", flush=True)
        subprocess.run(job["command"], cwd=ROOT, check=True)
        job["status"] = "complete"
        atomic_json_dump(
            {"ablation": config["ablation"], "job_count": len(jobs), "jobs": jobs}, manifest
        )


if __name__ == "__main__":
    main()
