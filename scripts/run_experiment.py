from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from datu.datasets.trajectory import TransitionDataset
from datu.experiments.preregistration import verify_preregistered_dataset
from datu.utils.config import deep_merge, load_config
from datu.utils.io import atomic_json_dump, canonical_json_hash, yaml_dump

ROOT = Path(__file__).resolve().parents[1]


def _resolve_config(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return load_config(candidate)


def _standard_training_config() -> dict[str, Any]:
    return {
        "seed": 0,
        "device": "auto",
        "training": {
            "steps": 200_000,
            "batch_size": 256,
            "log_interval": 1000,
            "progress_interval": 1000,
            "checkpoint_interval": 50_000,
            "checkpoint_selection": "final",
            "metric_abs_warning": 1_000_000.0,
            "throughput_warning_ratio": 0.5,
        },
        "evaluation": {"episodes": 20, "seed": 10_000},
        "logging": {
            "root": "results/runs",
            "tensorboard": True,
            "dataset_metrics_cache": "results/cache/dataset_metrics",
        },
    }


def _blocks(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    configured = experiment.get("blocks")
    if configured:
        return list(configured)
    # Deferred legacy configs remain dry-runnable. Core v1 uses explicit
    # blocks so algorithms/seeds/steps cannot form an accidental Cartesian product.
    return [{**experiment, "id": experiment.get("block", "LEGACY")}]


def _variants(block: dict[str, Any]) -> list[tuple[str, str]]:
    if block.get("conditions"):
        return [(str(value), "condition") for value in block["conditions"]]
    if block.get("removals"):
        return [(str(value), "removal") for value in block["removals"]]
    variants = block.get("variants", ["full"])
    group_count = int(block.get("groups", 0))
    expanded: list[tuple[str, str]] = []
    for variant in variants:
        if variant == "leave_group_out":
            expanded.extend(
                (f"leave_group_{group:02d}", "legacy_removal")
                for group in range(group_count)
            )
        else:
            expanded.append((str(variant), "full"))
    return expanded


def expand_jobs(experiment_config: dict[str, Any], plan_dir: Path) -> list[dict[str, Any]]:
    experiment = experiment_config["experiment"]
    base = _standard_training_config()
    if experiment.get("base_training_config"):
        base = deep_merge(base, _resolve_config(experiment["base_training_config"]))
    config_dir = plan_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    expected_minutes = experiment.get("expected_minutes", {})
    jobs: list[dict[str, Any]] = []

    for block in _blocks(experiment):
        block_id = str(block["id"])
        datasets = block.get("datasets", experiment.get("datasets", []))
        algorithms = block.get("algorithms", experiment.get("algorithms", []))
        seeds = block.get("seeds", experiment.get("seeds", [0]))
        steps = int(block.get("training_steps", experiment.get("training_steps", 200_000)))
        for dataset_path in datasets:
            dataset_overlay = _resolve_config(dataset_path)
            dataset_name = Path(dataset_path).stem
            for algorithm_path in algorithms:
                algorithm_overlay = _resolve_config(algorithm_path)
                algorithm_name = str(algorithm_overlay["algorithm"]["name"])
                for variant, kind in _variants(block):
                    for seed in seeds:
                        config = deep_merge(base, dataset_overlay)
                        config = deep_merge(config, algorithm_overlay)
                        config["seed"] = int(seed)
                        config["training"]["steps"] = steps
                        condition = variant if kind == "condition" else None
                        removed_group = variant if kind in {"removal", "legacy_removal"} else None
                        config["experiment"] = {
                            "id": experiment["id"],
                            "block": block_id,
                            "variant": variant,
                            "condition": condition,
                            "removed_group": removed_group,
                            "selector": None,
                            "retention_budget": None,
                        }
                        if kind == "condition":
                            template = block.get(
                                "condition_root",
                                experiment.get(
                                    "condition_root", "data/processed/matched/{dataset}"
                                ),
                            )
                            root = Path(str(template).format(dataset=dataset_name))
                            config["dataset"]["path"] = str(root / variant)
                            preregistration_path = block.get(
                                "preregistration", experiment.get("preregistration")
                            )
                            if preregistration_path:
                                config["preregistration"] = {
                                    "path": str(preregistration_path),
                                    "condition": variant,
                                    "required": True,
                                }
                        elif kind in {"removal", "legacy_removal"}:
                            template = block.get(
                                "removal_root",
                                experiment.get(
                                    "removal_root", "data/processed/removals/{dataset}"
                                ),
                            )
                            root = Path(str(template).format(dataset=dataset_name))
                            config["dataset"]["path"] = str(root / variant)
                            preregistration_path = block.get(
                                "preregistration", experiment.get("preregistration")
                            )
                            if preregistration_path:
                                config["preregistration"] = {
                                    "path": str(preregistration_path),
                                    "group": variant,
                                    "required": bool(block.get("requires_preregistration", True)),
                                }

                        job_id = (
                            f"{experiment['id']}_{block_id}_{dataset_name}_"
                            f"{algorithm_name}_{variant}_seed{int(seed)}"
                        )
                        run_dir = ROOT / "results" / "runs" / str(experiment["id"]) / job_id
                        config_path = config_dir / f"{job_id}.yaml"
                        yaml_dump(config, config_path)
                        jobs.append(
                            {
                                "job_id": job_id,
                                "block": block_id,
                                "dataset": dataset_name,
                                "algorithm": algorithm_name,
                                "variant": variant,
                                "seed": int(seed),
                                "steps": steps,
                                "expected_minutes": int(
                                    block.get(
                                        "expected_minutes",
                                        expected_minutes.get(algorithm_name, 120),
                                    )
                                ),
                                "config": str(config_path),
                                "config_hash": canonical_json_hash(config),
                                "run_dir": str(run_dir),
                                "command": [
                                    sys.executable,
                                    str(ROOT / "scripts/train.py"),
                                    "--config",
                                    str(config_path),
                                    "--run-dir",
                                    str(run_dir),
                                ],
                                "status": "planned",
                            }
                        )
    identifiers = [job["job_id"] for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Experiment expansion produced duplicate stable job IDs")
    return jobs


def _matrix_signature(experiment: dict[str, Any], jobs: list[dict[str, Any]]) -> str:
    return canonical_json_hash(
        {
            "experiment": experiment,
            "jobs": [
                {"job_id": job["job_id"], "config_hash": job["config_hash"]} for job in jobs
            ],
        }
    )


def _manifest_payload(
    experiment: dict[str, Any], jobs: list[dict[str, Any]], signature: str
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "matrix_signature": signature,
        "job_count": len(jobs),
        "total_training_updates": int(sum(job["steps"] for job in jobs)),
        "status_counts": {
            status: sum(job["status"] == status for job in jobs)
            for status in ("planned", "running", "complete", "failed")
        },
        "jobs": jobs,
    }


def _restore_statuses(
    jobs: list[dict[str, Any]], previous: dict[str, Any], retry_failed: bool
) -> None:
    old_jobs = {job["job_id"]: job for job in previous.get("jobs", [])}
    for job in jobs:
        old = old_jobs.get(job["job_id"], {})
        run_manifest = Path(job["run_dir"]) / "run_manifest.json"
        if run_manifest.exists():
            job["status"] = "complete"
        elif old.get("status") == "failed" and not retry_failed:
            job["status"] = "failed"
        else:
            # A killed Kaggle session can leave `running`; rerun that stable
            # job unless a complete run manifest exists.
            job["status"] = "planned"


def _selected(job: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        (not args.blocks or job["block"] in args.blocks)
        and (not args.algorithms or job["algorithm"] in args.algorithms)
        and (not args.seeds or job["seed"] in args.seeds)
    )


def _preflight_datasets(jobs: list[dict[str, Any]]) -> None:
    """Load every unique dataset and verify every job against the frozen protocol."""
    cache: dict[Path, TransitionDataset] = {}
    for job in jobs:
        config = load_config(job["config"])
        dataset_path = Path(config["dataset"]["path"])
        if not dataset_path.is_absolute():
            dataset_path = ROOT / dataset_path
        dataset_path = dataset_path.resolve()
        if dataset_path not in cache:
            if not (dataset_path / "dataset.json").exists():
                raise FileNotFoundError(
                    f"Preflight dataset is absent or incomplete: {dataset_path}"
                )
            dataset = TransitionDataset.load_directory(dataset_path)
            cache[dataset_path] = dataset
            print(
                f"[preflight-data] dataset={dataset_path} "
                f"transitions={len(dataset):,} trajectories={dataset.num_trajectories:,} "
                f"hash={dataset.fingerprint()}",
                flush=True,
            )
        preregistration = config.get("preregistration")
        if preregistration and preregistration.get("path"):
            preregistration_path = Path(preregistration["path"])
            if not preregistration_path.is_absolute():
                preregistration_path = ROOT / preregistration_path
            preregistration["path"] = str(preregistration_path.resolve())
        verified = verify_preregistered_dataset(config, cache[dataset_path])
        if config.get("preregistration", {}).get("required") and verified is None:
            raise RuntimeError(f"Required preregistration was not verified: {job['job_id']}")
    print(
        f"[preflight-data] PASS jobs={len(jobs)} unique_datasets={len(cache)} "
        "all required hashes verified",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand and safely execute a stable, resumable experiment matrix"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-dir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--preflight-data",
        action="store_true",
        help="Load every unique dataset and verify all frozen hashes before queue execution",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--blocks", nargs="*")
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--walltime-hours", type=float)
    parser.add_argument("--shutdown-buffer-minutes", type=float, default=30.0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    experiment = config["experiment"]
    plan_dir = (
        Path(args.plan_dir)
        if args.plan_dir
        else ROOT / "results" / "plans" / str(experiment["id"])
    )
    plan_dir.mkdir(parents=True, exist_ok=True)
    jobs = expand_jobs(config, plan_dir)
    signature = _matrix_signature(experiment, jobs)
    manifest_path = plan_dir / "jobs.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("matrix_signature") != signature:
            raise ValueError(
                f"Existing plan signature differs from active matrix: {manifest_path}. "
                "Use a new --plan-dir after an intentional protocol change."
            )
        if args.resume:
            _restore_statuses(jobs, previous, args.retry_failed)
        elif args.execute and any(job.get("status") != "planned" for job in previous["jobs"]):
            raise RuntimeError("Existing execution state requires --resume")

    expected_count = experiment.get("expected_job_count")
    expected_updates = experiment.get("expected_total_updates")
    total_updates = sum(job["steps"] for job in jobs)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"Expected {expected_count} jobs, expanded {len(jobs)}")
    if expected_updates is not None and total_updates != int(expected_updates):
        raise ValueError(f"Expected {expected_updates} updates, expanded {total_updates}")

    # Execution always performs the full matrix guard. The explicit flag makes
    # the same check available during a no-training dry run.
    if args.preflight_data or args.execute:
        _preflight_datasets(jobs)

    atomic_json_dump(_manifest_payload(experiment, jobs, signature), manifest_path)
    selected = [job for job in jobs if _selected(job, args) and job["status"] == "planned"]
    if args.max_jobs is not None:
        selected = selected[: args.max_jobs]
    print(
        f"Planned matrix jobs={len(jobs)} updates={total_updates:,}; "
        f"selected_pending={len(selected)} manifest={manifest_path}",
        flush=True,
    )
    if not args.execute:
        print("Dry run only. Add --execute to start the selected queue.")
        return

    deadline = time.monotonic() + args.walltime_hours * 3600.0 if args.walltime_hours else None
    for index, job in enumerate(selected, start=1):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            required = 60.0 * (job["expected_minutes"] + args.shutdown_buffer_minutes)
            if remaining < required:
                print(
                    f"[queue] stopping before {job['job_id']}: remaining={remaining / 60:.1f}m "
                    f"required={required / 60:.1f}m",
                    flush=True,
                )
                break
        print(f"[{index}/{len(selected)}] {job['job_id']}", flush=True)
        job["status"] = "running"
        atomic_json_dump(_manifest_payload(experiment, jobs, signature), manifest_path)
        try:
            subprocess.run(job["command"], cwd=ROOT, check=True)
            if not (Path(job["run_dir"]) / "run_manifest.json").exists():
                raise RuntimeError(f"Job exited without a run manifest: {job['job_id']}")
            job["status"] = "complete"
        except (subprocess.CalledProcessError, RuntimeError):
            job["status"] = "failed"
            atomic_json_dump(_manifest_payload(experiment, jobs, signature), manifest_path)
            raise
        atomic_json_dump(_manifest_payload(experiment, jobs, signature), manifest_path)
    final = _manifest_payload(experiment, jobs, signature)
    atomic_json_dump(final, manifest_path)
    print(f"[queue] status={final['status_counts']} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
