from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Claim-critical evidence first; mechanism diagnostics last.
DEFAULT_STAGES = (
    "A_CORE:iql",
    "D_HEADLINE_EXTRA_SEEDS:iql",
    "A_CORE:rebrac",
    "A_CORE:bc",
    "B_MARGINAL:iql",
    "B_MARGINAL:rebrac",
    "C_CQL_DIAGNOSTIC:cql",
)


def _stage_status(plan_path: Path, block: str, algorithm: str) -> tuple[int, int]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = [
        job
        for job in plan["jobs"]
        if job["block"] == block and job["algorithm"] == algorithm
    ]
    return sum(job["status"] == "complete" for job in jobs), len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one globally timed, claim-prioritized Kaggle GPU session"
    )
    parser.add_argument("--config", default="configs/experiments/lab_review_core_v1.yaml")
    parser.add_argument("--plan-dir", default="results/plans/LAB_REVIEW_CORE_V1_KAGGLE")
    parser.add_argument("--walltime-hours", type=float, default=10.5)
    parser.add_argument("--shutdown-buffer-minutes", type=float, default=30.0)
    parser.add_argument("--stages", nargs="*", default=list(DEFAULT_STAGES))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed stable job IDs after the underlying issue has been fixed",
    )
    args = parser.parse_args()
    if args.walltime_hours <= 0:
        raise ValueError("--walltime-hours must be positive")
    parsed_stages = []
    for stage in args.stages:
        if stage.count(":") != 1:
            raise ValueError(f"Stage must have BLOCK:ALGORITHM form: {stage}")
        parsed_stages.append(tuple(stage.split(":", maxsplit=1)))
    print(f"[session] priority={parsed_stages}", flush=True)
    if not args.execute:
        print("[session] dry run only; add --execute to train", flush=True)
        return

    started = time.monotonic()
    deadline = started + args.walltime_hours * 3600.0
    plan_dir = Path(args.plan_dir)
    if not plan_dir.is_absolute():
        plan_dir = ROOT / plan_dir
    plan_path = plan_dir / "jobs.json"
    for block, algorithm in parsed_stages:
        remaining_hours = (deadline - time.monotonic()) / 3600.0
        if remaining_hours <= args.shutdown_buffer_minutes / 60.0:
            print(f"[session] safe stop before {block}:{algorithm}; buffer reached", flush=True)
            break
        command = [
            sys.executable,
            str(ROOT / "scripts/run_experiment.py"),
            "--config",
            args.config,
            "--plan-dir",
            str(plan_dir),
            "--execute",
            "--resume",
            "--blocks",
            block,
            "--algorithms",
            algorithm,
            "--walltime-hours",
            str(remaining_hours),
            "--shutdown-buffer-minutes",
            str(args.shutdown_buffer_minutes),
        ]
        if args.retry_failed:
            command.append("--retry-failed")
        print(
            f"[session-stage] start={block}:{algorithm} remaining={remaining_hours:.2f}h",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)
        complete, total = _stage_status(plan_path, block, algorithm)
        print(f"[session-stage] status={complete}/{total} complete", flush=True)
        if complete < total:
            print("[session] stage remains pending; preserve priority and stop", flush=True)
            break
    elapsed = (time.monotonic() - started) / 3600.0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    print(
        f"[session] elapsed={elapsed:.2f}h matrix_status={plan['status_counts']} "
        f"manifest={plan_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
