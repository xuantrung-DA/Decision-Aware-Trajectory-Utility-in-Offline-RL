"""Run only the six approved supplemental controls, with a wall-clock guard."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
import yaml

from datu.algorithms.factory import build_agent
from datu.datasets.minari_loader import load_processed_dataset
from datu.evaluation.policy_eval import evaluate_policy
from datu.experiments.ijmlc_supplement import (
    EVAL_EPISODES,
    EVAL_SEED,
    STUDY,
    fresh_fingerprint,
    load_protocol,
    make_training_config,
)
from datu.training import load_dataset_from_config, make_environment, run_training
from datu.utils.io import atomic_json_dump, sha256_file, system_info, yaml_dump
from datu.utils.normalization import NormalizedAgent, ObservationNormalizer
from datu.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "supplement_assets"
OUTPUT = ROOT / "results" / STUDY


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def context():
    protocol = load_protocol(ASSETS / "protocol.json")
    index = read(ROOT / "supplement_input_index.json")
    return protocol, index


def verify_old_files(index):
    for item in index["runs"]:
        folder = Path(item["path"])
        for name, expected in item["sha256"].items():
            path = folder / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Original run file changed: {path}")


def preflight():
    protocol, index = context()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: enable a Kaggle GPU before training")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    verify_old_files(index)
    dataset = load_processed_dataset(
        ROOT / "data/processed/matched/pointmaze_medium/reference_critical_high"
    )
    if fresh_fingerprint(dataset) != protocol["reference_hash"]:
        raise ValueError("Frozen reference data hash mismatch")
    for name, group in protocol["groups"].items():
        remaining = dataset.remove_trajectories(group["trajectory_indices"])
        if fresh_fingerprint(remaining) != group["remaining_dataset_hash"]:
            raise ValueError(f"Removal hash mismatch: {name}")
        print(
            f"[matched-data] {name}: retained={len(remaining):,}, trajectories=140, "
            f"removed={group['removed_transitions']:,}, target=7,242, "
            f"overlap_top_DATU={group['target_overlap_trajectories']}/40",
            flush=True,
        )
    # Cheap reproducibility diagnostics, not hyperparameter tuning or pilot training.
    for algorithm in ("iql", "rebrac"):
        item = next(
            x
            for x in index["runs"]
            if x["algorithm"] == algorithm and x["block"] == "A_CORE" and x["seed"] == 0
        )
        agent, config, env = evaluator(Path(item["path"]))
        try:
            actual = evaluate_policy(
                agent,
                env,
                episodes=20,
                seed=10000,
                observation_keys=config["dataset"]["observation_keys"],
                max_episode_steps=250,
                critical_region=config["dataset"]["critical_region"],
            )
        finally:
            env.close()
        expected = read(Path(item["path"]) / "evaluation_metrics.json")
        if not np.allclose(
            actual["episodic_returns"], expected["episodic_returns"], rtol=0, atol=1e-5
        ):
            raise RuntimeError(
                f"{algorithm}: old 20-episode evaluation does not reproduce. "
                "Stop and inspect environment/runtime before spending training quota."
            )
        if actual["success_rate"] != expected["success_rate"]:
            raise RuntimeError(f"{algorithm}: sentinel success mismatch")
        print(f"[reproducibility] {algorithm}: original 20-episode checkpoint PASS", flush=True)
        del agent
    atomic_json_dump(
        {
            "protocol_signature": protocol["signature"],
            "system": system_info(),
            "old_checkpoints_verified": len(index["runs"]),
        },
        OUTPUT / "preflight.json",
    )
    print(
        f"PREFLIGHT PASS: {len(index['runs'])} original runs; 6 new ReBRAC jobs; no original retraining",
        flush=True,
    )


def evaluator(folder):
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    # Use STORED normalization, never refit on the full/reference dataset.
    normal = read(folder / "normalization.json")
    adapter = None if normal.get("disabled") else ObservationNormalizer.from_state_dict(normal)
    agent = build_agent(config, 6, 2, "cuda")
    agent.load(folder / "checkpoint_final.pt")
    env = make_environment(config["dataset"])
    return NormalizedAgent(agent, adapter), config, env


def train_one(job_id):
    protocol, _ = context()
    job = next(j for j in protocol["jobs"] if j["id"] == job_id)
    base = yaml.safe_load((ASSETS / "rebrac_base.yaml").read_text(encoding="utf-8"))
    config = make_training_config(base, protocol, job, ROOT, OUTPUT)
    dataset, _ = load_dataset_from_config(config)
    if fresh_fingerprint(dataset) != protocol["groups"][job["group"]]["remaining_dataset_hash"]:
        raise ValueError("Supplemental dataset hash mismatch before training")
    folder = OUTPUT / "runs" / job_id
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Refusing to overwrite a previous/partial run: {folder}")
    yaml_dump(config, OUTPUT / "configs" / f"{job_id}.yaml")
    run_training(config, folder)


def complete_job(job, protocol):
    folder = OUTPUT / "runs" / job["id"]
    manifest = folder / "run_manifest.json"
    if not manifest.exists():
        return False
    record = read(manifest)
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    if (
        record["steps"] != 200000
        or record["algorithm"] != "rebrac"
        or record["seed"] != job["seed"]
        or record["dataset_hash"] != protocol["groups"][job["group"]]["remaining_dataset_hash"]
        or config.get("supplement_protocol_signature") != protocol["signature"]
        or not (folder / "checkpoint_final.pt").is_file()
        or not np.isfinite(record["evaluation"]["success_rate"])
    ):
        raise ValueError(f"Invalid completion artifacts: {folder}")
    return True


def save_status(protocol, interrupted=None):
    jobs = []
    for job in protocol["jobs"]:
        folder = OUTPUT / "runs" / job["id"]
        status = (
            "complete"
            if complete_job(job, protocol)
            else ("partial" if folder.exists() and any(folder.iterdir()) else "planned")
        )
        jobs.append({**job, "status": status})
    result = {
        "study": STUDY,
        "protocol_signature": protocol["signature"],
        "jobs": jobs,
        "complete": sum(j["status"] == "complete" for j in jobs),
        "interrupted": interrupted,
        "original_study_unchanged": True,
    }
    atomic_json_dump(result, OUTPUT / "jobs.json")
    return result


def run_child(job, deadline):
    """Bound one child process; retain its interruption checkpoint on time expiry."""
    log = OUTPUT / "logs" / f"{job['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "train-one",
        "--job-id",
        job["id"],
    ]
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    timed_out = False
    with log.open("a", encoding="utf-8") as handle:

        def forward():
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()

        thread = threading.Thread(target=forward, daemon=True)
        thread.start()
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    print(
                        "[deadline] interrupting current job; partial checkpoint is NOT a completed result",
                        flush=True,
                    )
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=10)
                    break
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            thread.join(timeout=10)
    return process.returncode, timed_out


def evaluation_order(protocol, index):
    # Paired primary contrast first, followed by the entire old Block B, without
    # ordering on observed success. Each seed's full/top/randoms are consecutive.
    old = index["runs"]
    ordered = []
    for seed in (0, 1, 2):
        for variant in ("reference_critical_high", "remove_top_datu"):
            ordered.append(
                next(
                    x
                    for x in old
                    if x["algorithm"] == "rebrac" and x["seed"] == seed and x["variant"] == variant
                )
            )
        for job in protocol["jobs"]:
            if job["seed"] == seed and complete_job(job, protocol):
                ordered.append(
                    {
                        "id": job["id"],
                        "path": str(OUTPUT / "runs" / job["id"]),
                        "algorithm": "rebrac",
                        "seed": seed,
                        "variant": job["group"],
                    }
                )
    used = {x["id"] for x in ordered}
    ordered.extend(x for x in sorted(old, key=lambda x: x["id"]) if x["id"] not in used)
    return ordered


def reevaluate(item, protocol, deadline):
    destination = OUTPUT / "evaluation_100" / f"{item['id']}.json"
    folder = Path(item["path"])
    checkpoint_hash = sha256_file(folder / "checkpoint_final.pt")
    state = (
        read(destination)
        if destination.exists()
        else {
            "run_id": item["id"],
            "algorithm": item["algorithm"],
            "seed": item["seed"],
            "variant": item["variant"],
            "evaluation_seed": EVAL_SEED,
            "episodes": 0,
            "protocol_signature": protocol["signature"],
            "checkpoint_sha256": checkpoint_hash,
            "chunks": [],
            "status": "partial",
        }
    )
    if (
        state["checkpoint_sha256"] != checkpoint_hash
        or state["evaluation_seed"] != EVAL_SEED
        or state["protocol_signature"] != protocol["signature"]
    ):
        raise ValueError(f"Evaluation resume mismatch: {destination}")
    if state["episodes"] == EVAL_EPISODES:
        return True
    agent, config, env = evaluator(folder)
    try:
        while state["episodes"] < EVAL_EPISODES:
            if time.monotonic() + 15 >= deadline:
                return False
            metrics = evaluate_policy(
                agent,
                env,
                episodes=10,
                seed=EVAL_SEED + state["episodes"],
                observation_keys=config["dataset"]["observation_keys"],
                max_episode_steps=250,
                critical_region=config["dataset"]["critical_region"],
            )
            state["chunks"].append(metrics)
            state["episodes"] += 10
            state["success_rate"] = float(np.mean([m["success_rate"] for m in state["chunks"]]))
            state["status"] = "complete" if state["episodes"] == EVAL_EPISODES else "partial"
            atomic_json_dump(state, destination)
            print(
                f"[eval] {item['id']} episodes={state['episodes']}/100 "
                f"success={state['success_rate']:.3f}",
                flush=True,
            )
    finally:
        env.close()
        del agent
        gc.collect()
    return True


def summarize(protocol):
    records = [read(p) for p in (OUTPUT / "evaluation_100").glob("*.json")]
    ready = {
        (r["variant"], r["seed"]): r
        for r in records
        if r["status"] == "complete" and r["algorithm"] == "rebrac"
    }
    rows = []
    needed = ("reference_critical_high", "remove_top_datu", *protocol["groups"])
    for seed in (0, 1, 2):
        if all((name, seed) in ready for name in needed):
            values = {name: ready[name, seed]["success_rate"] for name in needed}
            random_mean = float(np.mean([values[g] for g in protocol["groups"]]))
            rows.append(
                {
                    "seed": seed,
                    "full": values[needed[0]],
                    "remove_top_datu": values[needed[1]],
                    "random_group_1": values[needed[2]],
                    "random_group_2": values[needed[3]],
                    "random_minus_top_datu_pp": 100 * (random_mean - values[needed[1]]),
                }
            )
    result = {
        "scope": "post-hoc ReBRAC control, two fixed random draws",
        "evaluation_episodes": 100,
        "evaluation_seed": EVAL_SEED,
        "paired_seed_rows": rows,
        "all_three_training_seeds_ready": len(rows) == 3,
        "no_confirmatory_claim_from_partial_panel": True,
        "complete_evaluations": sum(r["status"] == "complete" for r in records),
    }
    if len(rows) == 3:
        result["mean_random_minus_top_datu_pp"] = float(
            np.mean([r["random_minus_top_datu_pp"] for r in rows])
        )
    atomic_json_dump(result, OUTPUT / "summary.json")
    return result


def package_output():
    """New POSIX ZIP; keep recovery checkpoints for partial runs, not duplicates."""
    destination = ROOT.parent / "datu_ijmlc_supplement_results.zip"
    temporary = destination.with_suffix(".zip.tmp")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for directory in (OUTPUT, ASSETS):
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix == ".tmp":
                    continue
                if (
                    path.name.startswith("checkpoint_step_")
                    and (path.parent / "run_manifest.json").exists()
                ):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
        for path in (ROOT / "supplement_input_index.json", ROOT / "supplement_bundle.json"):
            if path.exists():
                archive.write(path, path.name)
    temporary.replace(destination)
    digest = sha256_file(destination)
    destination.with_suffix(".zip.sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    print(
        f"[backup] {destination} size={destination.stat().st_size / 1024**2:.1f} MiB sha256={digest}",
        flush=True,
    )


def run_session(hours, buffer_minutes):
    protocol, index = context()
    checked = read(OUTPUT / "preflight.json")
    if checked["protocol_signature"] != protocol["signature"]:
        raise ValueError("Run preflight for the current protocol first")
    started = time.monotonic()
    deadline = started + hours * 3600 - buffer_minutes * 60
    if deadline <= started:
        raise ValueError("Insufficient session time after backup buffer")
    durations = []
    try:
        for job in protocol["jobs"]:
            if complete_job(job, protocol):
                print(f"[skip-complete] {job['id']}", flush=True)
                continue
            estimate = max(38 * 60, 1.15 * max(durations, default=0))
            left = deadline - time.monotonic()
            if left < estimate:
                print(
                    f"[budget-stop] {left / 60:.1f}m left; next full job needs ~{estimate / 60:.1f}m",
                    flush=True,
                )
                break
            print(f"[job-start] {job['id']} work_time_left={left / 60:.1f}m", flush=True)
            begin = time.monotonic()
            code, timed_out = run_child(job, deadline)
            durations.append(time.monotonic() - begin)
            save_status(protocol, interrupted=job["id"] if code else None)
            package_output()
            if timed_out:
                print("[budget-stop] Partial training saved; exiting normally", flush=True)
                break
            if code:
                raise RuntimeError(
                    f"Training stopped with code={code}; artifacts saved, no automatic rerun"
                )
        status = save_status(protocol)
        print(f"[training-phase] complete={status['complete']}/6", flush=True)
        for item in evaluation_order(protocol, index):
            if time.monotonic() + 90 >= deadline:
                break
            if not reevaluate(item, protocol, deadline):
                break
        result = summarize(protocol)
        print("[supplement-summary]", json.dumps(result), flush=True)
    finally:
        save_status(protocol)
        package_output()
    print(
        "SESSION SAVED. Incomplete evaluations/jobs remain explicitly pending; no original run overwritten.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "train-one"))
    parser.add_argument("--job-id")
    parser.add_argument("--walltime-hours", type=float, default=4.25)
    parser.add_argument("--buffer-minutes", type=float, default=25)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    if args.mode == "preflight":
        preflight()
    elif args.mode == "train-one":
        train_one(args.job_id)
    else:
        run_session(args.walltime_hours, args.buffer_minutes)


if __name__ == "__main__":
    main()
