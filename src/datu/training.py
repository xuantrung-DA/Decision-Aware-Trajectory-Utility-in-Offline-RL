from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datu.algorithms import build_agent
from datu.datasets.matched_builder import CriticalRegion
from datu.datasets.minari_loader import load_minari_dataset, load_processed_dataset
from datu.datasets.trajectory import TransitionDataset
from datu.evaluation.dataset_metrics import compute_dataset_metrics
from datu.evaluation.policy_eval import evaluate_policy
from datu.experiments.preregistration import verify_preregistered_dataset
from datu.utils.config import validate_training_config
from datu.utils.io import atomic_json_dump, git_commit, system_info, unique_run_dir, yaml_dump
from datu.utils.normalization import NormalizedAgent, ObservationNormalizer, normalize_dataset
from datu.utils.seed import set_seed


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = round(seconds)
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _add_metrics(accumulator: dict[str, float], metrics: Mapping[str, float]) -> None:
    for key, value in metrics.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _mean_metrics(accumulator: Mapping[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        raise ValueError("Metric accumulator count must be positive")
    return {key: value / count for key, value in accumulator.items()}


def _gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated(index) / (1024**2),
        "gpu_reserved_mb": torch.cuda.memory_reserved(index) / (1024**2),
        "gpu_peak_mb": torch.cuda.max_memory_allocated(index) / (1024**2),
        "gpu_total_mb": torch.cuda.get_device_properties(index).total_memory / (1024**2),
    }


def _critical_region(config: Mapping[str, Any]) -> CriticalRegion | None:
    region = config.get("dataset", {}).get("critical_region")
    if not region:
        return None
    if region.get("frozen") is False:
        raise RuntimeError(
            "dataset.critical_region is an unfrozen placeholder; Phase 2 inspection must "
            "freeze it before metrics or policy training are allowed"
        )
    return CriticalRegion(
        tuple(region["observation_dims"]),
        tuple(region["lower"]),
        tuple(region["upper"]),
    )


def _dataset_metrics_with_cache(
    dataset: TransitionDataset, config: Mapping[str, Any]
) -> tuple[dict[str, Any], bool, Path | None]:
    metrics_seed = int(config.get("dataset", {}).get("metrics_seed", 0))
    dataset_hash = dataset.fingerprint()
    specification = {
        "version": 1,
        "metrics_seed": metrics_seed,
        "critical_region": config.get("dataset", {}).get("critical_region"),
    }
    signature = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_setting = config.get("logging", {}).get(
        "dataset_metrics_cache", "results/cache/dataset_metrics"
    )
    cache_path = None
    if cache_setting:
        cache_path = Path(cache_setting) / f"{dataset_hash}_{signature}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("dataset_hash") != dataset_hash:
                raise ValueError(f"Dataset metrics cache hash mismatch: {cache_path}")
            return cached, True, cache_path
    metrics = compute_dataset_metrics(
        dataset,
        critical_region=_critical_region(config),
        seed=metrics_seed,
    )
    if cache_path is not None:
        atomic_json_dump(metrics, cache_path)
    return metrics, False, cache_path


def load_dataset_from_config(config: Mapping[str, Any]) -> tuple[TransitionDataset, Any | None]:
    dataset_config = config["dataset"]
    path_value = dataset_config.get("path")
    source = Path(path_value) if path_value else None
    minari_handle = None
    if source is not None and source.exists():
        dataset = load_processed_dataset(source)
    elif dataset_config.get("id"):
        dataset, minari_handle = load_minari_dataset(
            dataset_config["id"],
            observation_keys=dataset_config.get("observation_keys"),
            max_episodes=dataset_config.get("max_episodes"),
        )
    else:
        raise FileNotFoundError("Dataset config has neither an existing path nor a Minari id")

    removal = dataset_config.get("remove_trajectory_indices")
    if removal:
        dataset = dataset.remove_trajectories(removal)
    max_transitions = dataset_config.get("max_transitions")
    if max_transitions is not None and len(dataset) > int(max_transitions):
        cumulative = np.cumsum(dataset.trajectory_lengths)
        count = int(np.searchsorted(cumulative, int(max_transitions), side="right"))
        count = max(1, count)
        dataset = dataset.select_trajectories(list(range(count)))
    action_limit = float(dataset_config.get("normalized_action_limit", 1.001))
    if np.max(np.abs(dataset.actions)) > action_limit:
        raise ValueError(
            "Dataset actions are outside [-1,1]. Supply an action-normalized processed dataset; "
            "silent clipping would change the Offline RL problem."
        )
    verify_preregistered_dataset(config, dataset)
    return dataset, minari_handle


def make_environment(dataset_config: Mapping[str, Any], minari_handle: Any | None = None) -> Any:
    if minari_handle is not None and hasattr(minari_handle, "recover_environment"):
        try:
            return minari_handle.recover_environment(eval_env=True)
        except (TypeError, ValueError):
            return minari_handle.recover_environment()
    import gymnasium as gym

    try:
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
    except ImportError:
        gymnasium_robotics = None
    env_id = dataset_config.get("env_id")
    if not env_id:
        raise ValueError(
            "dataset.env_id is required when an environment cannot be recovered from Minari"
        )
    return gym.make(env_id, **dataset_config.get("environment_kwargs", {}))


def run_training(
    config: Mapping[str, Any],
    run_dir: str | Path | None = None,
    evaluate: bool = True,
) -> tuple[Path, dict[str, Any]]:
    validate_training_config(config)
    # Dataset preparation/inspection is allowed before this freeze; policy
    # training is not, because doing so would make the oracle region post hoc.
    _critical_region(config)
    seed = int(config.get("seed", 0))
    rng = set_seed(seed)
    algorithm_name = str(config["algorithm"]["name"])
    dataset_label = config["dataset"].get("id") or config["dataset"].get("path")
    print(
        f"[setup] algorithm={algorithm_name} seed={seed} dataset={dataset_label} "
        f"requested_device={config.get('device', 'auto')}",
        flush=True,
    )
    print("[setup] loading offline trajectories...", flush=True)
    raw_dataset, minari_handle = load_dataset_from_config(config)
    print(
        f"[setup] loaded transitions={len(raw_dataset):,} "
        f"trajectories={raw_dataset.num_trajectories:,} "
        f"obs_dim={raw_dataset.observation_dim} action_dim={raw_dataset.action_dim}",
        flush=True,
    )
    preregistration = config.get("preregistration")
    if preregistration:
        print(
            f"[setup] preregistration=verified condition={preregistration.get('condition')} "
            f"group={preregistration.get('group')} "
            f"path={preregistration.get('path')}",
            flush=True,
        )
    print("[setup] resolving dataset hash and cached metrics...", flush=True)
    dataset_stats, metrics_cache_hit, metrics_cache_path = _dataset_metrics_with_cache(
        raw_dataset, config
    )
    cache_status = "hit" if metrics_cache_hit else "miss/computed"
    print(
        f"[setup] dataset_hash={dataset_stats['dataset_hash'][:16]}... "
        f"metrics_cache={cache_status} path={metrics_cache_path}",
        flush=True,
    )
    normalizer = None
    dataset = raw_dataset
    if config["dataset"].get("normalize_observations", True):
        print("[setup] fitting observation normalization...", flush=True)
        normalizer = ObservationNormalizer.fit(raw_dataset.observations)
        dataset = normalize_dataset(raw_dataset, normalizer)

    destination = (
        Path(run_dir)
        if run_dir is not None
        else unique_run_dir(
            config.get("logging", {}).get("root", "results/runs"), algorithm_name, seed
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    yaml_dump(config, destination / "config.yaml")
    atomic_json_dump(dataset_stats, destination / "dataset_statistics.json")
    atomic_json_dump(
        normalizer.state_dict() if normalizer is not None else {"disabled": True},
        destination / "normalization.json",
    )
    agent = build_agent(
        config,
        dataset.observation_dim,
        dataset.action_dim,
        device=str(config.get("device", "auto")),
    )
    if agent.device.type == "cuda":
        device_name = torch.cuda.get_device_name(agent.device)
    else:
        device_name = str(agent.device)
    print(f"[setup] active_device={device_name} run_dir={destination}", flush=True)
    writer = None
    if config.get("logging", {}).get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(destination / "tensorboard")

    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    log_interval = int(config["training"].get("log_interval", 1000))
    progress_interval = int(config["training"].get("progress_interval", log_interval))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", steps))
    if min(log_interval, progress_interval, checkpoint_interval) <= 0:
        raise ValueError("log/progress/checkpoint intervals must be positive")
    metric_abs_warning = float(config["training"].get("metric_abs_warning", 1e6))
    throughput_warning_ratio = float(config["training"].get("throughput_warning_ratio", 0.5))
    if not 0.0 < throughput_warning_ratio <= 1.0:
        raise ValueError("training.throughput_warning_ratio must lie in (0,1]")
    print(
        f"[train] steps={steps:,} batch={batch_size} log_every={log_interval:,} "
        f"progress_every={progress_interval:,} checkpoint_every={checkpoint_interval:,}",
        flush=True,
    )
    metric_path = destination / "training_metrics.csv"
    fieldnames: list[str] | None = None
    start_time = time.perf_counter()
    log_window_start = start_time
    progress_window_start = start_time
    last_metrics: dict[str, float] = {}
    last_reported_metrics: dict[str, float] = {}
    log_sums: dict[str, float] = {}
    progress_sums: dict[str, float] = {}
    log_count = 0
    progress_count = 0
    log_sample_seconds = 0.0
    log_update_seconds = 0.0
    progress_sample_seconds = 0.0
    progress_update_seconds = 0.0
    total_sample_seconds = 0.0
    total_update_seconds = 0.0
    best_progress_rate = 0.0
    last_step = 0
    try:
        with metric_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = None
            for step in range(1, steps + 1):
                last_step = step
                sample_start = time.perf_counter()
                batch = dataset.sample(batch_size, rng, agent.device)
                sample_seconds = time.perf_counter() - sample_start
                update_start = time.perf_counter()
                last_metrics = agent.train_step(batch)
                update_seconds = time.perf_counter() - update_start
                if not all(np.isfinite(value) for value in last_metrics.values()):
                    print(
                        f"[error] non-finite training metric at step={step:,}: {last_metrics}",
                        flush=True,
                    )
                    atomic_json_dump(
                        {"step": step, "metrics": last_metrics},
                        destination / "failure_metrics.json",
                    )
                    raise FloatingPointError(
                        f"Non-finite training metric at step {step}: {last_metrics}"
                    )

                _add_metrics(log_sums, last_metrics)
                _add_metrics(progress_sums, last_metrics)
                log_count += 1
                progress_count += 1
                log_sample_seconds += sample_seconds
                log_update_seconds += update_seconds
                progress_sample_seconds += sample_seconds
                progress_update_seconds += update_seconds
                total_sample_seconds += sample_seconds
                total_update_seconds += update_seconds

                should_log = step == 1 or step % log_interval == 0 or step == steps
                if should_log:
                    now = time.perf_counter()
                    log_metrics = _mean_metrics(log_sums, log_count)
                    elapsed = now - start_time
                    window_elapsed = max(now - log_window_start, 1e-12)
                    performance = {
                        "interval_updates_per_second": log_count / window_elapsed,
                        "overall_updates_per_second": step / max(elapsed, 1e-12),
                        "sample_ms_per_batch": 1000.0 * log_sample_seconds / log_count,
                        "update_ms_per_batch": 1000.0 * log_update_seconds / log_count,
                    }
                    row = {
                        "step": step,
                        "elapsed_seconds": elapsed,
                        **performance,
                        **_gpu_memory(agent.device),
                        **log_metrics,
                    }
                    if fieldnames is None:
                        fieldnames = list(row)
                        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        csv_writer.writeheader()
                    assert csv_writer is not None
                    csv_writer.writerow(row)
                    handle.flush()
                    if writer is not None:
                        for key, value in log_metrics.items():
                            writer.add_scalar(f"train/{key}", value, step)
                        for key, value in performance.items():
                            writer.add_scalar(f"performance/{key}", value, step)
                    log_sums = {}
                    log_count = 0
                    log_sample_seconds = 0.0
                    log_update_seconds = 0.0
                    log_window_start = now

                should_report = step == 1 or step % progress_interval == 0 or step == steps
                if should_report:
                    now = time.perf_counter()
                    last_reported_metrics = _mean_metrics(progress_sums, progress_count)
                    interval_elapsed = max(now - progress_window_start, 1e-12)
                    interval_rate = progress_count / interval_elapsed
                    overall_rate = step / max(now - start_time, 1e-12)
                    eta_seconds = (steps - step) / max(overall_rate, 1e-12)
                    gpu = _gpu_memory(agent.device)
                    gpu_text = ""
                    if gpu:
                        gpu_text = (
                            f" vram={gpu['gpu_allocated_mb']:.0f}/"
                            f"{gpu['gpu_reserved_mb']:.0f}/"
                            f"{gpu['gpu_total_mb']:.0f}MB(alloc/reserved/total)"
                        )
                    metric_text = " ".join(
                        f"{key}={value:.5g}" for key, value in sorted(last_reported_metrics.items())
                    )
                    print(
                        f"[progress] step={step:,}/{steps:,} ({100.0 * step / steps:6.2f}%) "
                        f"rate={interval_rate:.1f}upd/s overall={overall_rate:.1f}upd/s "
                        f"eta={_format_duration(eta_seconds)} "
                        f"sample={1000.0 * progress_sample_seconds / progress_count:.2f}ms "
                        f"update={1000.0 * progress_update_seconds / progress_count:.2f}ms"
                        f"{gpu_text} | {metric_text}",
                        flush=True,
                    )
                    if step > progress_interval:
                        if (
                            best_progress_rate
                            and interval_rate < best_progress_rate * throughput_warning_ratio
                        ):
                            print(
                                f"[warning] throughput dropped to {interval_rate:.1f} upd/s; "
                                f"best interval was {best_progress_rate:.1f} upd/s.",
                                flush=True,
                            )
                        best_progress_rate = max(best_progress_rate, interval_rate)
                    large_metrics = {
                        key: value
                        for key, value in last_reported_metrics.items()
                        if abs(value) >= metric_abs_warning
                    }
                    if large_metrics:
                        print(
                            f"[warning] metric magnitude exceeded {metric_abs_warning:g}: "
                            f"{large_metrics}",
                            flush=True,
                        )
                    if gpu and gpu["gpu_reserved_mb"] >= 0.9 * gpu["gpu_total_mb"]:
                        print(
                            "[warning] CUDA reserved memory is above 90% of total VRAM.", flush=True
                        )
                    progress_sums = {}
                    progress_count = 0
                    progress_sample_seconds = 0.0
                    progress_update_seconds = 0.0
                    progress_window_start = now

                if step % checkpoint_interval == 0 and step != steps:
                    intermediate = destination / f"checkpoint_step_{step}.pt"
                    checkpoint_start = time.perf_counter()
                    agent.save(intermediate)
                    checkpoint_seconds = time.perf_counter() - checkpoint_start
                    # Checkpoint I/O is expected overhead, not a sampling/update
                    # slowdown. Exclude it from the next interval warning while
                    # retaining it in the end-to-end overall rate and runtime.
                    log_window_start += checkpoint_seconds
                    progress_window_start += checkpoint_seconds
                    print(
                        f"[checkpoint] saved step={step:,} "
                        f"duration={checkpoint_seconds:.2f}s path={intermediate}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        interrupted = destination / f"checkpoint_interrupted_step_{last_step}.pt"
        agent.save(interrupted)
        print(
            f"[interrupt] saved recoverable checkpoint at step={last_step:,}: {interrupted}",
            flush=True,
        )
        raise
    finally:
        if writer is not None:
            writer.close()
    checkpoint = destination / "checkpoint_final.pt"
    agent.save(checkpoint)
    training_runtime_seconds = time.perf_counter() - start_time
    overall_update_rate = steps / max(training_runtime_seconds, 1e-12)
    print(
        f"[train] complete runtime={_format_duration(training_runtime_seconds)} "
        f"overall={overall_update_rate:.1f}upd/s checkpoint={checkpoint}",
        flush=True,
    )

    evaluation_metrics = None
    if evaluate:
        print("[eval] starting predefined final-checkpoint evaluation...", flush=True)
        env = make_environment(config["dataset"], minari_handle)
        try:
            evaluation_config = config.get("evaluation", {})
            normalized_spec = evaluation_config.get("normalized_score")
            normalized_range = tuple(normalized_spec) if normalized_spec is not None else None
            evaluation_metrics = evaluate_policy(
                NormalizedAgent(agent, normalizer),
                env,
                episodes=int(evaluation_config.get("episodes", 20)),
                seed=int(evaluation_config.get("seed", 10_000)),
                observation_keys=config["dataset"].get("observation_keys"),
                normalized_score=normalized_range,
                max_episode_steps=(
                    int(evaluation_config["max_episode_steps"])
                    if evaluation_config.get("max_episode_steps") is not None
                    else None
                ),
                critical_region=config["dataset"].get("critical_region"),
            )
            atomic_json_dump(evaluation_metrics, destination / "evaluation_metrics.json")
            print(
                f"[eval] return_mean={evaluation_metrics['return_mean']:.5g} "
                f"success_rate={evaluation_metrics['success_rate']:.3f} "
                f"episodes={evaluation_metrics['episodes']}",
                flush=True,
            )
        finally:
            env.close()

    result = {
        "run_dir": str(destination),
        "algorithm": algorithm_name,
        "seed": seed,
        "environment": config["dataset"].get("env_id"),
        "dataset_id": config["dataset"].get("id"),
        "dataset_hash": dataset_stats["dataset_hash"],
        "steps": steps,
        "runtime_seconds": time.perf_counter() - start_time,
        "training_runtime_seconds": training_runtime_seconds,
        "overall_updates_per_second": overall_update_rate,
        "sample_ms_per_batch": 1000.0 * total_sample_seconds / steps,
        "update_ms_per_batch": 1000.0 * total_update_seconds / steps,
        "checkpoint": str(checkpoint),
        "checkpoint_selection": config["training"].get("checkpoint_selection", "final"),
        "online_evaluation_used_for_selection": False,
        "final_training_metrics": last_metrics,
        "final_interval_metrics": last_reported_metrics,
        "evaluation": evaluation_metrics,
        "git_commit": git_commit(),
        "system": system_info(),
        "experiment_id": config.get("experiment", {}).get("id"),
        "experiment_block": config.get("experiment", {}).get("block"),
        "experiment_variant": config.get("experiment", {}).get("variant"),
        "dataset_condition": config.get("experiment", {}).get("condition"),
        "removed_group": config.get("experiment", {}).get("removed_group"),
        "selector": config.get("experiment", {}).get("selector"),
        "retention_budget": config.get("experiment", {}).get("retention_budget"),
    }
    atomic_json_dump(result, destination / "run_manifest.json")
    return destination, result


def load_agent_for_evaluation(
    config: Mapping[str, Any], checkpoint: str | Path
) -> tuple[Any, ObservationNormalizer | None]:
    _critical_region(config)
    dataset, _ = load_dataset_from_config(config)
    normalizer = (
        ObservationNormalizer.fit(dataset.observations)
        if config["dataset"].get("normalize_observations", True)
        else None
    )
    agent = build_agent(
        config, dataset.observation_dim, dataset.action_dim, str(config.get("device", "auto"))
    )
    agent.load(checkpoint)
    return agent, normalizer
