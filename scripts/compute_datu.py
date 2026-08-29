from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from datu.datasets.splits import group_trajectories
from datu.evaluation.plotting import plot_spatial_landscape
from datu.training import load_dataset_from_config
from datu.utility.datu import DATUConfig, compute_datu_scores, datu_variants
from datu.utility.decision_sensitivity import (
    CandidateActionGenerator,
    decision_sensitivity,
    evaluate_candidate_q,
)
from datu.utility.local_coverage import group_support_contribution
from datu.utility.uncertainty import BootstrappedIQLCriticEnsemble, q_uncertainty
from datu.utils.config import load_config
from datu.utils.io import atomic_json_dump, system_info
from datu.utils.normalization import ObservationNormalizer, normalize_dataset
from datu.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute offline-only DATU trajectory scores")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--critic-steps", type=int, default=100_000)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--score-progress-interval", type=int, default=50000)
    parser.add_argument("--ensemble-cache")
    parser.add_argument("--aggregation", default="topk_mean")
    parser.add_argument("--combination", default="product")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.progress_interval <= 0 or args.score_progress_interval <= 0:
        raise ValueError("Progress intervals must be positive")
    seed = int(config.get("seed", 0))
    set_seed(seed)
    started = time.perf_counter()
    raw_dataset, _ = load_dataset_from_config(config)
    normalizer = ObservationNormalizer.fit(raw_dataset.observations)
    dataset = normalize_dataset(raw_dataset, normalizer)
    assignments = group_trajectories(raw_dataset, args.groups, strategy="structure", seed=seed)
    transition_groups = assignments[raw_dataset.trajectory_ids]

    ensemble = BootstrappedIQLCriticEnsemble(
        dataset.observation_dim,
        dataset.action_dim,
        ensemble_size=args.ensemble_size,
        hidden_dims=config["algorithm"].get("hidden_dims", [256, 256]),
        device=str(config.get("device", "auto")),
        seed=seed,
    )
    cache_used = False
    reference_final_metrics = None
    if args.ensemble_cache and Path(args.ensemble_cache).exists():
        print(f"[reference] loading critic cache={args.ensemble_cache}", flush=True)
        cache_manifest_path = Path(args.ensemble_cache) / "ensemble_manifest.json"
        if not cache_manifest_path.exists():
            raise FileNotFoundError(f"Cached ensemble has no manifest: {cache_manifest_path}")
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("dataset_hash") != raw_dataset.fingerprint():
            raise ValueError("Cached ensemble dataset hash differs from the active dataset")
        if int(cache_manifest.get("ensemble_size", -1)) != args.ensemble_size:
            raise ValueError("Cached ensemble size differs from --ensemble-size")
        ensemble.load(args.ensemble_cache)
        cache_used = True
    else:
        print(
            f"[reference] fitting ensemble={args.ensemble_size} "
            f"steps_per_member={args.critic_steps:,} device={ensemble.members[0].device}",
            flush=True,
        )
        reference_final_metrics = ensemble.fit(
            dataset,
            args.critic_steps,
            int(config["training"].get("batch_size", 256)),
            progress_interval=args.progress_interval,
        )
        if args.ensemble_cache:
            ensemble.save(args.ensemble_cache)
            atomic_json_dump(
                {
                    "dataset_hash": raw_dataset.fingerprint(),
                    "ensemble_size": args.ensemble_size,
                    "critic_steps_per_member": args.critic_steps,
                    "seed": seed,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "device": str(ensemble.members[0].device),
                },
                Path(args.ensemble_cache) / "ensemble_manifest.json",
            )
    print(f"[score] fitting local candidate index k={args.knn_k}", flush=True)
    generator = CandidateActionGenerator(k=args.knn_k).fit(
        dataset.observations, dataset.actions, seed=seed
    )
    sensitivities, uncertainties = [], []
    score_start = time.perf_counter()
    next_score_report = args.score_progress_interval
    for start in range(0, len(dataset), args.chunk_size):
        stop = min(start + args.chunk_size, len(dataset))
        observations = dataset.observations[start:stop]
        policy_actions = ensemble.policy_actions(observations)
        candidates = generator.generate(observations, policy_actions)
        candidate_values = evaluate_candidate_q(ensemble.predict, observations, candidates)
        sensitivities.append(decision_sensitivity(candidate_values))
        uncertainties.append(q_uncertainty(candidate_values)["std"].mean(axis=1))
        if stop >= next_score_report or stop == len(dataset):
            rate = stop / max(time.perf_counter() - score_start, 1e-12)
            eta = (len(dataset) - stop) / max(rate, 1e-12)
            print(
                f"[score-progress] sensitivity_uncertainty={stop:,}/{len(dataset):,} "
                f"rate={rate:.1f} transitions/s eta={eta / 60.0:.1f}m",
                flush=True,
            )
            while next_score_report <= stop:
                next_score_report += args.score_progress_interval

    support_start = time.perf_counter()
    next_support_report = args.score_progress_interval

    def report_support(processed: int, total: int) -> None:
        nonlocal next_support_report
        if processed >= next_support_report or processed == total:
            rate = processed / max(time.perf_counter() - support_start, 1e-12)
            eta = (total - processed) / max(rate, 1e-12)
            print(
                f"[score-progress] local_support={processed:,}/{total:,} "
                f"rate={rate:.1f} transitions/s eta={eta / 60.0:.1f}m",
                flush=True,
            )
            while next_support_report <= processed:
                next_support_report += args.score_progress_interval

    print("[score] computing marginal local support", flush=True)
    components = {
        "sensitivity": np.concatenate(sensitivities),
        "uncertainty": np.concatenate(uncertainties),
        "local_support": group_support_contribution(
            raw_dataset.observations,
            raw_dataset.actions,
            transition_groups,
            k=args.knn_k,
            query_batch_size=args.chunk_size,
            progress_callback=report_support,
        ),
    }
    datu_section = config.get("datu", {})
    variant = datu_section.get("variant")
    if variant:
        if variant not in datu_variants():
            raise ValueError(f"Unknown DATU variant: {variant}")
        base_variant = datu_variants()[variant]
    else:
        base_variant = DATUConfig()
    datu_config = DATUConfig(
        use_sensitivity=bool(datu_section.get("use_sensitivity", base_variant.use_sensitivity)),
        use_uncertainty=bool(datu_section.get("use_uncertainty", base_variant.use_uncertainty)),
        use_local_support=bool(
            datu_section.get("use_local_support", base_variant.use_local_support)
        ),
        combination=str(datu_section.get("combination", args.combination)),
        aggregation=str(datu_section.get("aggregation", args.aggregation)),
        topk_fraction=float(datu_section.get("topk_fraction", base_variant.topk_fraction)),
        percentile=float(datu_section.get("percentile", base_variant.percentile)),
        temporal_discount=float(
            datu_section.get("temporal_discount", base_variant.temporal_discount)
        ),
    )
    point_scores, trajectory_scores = compute_datu_scores(
        components, dataset.trajectory_offsets, datu_config
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "datu_components.npz",
        point_scores=point_scores,
        trajectory_scores=trajectory_scores,
        trajectory_groups=assignments,
        **components,
    )
    with (output / "trajectory_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_index", "episode_id", "group", "datu"])
        for index, score in enumerate(trajectory_scores):
            writer.writerow([index, raw_dataset.episode_ids[index], assignments[index], score])
    summary = {
        "dataset_hash": raw_dataset.fingerprint(),
        "offline_only": True,
        "oracle_environment_information_used": False,
        "critic_steps_per_member": args.critic_steps,
        "ensemble_size": args.ensemble_size,
        "ensemble_cache": args.ensemble_cache,
        "ensemble_cache_used": cache_used,
        "knn_k": args.knn_k,
        "groups": args.groups,
        "config": datu_config.__dict__,
        "reference_final_metrics": reference_final_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "active_device": str(ensemble.members[0].device),
        "system": system_info(),
    }
    atomic_json_dump(summary, output / "datu_manifest.json")
    if raw_dataset.observation_dim >= 2:
        plot_spatial_landscape(
            raw_dataset.observations,
            point_scores,
            output / "datu_landscape.png",
            "DATU landscape",
            "DATU",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
