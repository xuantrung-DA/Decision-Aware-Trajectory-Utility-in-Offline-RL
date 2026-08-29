from __future__ import annotations

import json

import numpy as np
import pytest

from datu.datasets.matched_builder import CriticalRegion, MatchedDatasetBuilder
from datu.datasets.trajectory import TransitionDataset
from datu.experiments.preregistration import (
    CORE_GROUPS,
    freeze_core_preregistration,
    load_preregistration,
    verify_preregistered_dataset,
)


def test_matched_dataset_builder_and_manifest(tiny_dataset: TransitionDataset) -> None:
    region = CriticalRegion((0, 1), (-0.2, -0.2), (0.2, 0.2))
    builder = MatchedDatasetBuilder(region, seed=2)
    conditions, manifest = builder.build(
        tiny_dataset,
        base_count=3,
        added_count=1,
        tolerances={
            "max_transition_error": 0.2,
            "max_mean_return_error_fraction": 0.2,
            "max_global_coverage_error_fraction": 1.0,
            "max_median_length_error_fraction": 0.2,
            "min_critical_coverage_separation": 0.01,
        },
    )
    assert set(conditions) == {
        "reference_critical_high",
        "critical_low",
        "helpful_low_return",
        "harmful_low_return",
    }
    assert len(set(manifest["condition_hashes"].values())) == 4
    assert all(len(condition.episode_ids) == 5 for condition in conditions.values())
    assert "errors" in manifest["comparisons"]["c3_helpful_vs_harmful"]
    assert manifest["source_hash"] == tiny_dataset.fingerprint()


def test_preregistration_freezes_exact_groups_and_detects_tampering(
    tiny_dataset: TransitionDataset, tmp_path
) -> None:
    region = CriticalRegion((0, 1), (-0.2, -0.2), (0.2, 0.2))
    conditions, manifest = MatchedDatasetBuilder(region, seed=2).build(
        tiny_dataset,
        base_count=3,
        added_count=1,
        tolerances={
            "max_transition_error": 0.2,
            "max_mean_return_error_fraction": 0.2,
            "max_global_coverage_error_fraction": 1.0,
            "max_median_length_error_fraction": 0.2,
            "min_critical_coverage_separation": 0.01,
        },
    )
    # The unit fixture is designed to exercise construction, not pass the much
    # tighter real-data tolerances. Freeze only after emulating the prior audit gate.
    manifest["all_comparisons_matched"] = True
    manifest_path = tmp_path / "matched.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reference = conditions["reference_critical_high"]
    components_path = tmp_path / "components.npz"
    np.savez(
        components_path,
        trajectory_scores=np.linspace(0.0, 1.0, reference.num_trajectories),
    )
    prereg_path = tmp_path / "pre_registration.json"
    artifact = freeze_core_preregistration(
        reference,
        manifest_path,
        components_path,
        tmp_path / "removals",
        prereg_path,
        group_size=1,
    )
    assert set(artifact["intervention_groups"]) == set(CORE_GROUPS)
    assert {item["trajectory_count"] for item in artifact["intervention_groups"].values()} == {
        1
    }
    loaded = load_preregistration(prereg_path)
    removal = loaded["intervention_groups"]["remove_top_datu"]
    reduced = TransitionDataset.load_directory(removal["removal_dataset_path"])
    verified = verify_preregistered_dataset(
        {
            "preregistration": {
                "path": str(prereg_path),
                "group": "remove_top_datu",
                "required": True,
            }
        },
        reduced,
    )
    assert verified is not None
    condition_verified = verify_preregistered_dataset(
        {
            "preregistration": {
                "path": str(prereg_path),
                "condition": "critical_low",
                "required": True,
            }
        },
        conditions["critical_low"],
    )
    assert condition_verified is not None

    tampered = json.loads(prereg_path.read_text(encoding="utf-8"))
    tampered["intervention_budget_trajectories"] = 2
    prereg_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_preregistration(prereg_path)
