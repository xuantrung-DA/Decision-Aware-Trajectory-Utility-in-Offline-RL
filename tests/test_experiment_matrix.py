from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from datu.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_core_matrix_dry_run_is_exact_and_preregistered(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_experiment.py"),
            "--config",
            str(ROOT / "configs/experiments/lab_review_core_v1.yaml"),
            "--plan-dir",
            str(plan_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    plan = json.loads((plan_dir / "jobs.json").read_text(encoding="utf-8"))
    assert plan["job_count"] == 86
    assert plan["total_training_updates"] == 17_800_000
    assert plan["matrix_signature"] == (
        "a6ec37a53ef4b60a66520252a44625b50399982e402f6137383ea725310c1168"
    )
    assert Counter(job["block"] for job in plan["jobs"]) == {
        "A_CORE": 36,
        "B_MARGINAL": 36,
        "C_CQL_DIAGNOSTIC": 6,
        "D_HEADLINE_EXTRA_SEEDS": 8,
    }
    for job in plan["jobs"]:
        config = load_config(job["config"])
        assert config["evaluation"]["episodes"] == 20
        assert config["evaluation"]["max_episode_steps"] == 250
        assert config["preregistration"]["required"] is True
        if job["block"] == "B_MARGINAL":
            assert config["preregistration"]["group"] == job["variant"]
        else:
            assert config["preregistration"]["condition"] == job["variant"]


def test_component_ablation_dry_run_has_only_four_frozen_variants(tmp_path: Path) -> None:
    output = tmp_path / "ablations"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ablation.py"),
            "--config",
            str(ROOT / "configs/ablations/datu.yaml"),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads((output / "jobs.json").read_text(encoding="utf-8"))
    assert manifest["job_count"] == 4
    assert [job["name"] for job in manifest["jobs"]] == [
        "component_datu_full",
        "component_datu_no_uncertainty",
        "component_datu_no_sensitivity",
        "component_datu_no_local_support",
    ]
