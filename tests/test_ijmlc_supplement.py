from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from datu.experiments.ijmlc_supplement import (
    STUDY,
    fresh_fingerprint,
    load_protocol,
    make_training_config,
    matched_random_indices,
    signature,
)


@pytest.fixture
def lengths_and_target():
    lengths = np.arange(50, 230, dtype=np.int64)
    target = np.arange(0, 160, 4)
    return lengths, target


def test_controls_match_length_strata_and_removed_transition_budget(lengths_and_target):
    lengths, target = lengths_and_target
    outputs = []
    for seed in (2026090901, 2026090902):
        indices, audit = matched_random_indices(lengths, target, seed)
        assert len(indices) == len(np.unique(indices)) == 40
        assert (
            abs(int(lengths[indices].sum()) - int(lengths[target].sum()))
            <= audit["tolerance_transitions"]
        )
        bins = np.searchsorted(audit["length_bin_edges"], lengths, side="right")
        np.testing.assert_array_equal(np.bincount(bins[indices]), np.bincount(bins[target]))
        repeated, _ = matched_random_indices(lengths, target, seed)
        np.testing.assert_array_equal(indices, repeated)
        outputs.append(indices)
    assert not np.array_equal(*outputs)


def test_control_invalid_target_is_rejected(lengths_and_target):
    lengths, _ = lengths_and_target
    with pytest.raises(ValueError):
        matched_random_indices(lengths, np.array([1, 1]), 0)
    with pytest.raises(ValueError):
        matched_random_indices(lengths, np.array([-1, 20]), 0)


def test_protocol_tampering_fails(tmp_path):
    protocol = {"study": STUDY, "steps": 200000}
    protocol["signature"] = signature(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    assert load_protocol(path)["steps"] == 200000
    protocol["steps"] = 1000
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="signature"):
        load_protocol(path)


def test_fresh_hash_does_not_trust_cached_hash():
    class FakeDataset:
        _fingerprint_cache = "forged"

        def fingerprint(self):
            return self._fingerprint_cache or "computed"

    assert fresh_fingerprint(FakeDataset()) == "computed"


def test_supplement_config_keeps_scientific_settings_and_original_unchanged(tmp_path):
    base = {
        "seed": 99,
        "training": {"steps": 200000, "batch_size": 256},
        "algorithm": {"name": "rebrac", "critic_bc_coef": 0.1},
        "dataset": {"normalize_observations": True},
        "evaluation": {"episodes": 20, "seed": 10000},
        "logging": {"tensorboard": True},
        "preregistration": {"required": True},
    }
    before = copy.deepcopy(base)
    protocol = {"signature": "abc", "groups": {"random1": {"trajectory_indices": [3, 5]}}}
    job = {"seed": 1, "group": "random1"}
    new = make_training_config(base, protocol, job, tmp_path, tmp_path / "out")
    assert base == before
    assert new["algorithm"] == base["algorithm"]
    assert new["evaluation"] == base["evaluation"]
    assert new["training"] == base["training"]
    assert "preregistration" not in new
    assert new["supplement_protocol_signature"] == "abc"
    assert new["dataset"]["remove_trajectory_indices"] == [3, 5]


def script_module(name):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_archive_is_posix_and_retains_partial_recovery(tmp_path, monkeypatch):
    runner = script_module("run_ijmlc_supplement")
    root = tmp_path / "work"
    output = root / "results" / STUDY
    assets = root / "supplement_assets"
    assets.mkdir(parents=True)
    (assets / "protocol.json").write_text("{}")
    for name in ("finished", "partial"):
        folder = output / "runs" / name
        folder.mkdir(parents=True)
        (folder / "checkpoint_step_50000.pt").write_bytes(b"recoverable")
    (output / "runs/finished/run_manifest.json").write_text("{}")
    (output / "runs/finished/checkpoint_final.pt").write_bytes(b"final")
    for name, value in (("ROOT", root), ("OUTPUT", output), ("ASSETS", assets)):
        monkeypatch.setattr(runner, name, value)
    runner.package_output()
    with zipfile.ZipFile(tmp_path / "datu_ijmlc_supplement_results.zip") as archive:
        names = archive.namelist()
        assert all("\\" not in name for name in names)
        assert not any("finished/checkpoint_step" in name for name in names)
        assert any("partial/checkpoint_step" in name for name in names)
        assert any("finished/checkpoint_final" in name for name in names)


def test_summary_averages_random_draws_within_seed(tmp_path, monkeypatch):
    runner = script_module("run_ijmlc_supplement")
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    directory = tmp_path / "evaluation_100"
    directory.mkdir()
    for seed in (0, 1, 2):
        for variant, value in (
            ("reference_critical_high", 0.5),
            ("remove_top_datu", 0.3),
            ("random_1", 0.4),
            ("random_2", 0.6),
        ):
            (directory / f"{variant}_{seed}.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "algorithm": "rebrac",
                        "variant": variant,
                        "seed": seed,
                        "success_rate": value,
                    }
                )
            )
    report = runner.summarize({"groups": {"random_1": {}, "random_2": {}}})
    assert len(report["paired_seed_rows"]) == 3
    assert report["mean_random_minus_top_datu_pp"] == pytest.approx(20)
    (directory / "random_2_2.json").unlink()
    report = runner.summarize({"groups": {"random_1": {}, "random_2": {}}})
    assert not report["all_three_training_seeds_ready"]
    assert "mean_random_minus_top_datu_pp" not in report


def test_setup_accepts_nested_input_and_rejects_tampered_checkpoint(tmp_path):
    setup_module = script_module("setup_ijmlc_kaggle")
    bundle = tmp_path / "bundle"
    assets = bundle / "supplement_assets"
    assets.mkdir(parents=True)
    inputs = tmp_path / "input/datasets/user/arbitrary-slug/results/runs/study/run1"
    inputs.mkdir(parents=True)
    (inputs / "run_manifest.json").write_text("{}")
    (inputs / "checkpoint_final.pt").write_bytes(b"original")
    records = {
        "run1": {
            "sha256": {"checkpoint_final.pt": setup_module.digest(inputs / "checkpoint_final.pt")}
        }
    }
    baseline = assets / "baselines.json"
    baseline.write_text(json.dumps(records))
    manifest = {
        "study": STUDY,
        "protocol_signature": "abc",
        "files": {"supplement_assets/baselines.json": setup_module.digest(baseline)},
    }
    (bundle / "supplement_bundle.json").write_text(json.dumps(manifest))
    work = tmp_path / "work"
    setup_module.setup(bundle, tmp_path / "input", work)
    assert json.loads((work / "supplement_input_index.json").read_text())["runs"][0]["id"] == "run1"
    (inputs / "checkpoint_final.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="EXPANDED"):
        setup_module.setup(bundle, tmp_path / "input", work)


def test_notebook_cells_compile_and_budget_includes_setup():
    builder = script_module("build_ijmlc_bundle")
    for code in (builder.CELL_1, builder.CELL_2, builder.CELL_3):
        compile(code, "notebook_cell", "exec")
    assert "time.time() - SUPP_SESSION_START" in builder.CELL_3
    assert '"--buffer-minutes", "25"' in builder.CELL_3


@pytest.mark.parametrize(
    "name", ["../outside.py", "/absolute.py", "C:/absolute.py", "a\\b.py", "", "."]
)
def test_setup_rejects_nonportable_or_escaping_paths(tmp_path, name):
    setup_module = script_module("setup_ijmlc_kaggle")
    with pytest.raises(RuntimeError, match="Unsafe bundle path"):
        setup_module.contained_path(tmp_path, name)


def test_setup_checks_all_conflicts_before_copying_payload(tmp_path):
    setup_module = script_module("setup_ijmlc_kaggle")
    bundle, work = tmp_path / "bundle", tmp_path / "work"
    bundle.mkdir()
    work.mkdir()
    files = {}
    for name in ("first.py", "second.py"):
        (bundle / name).write_bytes(b"new-code")
        files[name] = setup_module.digest(bundle / name)
    (bundle / "supplement_bundle.json").write_text(
        json.dumps({"study": STUDY, "files": files}), encoding="utf-8"
    )
    (work / "second.py").write_bytes(b"existing-code")
    with pytest.raises(RuntimeError, match="Existing workspace code differs"):
        setup_module.setup(bundle, tmp_path / "input", work)
    assert not (work / "first.py").exists()
    assert (work / "second.py").read_bytes() == b"existing-code"


@pytest.mark.parametrize("timed_out", [True, False])
def test_budget_expiry_is_graceful_but_real_failure_is_not(tmp_path, monkeypatch, timed_out):
    runner = script_module("run_ijmlc_supplement")
    protocol = {"signature": "abc", "jobs": [{"id": "job1"}, {"id": "job2"}]}
    (tmp_path / "preflight.json").write_text(json.dumps({"protocol_signature": "abc"}))
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "context", lambda: (protocol, {}))
    monkeypatch.setattr(runner, "complete_job", lambda *_: False)
    monkeypatch.setattr(runner, "save_status", lambda *_, **__: {"complete": 0})
    backups, launched = [], []
    monkeypatch.setattr(runner, "package_output", lambda: backups.append(True))

    def child(*_):
        launched.append(True)
        return -2, timed_out

    monkeypatch.setattr(runner, "run_child", child)
    monkeypatch.setattr(runner, "evaluation_order", lambda *_: [])
    monkeypatch.setattr(runner, "summarize", lambda *_: {})
    if timed_out:
        runner.run_session(1.5, 25)
    else:
        with pytest.raises(RuntimeError, match="Training stopped"):
            runner.run_session(1.5, 25)
    assert len(launched) == 1
    assert len(backups) >= 2
