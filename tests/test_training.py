from __future__ import annotations

import csv

import pytest

from datu.training import make_environment, run_training


class RecoverableDataset:
    def __init__(self) -> None:
        self.requested_eval = None

    def recover_environment(self, eval_env: bool = False):
        self.requested_eval = eval_env
        return {"evaluation_environment": eval_env}


def test_minari_evaluation_environment_is_preferred() -> None:
    dataset = RecoverableDataset()
    environment = make_environment({}, dataset)
    assert dataset.requested_eval is True
    assert environment == {"evaluation_environment": True}


def test_training_prints_progress_and_reuses_metric_cache(tiny_dataset, tmp_path, capsys) -> None:
    dataset_path = tmp_path / "dataset"
    tiny_dataset.save_directory(dataset_path)
    config = {
        "seed": 3,
        "device": "cpu",
        "algorithm": {"name": "bc", "hidden_dims": [8, 8]},
        "dataset": {
            "path": str(dataset_path),
            "normalize_observations": True,
        },
        "training": {
            "steps": 2,
            "batch_size": 4,
            "log_interval": 1,
            "progress_interval": 1,
            "checkpoint_interval": 2,
            "checkpoint_selection": "final",
        },
        "logging": {
            "tensorboard": False,
            "dataset_metrics_cache": str(tmp_path / "metrics_cache"),
        },
    }
    first_dir, first_result = run_training(config, run_dir=tmp_path / "run_first", evaluate=False)
    first_output = capsys.readouterr().out
    assert "[progress]" in first_output
    assert "rate=" in first_output and "sample=" in first_output and "update=" in first_output
    assert "metrics_cache=miss/computed" in first_output
    assert first_result["overall_updates_per_second"] > 0
    with (first_dir / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "interval_updates_per_second" in rows[0]

    run_training(config, run_dir=tmp_path / "run_second", evaluate=False)
    second_output = capsys.readouterr().out
    assert "metrics_cache=hit" in second_output


def test_core_training_refuses_unfrozen_critical_region(tiny_dataset, tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    tiny_dataset.save_directory(dataset_path)
    config = {
        "seed": 0,
        "device": "cpu",
        "algorithm": {"name": "bc", "hidden_dims": [8, 8]},
        "dataset": {
            "path": str(dataset_path),
            "normalize_observations": True,
            "critical_region": {
                "frozen": False,
                "observation_dims": [0, 1],
                "lower": [-1.0, -1.0],
                "upper": [1.0, 1.0],
            },
        },
        "training": {"steps": 1, "batch_size": 4},
    }
    with pytest.raises(RuntimeError, match="unfrozen placeholder"):
        run_training(config, run_dir=tmp_path / "must_not_exist", evaluate=False)
    assert not (tmp_path / "must_not_exist").exists()
