from __future__ import annotations

import os

from datu.utils.config import load_config
from datu.utils.seed import set_seed


def test_config_include_and_override(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "training:\n  steps: 10\n  batch_size: 4\nalgorithm:\n  name: bc\n", encoding="utf-8"
    )
    child.write_text(
        "_base_: base.yaml\ndataset:\n  id: test\ntraining:\n  steps: 20\n", encoding="utf-8"
    )
    config = load_config(child, ["training.steps=30", "seed=7"])
    assert config["training"] == {"steps": 30, "batch_size": 4}
    assert config["algorithm"]["name"] == "bc"
    assert config["seed"] == 7


def test_deterministic_seed_configures_cublas(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(3, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
