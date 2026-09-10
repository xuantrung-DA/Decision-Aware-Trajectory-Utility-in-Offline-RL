from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from datu.datasets.minari_loader import require_minari


def test_kaggle_package_contains_supplement_test_dependencies(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("package_kaggle", scripts / "package_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "scripts/build_ijmlc_bundle.py",
        "scripts/run_ijmlc_supplement.py",
        "scripts/setup_ijmlc_kaggle.py",
    }
    # Phase 3 bundles include the entire tests/ tree; importing its supplement
    # tests must not depend on scripts left behind on the developer's machine.
    assert required.issubset(module.INCLUDE_PATHS)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "INCLUDE_PATHS", (*sorted(required), "src", "tests"))
    for name in (*required, "src/datu/__init__.py", "tests/test_ijmlc_supplement.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test payload\n", encoding="utf-8")
    for name in ("src/datu/datasets/perturbations.py", "src/datu/__pycache__/cached.pyc"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded")
    files = {p.relative_to(tmp_path).as_posix() for p in module._files()}
    assert required.issubset(files)
    assert "tests/test_ijmlc_supplement.py" in files
    assert not any("__pycache__" in name or "perturbations" in name for name in files)


def test_missing_minari_message_points_to_public_readme(monkeypatch):
    monkeypatch.setitem(sys.modules, "minari", None)
    with pytest.raises(RuntimeError, match=r"README\.md"):
        require_minari()
