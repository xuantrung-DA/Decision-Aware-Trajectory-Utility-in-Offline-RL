"""Freeze controls and build a small, self-contained Kaggle supplement input."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from datu.datasets.minari_loader import load_processed_dataset
from datu.experiments.ijmlc_supplement import freeze_protocol, score_ablations
from datu.experiments.preregistration import load_preregistration
from datu.utils.io import atomic_json_dump, sha256_file

ROOT = Path(__file__).resolve().parents[1]

CELL_1 = """from pathlib import Path
import json, runpy, time

SUPP_SESSION_START = time.time()
TOTAL_BUDGET_HOURS = 4.25  # 4h15 TOTAL including setup; keeps margin below 4h42 quota
INPUT = Path("/kaggle/input")
markers = list(INPUT.rglob("supplement_bundle.json"))
assert len(markers) == 1, f"Add datu_ijmlc_supplement_input.zip as Input first. Found: {markers}"
BUNDLE = markers[0].parent
setup = runpy.run_path(str(BUNDLE / "scripts/setup_ijmlc_kaggle.py"))["setup"]
WORK = setup(BUNDLE, INPUT, Path("/kaggle/working/DATU_IJMLC_SUPP_V1"))
"""

CELL_2 = """import os, subprocess, sys, time
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Existing Kaggle PyTorch is retained if it already satisfies torch>=2.2.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[robotics]"],
               cwd=WORK, check=True)
subprocess.run([sys.executable, "-u", "scripts/run_ijmlc_supplement.py", "preflight"],
               cwd=WORK, check=True)
print("CELL 2 PASS — hashes, CUDA and original-checkpoint reproduction verified")
print("Budget remaining (hours):", round(TOTAL_BUDGET_HOURS - (time.time()-SUPP_SESSION_START)/3600, 2))
"""

CELL_3 = """import subprocess, sys, time
remaining = TOTAL_BUDGET_HOURS - (time.time() - SUPP_SESSION_START) / 3600
assert remaining > 0.75, "Too little time; do not start a new full-training job."
print("Starting SUPPLEMENT ONLY: 2 random groups x 3 seeds, ReBRAC 200k steps.")
print("Original 86-run study is not modified or rerun.")
subprocess.run([sys.executable, "-u", "scripts/run_ijmlc_supplement.py", "run",
                "--walltime-hours", str(remaining), "--buffer-minutes", "25"],
               cwd=WORK, check=True)
print("DOWNLOAD: /kaggle/working/datu_ijmlc_supplement_results.zip")
print("DOWNLOAD: /kaggle/working/datu_ijmlc_supplement_results.zip.sha256")
"""


def build(archive_path: Path, output: Path):
    dataset_path = ROOT / "data/processed/matched/pointmaze_medium/reference_critical_high"
    dataset = load_processed_dataset(dataset_path)
    original_path = ROOT / "data/manifests/lab_review_core_v1/pre_registration.json"
    original = load_preregistration(original_path)
    # Freeze group construction BEFORE reading policy outcomes or scoring ablations.
    protocol = freeze_protocol(dataset, original)
    analysis_dir = ROOT / "results/IJMLC_SUPP_V1_PREP"
    protocol_path = analysis_dir / "protocol.json"
    if (
        protocol_path.exists()
        and json.loads(protocol_path.read_text())["signature"] != protocol["signature"]
    ):
        raise RuntimeError("Refusing to replace a previously frozen supplementary protocol")
    atomic_json_dump(protocol, protocol_path)
    with zipfile.ZipFile(archive_path) as archive:
        baselines = {}
        manifests = {}
        for name in archive.namelist():
            if not name.endswith("/run_manifest.json"):
                continue
            record = json.loads(archive.read(name))
            algo, seed = record["algorithm"], record["seed"]
            block = record["experiment_block"]
            variant = record["experiment_variant"]
            if algo not in ("iql", "rebrac") or seed not in (0, 1, 2):
                continue
            if not (
                block == "B_MARGINAL"
                or (block == "A_CORE" and variant == "reference_critical_high")
            ):
                continue
            folder = name.rsplit("/", 1)[0]
            run_id = folder.rsplit("/", 1)[1]
            files = (
                "checkpoint_final.pt",
                "config.yaml",
                "normalization.json",
                "evaluation_metrics.json",
                "run_manifest.json",
            )
            baselines[run_id] = {
                "algorithm": algo,
                "seed": seed,
                "block": block,
                "variant": variant,
                "sha256": {
                    file: hashlib.sha256(archive.read(f"{folder}/{file}")).hexdigest()
                    for file in files
                },
            }
            manifests[algo, variant, seed] = record
        if len(baselines) != 42:
            raise RuntimeError(f"Expected 42 original baselines, got {len(baselines)}")
        base_folder = next(
            name.rsplit("/", 1)[0]
            for name in archive.namelist()
            if name.endswith("/run_manifest.json")
            and "_rebrac_reference_critical_high_seed0/" in name
        )
        base_config = archive.read(f"{base_folder}/config.yaml")
        component_bytes = archive.read("results/datu/pointmaze_core_reference/datu_components.npz")
    components = dict(np.load(io.BytesIO(component_bytes), allow_pickle=False))
    outcomes = {}
    for algo in ("iql", "rebrac"):
        outcomes[algo] = {
            group: float(
                np.mean(
                    [
                        manifests[algo, "reference_critical_high", seed]["evaluation"][
                            "success_rate"
                        ]
                        - manifests[algo, group, seed]["evaluation"]["success_rate"]
                        for seed in (0, 1, 2)
                    ]
                )
            )
            for group in original["intervention_groups"]
        }
    rows = score_ablations(dataset, components, original, outcomes, analysis_dir)
    for row in rows:
        print(row)
    payload = {}
    for path in sorted((ROOT / "src/datu").rglob("*.py")):
        payload[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in (
        "scripts/_bootstrap.py",
        "scripts/setup_ijmlc_kaggle.py",
        "scripts/run_ijmlc_supplement.py",
        "pyproject.toml",
        "LICENSE",
    ):
        payload[name] = (ROOT / name).read_bytes()
    payload["README.md"] = (
        b"# DATU IJMLC supplement\nSix frozen post-hoc ReBRAC random controls. Original runs unchanged.\n"
    )
    for path in dataset_path.iterdir():
        if path.is_file() and path.suffix in (".npy", ".json"):
            payload[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    payload["supplement_assets/protocol.json"] = protocol_path.read_bytes()
    payload["supplement_assets/baselines.json"] = json.dumps(baselines, indent=2).encode()
    payload["supplement_assets/rebrac_base.yaml"] = base_config
    payload["supplement_assets/original_preregistration.json"] = original_path.read_bytes()
    payload["supplement_assets/datu_components.npz"] = component_bytes
    payload["supplement_assets/component_score_ablation.csv"] = (
        analysis_dir / "component_score_ablation.csv"
    ).read_bytes()
    payload["supplement_assets/ablation.json"] = (analysis_dir / "ablation.json").read_bytes()
    manifest = {
        "study": "IJMLC_SUPP_V1",
        "protocol_signature": protocol["signature"],
        "source_archive_sha256": sha256_file(archive_path),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
    }
    payload["supplement_bundle.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(f"ijmlc_supplement/{name}", data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert all("\\" not in name for name in archive.namelist())
        assert len(archive.namelist()) == len(set(archive.namelist()))
    instructions = "# Kaggle IJMLC supplement\n\nKeep session5 and frozen inputs; add datu_ijmlc_supplement_input.zip (private).\nSelect GPU T4 x2 and Internet ON. Run cells 1-2 to check. For background training use Save Version -> Save & Run All; do not run cell 3 twice.\nBudget is 4h15 including setup, with a further 25m backup reserve.\n\n"
    for number, code in enumerate((CELL_1, CELL_2, CELL_3), 1):
        instructions += f"## Cell {number}\n\n```python\n{code}```\n\n"
    (output.parent / "kaggle_ijmlc_cells.md").write_text(instructions, encoding="utf-8")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "cell_type": "code",
                "id": f"cell-{i}",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
            for i, code in enumerate((CELL_1, CELL_2, CELL_3), 1)
        ],
    }
    atomic_json_dump(notebook, output.parent / "kaggle_ijmlc_supplement.ipynb")
    print("PROTOCOL", protocol["signature"])
    for name, group in protocol["groups"].items():
        print(name, json.dumps(group))
    print(
        "BUNDLE",
        output,
        "MiB",
        round(output.stat().st_size / 1024**2, 2),
        "SHA256",
        sha256_file(output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "datu_phase3_final_86jobs_kaggle_posix.zip"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dist/datu_ijmlc_supplement_input.zip"
    )
    args = parser.parse_args()
    build(args.archive, args.output)
