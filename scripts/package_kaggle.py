from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from datu.datasets.minari_loader import load_processed_dataset
from datu.experiments.preregistration import load_preregistration

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_PATHS = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "configs/ablations/datu.yaml",
    "configs/algorithms",
    "configs/datasets/pointmaze_medium.yaml",
    "configs/experiments/lab_review_core_v1.yaml",
    "configs/experiments/pointmaze_iql.yaml",
    "configs/matched/pointmaze.yaml",
    "scripts/_bootstrap.py",
    "scripts/aggregate_results.py",
    "scripts/build_matched_datasets.py",
    "scripts/compute_dataset_metrics.py",
    "scripts/compute_datu.py",
    "scripts/download_datasets.py",
    "scripts/evaluate.py",
    "scripts/evaluate_utility_scores.py",
    "scripts/freeze_preregistration.py",
    "scripts/inspect_pointmaze.py",
    "scripts/package_kaggle.py",
    "scripts/prepare_dataset.py",
    "scripts/run_ablation.py",
    "scripts/run_experiment.py",
    "scripts/run_kaggle_session.py",
    "scripts/smoke_test.py",
    "scripts/train.py",
    "src",
    "tests",
    "data/processed/matched/pointmaze_medium",
    "data/processed/removals/pointmaze_core",
    "data/manifests/lab_review_core_v1",
    "results/dataset_audit/pointmaze_medium",
    "results/datu/pointmaze_core_reference",
)
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_RELATIVE_PATHS = {"src/datu/datasets/perturbations.py"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files() -> list[Path]:
    files: set[Path] = set()
    for value in INCLUDE_PATHS:
        source = ROOT / value
        if not source.exists():
            raise FileNotFoundError(f"Required Kaggle bundle path is absent: {source}")
        candidates = [source] if source.is_file() else source.rglob("*")
        for path in candidates:
            relative = path.relative_to(ROOT)
            if (
                path.is_file()
                and not any(part in EXCLUDED_NAMES for part in relative.parts)
                and path.suffix not in EXCLUDED_SUFFIXES
                and relative.as_posix() not in EXCLUDED_RELATIVE_PATHS
            ):
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _validate_frozen_inputs() -> tuple[dict, dict]:
    preregistration = load_preregistration(
        ROOT / "data/manifests/lab_review_core_v1/pre_registration.json"
    )
    expected: dict[str, str] = {}
    for name, fingerprint in preregistration["condition_hashes"].items():
        expected[f"data/processed/matched/pointmaze_medium/{name}"] = fingerprint
    for record in preregistration["intervention_groups"].values():
        expected[str(record["removal_dataset_path"]).replace("\\", "/")] = record[
            "removal_dataset_hash"
        ]
    if len(expected) != 10:
        raise ValueError(f"Expected 10 frozen datasets, found {len(expected)}")
    actual_hashes: dict[str, str] = {}
    for relative, fingerprint in expected.items():
        dataset = load_processed_dataset(ROOT / relative)
        actual = dataset.fingerprint()
        if actual != fingerprint:
            raise ValueError(f"Frozen dataset hash mismatch: {relative}")
        actual_hashes[relative] = actual

    plan_path = ROOT / "results/plans/LAB_REVIEW_CORE_V1/jobs.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            "Canonical Phase 2 plan is absent; run the 86-job preflight before packaging"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("job_count") != 86 or plan.get("total_training_updates") != 17_800_000:
        raise ValueError("Canonical plan is not the frozen 86-job/17.8M-update matrix")
    return preregistration, {
        "matrix_signature": plan["matrix_signature"],
        "dataset_hashes": actual_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minimal frozen Kaggle Phase 3 bundle")
    parser.add_argument("--output", default="dist/datu_kaggle_phase3.zip")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    preregistration, validation = _validate_frozen_inputs()
    files = _files()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = {
        "version": 1,
        "purpose": "LAB-REVIEW CORE v1 Kaggle policy training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "preregistration_hash": preregistration["preregistration_hash"],
        "matrix_signature": validation["matrix_signature"],
        "dataset_hashes": validation["dataset_hashes"],
        "file_count": len(records),
        "uncompressed_bytes": sum(record["bytes"] for record in records),
        "excluded": [
            ".venv",
            "raw Minari dataset",
            "full prepared 1M-transition source",
            "reference critic cache",
            "local smoke/results caches",
        ],
        "files": records,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, f"DATU/{path.relative_to(ROOT).as_posix()}")
        archive.writestr(
            "DATU/datu_bundle_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
        )
    archive_hash = _sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{archive_hash}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(output),
                "archive_sha256": archive_hash,
                "compressed_mb": round(output.stat().st_size / (1024**2), 2),
                "uncompressed_mb": round(manifest["uncompressed_bytes"] / (1024**2), 2),
                "files": len(files),
                "preregistration_hash": manifest["preregistration_hash"],
                "matrix_signature": manifest["matrix_signature"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
