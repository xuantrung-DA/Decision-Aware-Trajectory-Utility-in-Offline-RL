"""Standard-library-only setup; locate inputs by contents, not Kaggle dataset slug."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def contained_path(root, name):
    """Accept only portable relative bundle paths, including at the destination."""
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
    ):
        raise RuntimeError(f"Unsafe bundle path: {name!r}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"Bundle path escapes its root: {name!r}")
    return path


def setup(bundle, input_root, work):
    bundle, input_root, work = map(Path, (bundle, input_root, work))
    manifest = json.loads((bundle / "supplement_bundle.json").read_text(encoding="utf-8"))
    if manifest["study"] != "IJMLC_SUPP_V1":
        raise RuntimeError("Not the approved supplement bundle")
    # Validate every source and destination BEFORE copying any payload. A failed
    # restore must not leave half-updated code alongside existing study results.
    payload = []
    for name, expected in manifest["files"].items():
        source = contained_path(bundle, name)
        target = contained_path(work, name)
        if not source.is_file() or digest(source) != expected:
            raise RuntimeError(f"Bundle file missing/corrupt: {name}")
        if target.exists() and (not target.is_file() or digest(target) != expected):
            raise RuntimeError(
                f"Existing workspace code differs: {target}; do not overwrite a running study"
            )
        payload.append((source, target))
    expected_runs = json.loads(
        (bundle / "supplement_assets/baselines.json").read_text(encoding="utf-8")
    )
    folders = {}
    for path in input_root.rglob("run_manifest.json"):
        if path.parent.name in expected_runs:
            folders.setdefault(path.parent.name, []).append(path.parent)
    index = []
    missing = []
    for run_id, record in expected_runs.items():
        candidates = folders.get(run_id, [])
        selected = None
        for folder in candidates:
            if all(
                (folder / name).is_file() and digest(folder / name) == value
                for name, value in record["sha256"].items()
            ):
                selected = folder
                break
        if selected is None:
            missing.append(run_id)
        else:
            index.append({**record, "id": run_id, "path": str(selected)})
    if missing:
        print("Missing/changed checkpoint examples:", missing[:6])
        print("Attached input directories:", [str(p) for p in input_root.iterdir()])
        raise RuntimeError(
            "Need the EXPANDED results/runs folder from session 5 or final-86 input. "
            "Keep datu-phase3-frozen; do not launch training yet."
        )
    work.mkdir(parents=True, exist_ok=True)
    for source, target in payload:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    shutil.copy2(bundle / "supplement_bundle.json", work / "supplement_bundle.json")
    with (work / "supplement_input_index.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": index,
                "note": "read-only original checkpoints; original plan is never executed",
            },
            handle,
            indent=2,
        )
    print("Bundle verified:", manifest["protocol_signature"])
    print("Original IQL/ReBRAC checkpoints verified:", len(index), "/42")
    print("Writable supplement workdir:", work)
    print("CELL 1 PASS — session5 input is sufficient; no repeat of the 3 remaining CQL jobs")
    return work
