from __future__ import annotations

import argparse
import shutil
import subprocess

import _bootstrap  # noqa: F401

from datu.datasets.registry import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print verified Minari links; download only when --download is explicit."
    )
    parser.add_argument("dataset_ids", nargs="*", choices=sorted(DATASETS))
    parser.add_argument(
        "--download", action="store_true", help="Run Minari downloads after printing links"
    )
    args = parser.parse_args()
    selected = args.dataset_ids or list(DATASETS)
    for dataset_id in selected:
        record = DATASETS[dataset_id]
        print(f"\n{dataset_id}\n  role: {record['role']}\n  docs: {record['url']}")
        print(f"  manual command: minari download {dataset_id}")
    if args.download:
        executable = shutil.which("minari")
        if executable is None:
            raise RuntimeError("The minari executable is unavailable; install requirements first")
        for dataset_id in selected:
            subprocess.run([executable, "download", dataset_id], check=True)


if __name__ == "__main__":
    main()
