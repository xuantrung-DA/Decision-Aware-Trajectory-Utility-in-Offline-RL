from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def suppress_loky_physical_core_probe() -> Iterator[None]:
    """Avoid noisy Windows physical-core probing while preserving parallelism."""
    logical_cores = os.cpu_count()
    existing_limit = os.environ.get("LOKY_MAX_CPU_COUNT")
    should_set_limit = existing_limit is None and logical_cores is not None and logical_cores > 1
    if should_set_limit:
        # A limit below the logical-core count makes Joblib skip its fragile
        # Windows physical-core probe. Keep nearly all available parallelism.
        os.environ["LOKY_MAX_CPU_COUNT"] = str(logical_cores - 1)
    try:
        yield
    finally:
        if should_set_limit:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
