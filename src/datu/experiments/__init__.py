"""Experiment-freezing and execution safeguards."""

from datu.experiments.preregistration import (
    freeze_core_preregistration,
    load_preregistration,
    verify_preregistered_dataset,
)

__all__ = [
    "freeze_core_preregistration",
    "load_preregistration",
    "verify_preregistered_dataset",
]
