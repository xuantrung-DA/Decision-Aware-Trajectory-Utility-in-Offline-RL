from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config include detected at {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Config does not exist: {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level config must be a mapping: {resolved}")

    merged: dict[str, Any] = {}
    includes = data.pop("_base_", [])
    if isinstance(includes, str):
        includes = [includes]
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = resolved.parent / include_path
        merged = deep_merge(merged, _load_yaml(include_path, seen))
    seen.remove(resolved)
    return deep_merge(merged, data)


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply Hydra-like ``a.b=value`` overrides to a copy of a config."""
    result = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError(f"Empty override key: {item!r}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            child = cursor.get(key)
            if child is None:
                child = {}
                cursor[key] = child
            if not isinstance(child, dict):
                raise TypeError(f"Cannot descend into non-mapping key: {dotted_key}")
            cursor = child
        cursor[keys[-1]] = _parse_value(raw_value)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load YAML with recursive ``_base_`` includes and command-line overrides."""
    return apply_overrides(_load_yaml(Path(path), set()), overrides)


def validate_training_config(config: Mapping[str, Any]) -> None:
    required = ("algorithm", "dataset", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required training config sections: {missing}")
    if int(config["training"].get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(config["training"].get("steps", 0)) <= 0:
        raise ValueError("training.steps must be positive")
