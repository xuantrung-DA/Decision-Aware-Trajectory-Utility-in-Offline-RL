from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from datu.datasets.trajectory import TransitionDataset, trajectory_from_episode


def require_minari() -> Any:
    try:
        import minari
    except ImportError as exc:
        raise RuntimeError(
            "Minari is required for public datasets. Install requirements, then download the "
            "dataset manually with the command in README.md (Reproduction)."
        ) from exc
    return minari


def available_local_datasets() -> list[str]:
    minari = require_minari()
    if hasattr(minari, "list_local_datasets"):
        datasets = minari.list_local_datasets()
        return sorted(datasets if isinstance(datasets, list) else list(datasets))
    return []


def load_minari_dataset(
    dataset_id: str,
    observation_keys: Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> tuple[TransitionDataset, Any]:
    """Load an already-downloaded Minari dataset without implicit network access."""
    minari = require_minari()
    try:
        dataset = minari.load_dataset(dataset_id, download=False)
    except TypeError:
        dataset = minari.load_dataset(dataset_id)
    except Exception as exc:
        raise FileNotFoundError(
            f"Minari dataset {dataset_id!r} is unavailable locally. Download it manually first."
        ) from exc

    if hasattr(dataset, "iterate_episodes"):
        iterator: Iterable[Any] = dataset.iterate_episodes()
    elif hasattr(dataset, "episodes"):
        iterator = iter(dataset.episodes)
    else:
        raise TypeError("Unsupported Minari dataset API: no episode iterator found")

    trajectories = []
    for index, episode in enumerate(iterator):
        if max_episodes is not None and index >= max_episodes:
            break
        episode_id = getattr(episode, "id", getattr(episode, "episode_id", index))
        trajectories.append(trajectory_from_episode(episode, episode_id, observation_keys))
    if not trajectories:
        raise ValueError(f"Dataset {dataset_id!r} yielded no trajectories")
    metadata = {"minari_id": dataset_id, "source": "minari"}
    return TransitionDataset.from_trajectories(trajectories, metadata), dataset


def load_processed_dataset(path: str | Path) -> TransitionDataset:
    source = Path(path)
    if source.is_dir():
        return TransitionDataset.load_directory(source)
    if source.suffix.lower() == ".npz":
        return TransitionDataset.load_npz(source)
    raise ValueError(f"Processed dataset must be an .npz file or array directory: {source}")


def recover_environment(minari_dataset: Any, **kwargs: Any) -> Any:
    if not hasattr(minari_dataset, "recover_environment"):
        raise TypeError("This Minari dataset cannot recover its Gymnasium environment")
    return minari_dataset.recover_environment(**kwargs)
