from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from datu.datasets.splits import group_trajectories, split_by_trajectory
from datu.datasets.trajectory import TransitionDataset, flatten_observation, trajectory_from_episode


def test_flatten_and_episode_extraction() -> None:
    episode = SimpleNamespace(
        observations={
            "observation": np.arange(15, dtype=np.float32).reshape(5, 3),
            "desired_goal": np.ones((5, 2), dtype=np.float32),
        },
        actions=np.zeros((4, 2), dtype=np.float32),
        rewards=np.arange(4, dtype=np.float32),
        terminations=np.array([False, False, False, True]),
        truncations=np.zeros(4, dtype=np.bool_),
        infos={},
    )
    trajectory = trajectory_from_episode(episode, "episode-1", ["observation", "desired_goal"])
    assert len(trajectory) == 4
    assert trajectory.observations.shape == (4, 5)
    assert trajectory.next_observations.shape == (4, 5)
    assert flatten_observation(episode.observations).shape == (5, 5)


def test_dataset_round_trip_and_group_removal(tiny_dataset: TransitionDataset, tmp_path) -> None:
    npz_path = tmp_path / "dataset.npz"
    tiny_dataset.save_npz(npz_path)
    from_npz = TransitionDataset.load_npz(npz_path)
    assert from_npz.fingerprint() == tiny_dataset.fingerprint()
    directory = tmp_path / "array_dataset"
    tiny_dataset.save_directory(directory)
    memory_mapped = TransitionDataset.load_directory(directory)
    assert memory_mapped.fingerprint() == tiny_dataset.fingerprint()
    reduced = tiny_dataset.remove_trajectories([0, 2])
    assert reduced.num_trajectories == tiny_dataset.num_trajectories - 2
    assert "0" not in reduced.episode_ids and "2" not in reduced.episode_ids


def test_seeded_split_and_groups(tiny_dataset: TransitionDataset) -> None:
    train_a, validation_a = split_by_trajectory(tiny_dataset, 0.25, seed=4)
    train_b, validation_b = split_by_trajectory(tiny_dataset, 0.25, seed=4)
    assert train_a.episode_ids == train_b.episode_ids
    assert validation_a.episode_ids == validation_b.episode_ids
    groups = group_trajectories(tiny_dataset, 4, strategy="random", seed=3)
    assert groups.shape == (tiny_dataset.num_trajectories,)
    assert set(groups) == {0, 1, 2, 3}


def test_sampling_is_batch_local_and_respects_episode_boundaries(
    tiny_dataset: TransitionDataset, monkeypatch
) -> None:
    class FixedIndexSource:
        def integers(self, low, high, size):
            del low, high, size
            first_stop = int(tiny_dataset.trajectory_offsets[1])
            return np.asarray([0, first_stop - 1, first_stop], dtype=np.int64)

    def forbidden_full_array(_self):
        raise AssertionError("sample() must not materialize a full-dataset derived array")

    monkeypatch.setattr(TransitionDataset, "next_actions", property(forbidden_full_array))
    monkeypatch.setattr(TransitionDataset, "dones", property(forbidden_full_array))
    batch = tiny_dataset.sample(3, FixedIndexSource(), "cpu")
    first_stop = int(tiny_dataset.trajectory_offsets[1])
    expected_next_indices = [1, first_stop - 1, first_stop + 1]
    assert np.allclose(batch["next_actions"].numpy(), tiny_dataset.actions[expected_next_indices])
    assert batch["dones"].shape == (3, 1)
