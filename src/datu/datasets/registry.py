from __future__ import annotations

from typing import TypedDict


class DatasetRecord(TypedDict):
    role: str
    url: str
    steps: int


DATASETS: dict[str, DatasetRecord] = {
    "D4RL/pointmaze/medium-v2": {
        "role": "primary controlled bottleneck experiments",
        "url": "https://minari.farama.org/datasets/D4RL/pointmaze/medium-v2/",
        "steps": 1_000_000,
    },
}
