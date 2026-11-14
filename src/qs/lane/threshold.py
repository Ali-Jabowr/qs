"""The pre-registered threshold rule and the threshold file (#158)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Threshold:
    """The chosen value and every input it came from."""

    model_id: str
    far_pct: int
    value: float
    n_full: int
    n_above_half: int
    n_unseen: int
    labels_sha256: str


def lane_threshold(scores: Sequence[float], far_pct: int) -> float:
    return 0.0  # stub


def write_threshold(threshold: Threshold, path: Path) -> None:
    return None  # stub


def load_threshold(path: Path) -> Threshold:
    return Threshold("", 0, 0.0, 0, 0, 0, "")  # stub
