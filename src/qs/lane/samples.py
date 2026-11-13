"""Which frames of a labelled video train which class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qs.labels import Cup

CLASSES = ("full", "empty", "none")


@dataclass(frozen=True, slots=True)
class Sample:
    """One training frame: its `video`, `frame`, class `cls`, and the `order` of its pass."""

    video: str
    frame: int
    cls: str
    order: int


def samples(
    cups: Sequence[Cup],
    frame_counts: Mapping[str, int],
    *,
    half: int = 2,
    between: tuple[int, int] = (16, 48),
    fractions: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> list[Sample]:
    raise NotImplementedError
