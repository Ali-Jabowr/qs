"""Stub: filled in after the tests are committed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qs.labels import Labels
from qs.lane.split import Split


@dataclass(frozen=True, slots=True)
class Defect:
    video: str
    frame: int
    kind: str
    unit: str


def load_defects(path: Path, labels: Labels, split: Split) -> tuple[Defect, ...]:
    raise NotImplementedError
