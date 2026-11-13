"""Per-frame probabilities to passes, verdicts and pairing (stub)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qs.contracts import Label
from qs.labels import Cup


@dataclass(frozen=True, slots=True)
class Run:
    first: int
    last: int
    frames: int
    score: float | None


def runs_from_probs(
    probs: Sequence[Sequence[float]], *, min_frames: int = 3, max_gap: int = 1
) -> list[Run]:
    raise NotImplementedError


def verdict(run: Run, threshold: float) -> Label:
    raise NotImplementedError


def pair_runs(
    cups: Sequence[Cup], runs: Sequence[Run], window: int
) -> tuple[list[tuple[Cup, Run | None]], int]:
    raise NotImplementedError
