"""Still rings are reflections (ADR-0029 decision 3): a ring that stays still in the picture while
the chain moves is not a cup.

`StillRings.update(points, now)` takes the rims of one frame as (along, cross) positions: along is
the coordinate the chain's travel changes, larger being further along, and cross the other one. It
says which of them to follow. The default is to follow: a false record shows on the screen, a cup
that was never judged does not.

A rim is left out only when it matches a confirmed reflection. A spot is a place seen on one frame
and the next: within `still_px` along and `across_px` across the travel of the last time. It is
confirmed once it has been seen on `CONFIRM_FRAMES` frames while the chain moved. The chain moved
when the median step of the other rims is over `still_px`, a step being a rim's move along since
the previous frame, with rims matched one to one and no step over `max_step_px`. A spot not seen
for `forget_after_s` is forgotten, confirmed or not, so a place is never blind for good.

Limits: a ring and a cup both still from the first frame cannot be told apart, and a cup that stops
exactly on a confirmed reflection is taken for it. Unmeasured on real footage.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

CONFIRM_FRAMES = 3  # frames a spot is seen while the chain moves, to be a reflection


@dataclass(slots=True)
class _Spot:
    along: float
    cross: float
    last_seen: float
    moving_frames: int = 0

    @property
    def confirmed(self) -> bool:
        return self.moving_frames >= CONFIRM_FRAMES


class StillRings:
    def __init__(
        self, still_px: float, across_px: float, max_step_px: float, forget_after_s: float
    ) -> None:
        self._still_px = still_px
        self._across_px = across_px
        self._max_step_px = max_step_px
        self._forget_after_s = forget_after_s
        self._previous: list[tuple[float, float]] = []
        self._spots: list[_Spot] = []

    def _steps(self, points: Sequence[tuple[float, float]]) -> dict[int, float]:
        """Each rim's move along since the previous frame, rims matched one to one, smallest
        move first."""
        pairs = sorted(
            (abs(a - pa), i, j)
            for i, (a, c) in enumerate(points)
            for j, (pa, pc) in enumerate(self._previous)
            if abs(c - pc) <= self._across_px and -self._still_px <= a - pa <= self._max_step_px
        )
        steps: dict[int, float] = {}
        taken: set[int] = set()
        for _, i, j in pairs:
            if i not in steps and j not in taken:
                steps[i] = points[i][0] - self._previous[j][0]
                taken.add(j)
        return steps

    def _spot_at(self, along: float, cross: float) -> _Spot | None:
        near = [
            s for s in self._spots
            if abs(s.along - along) <= self._still_px and abs(s.cross - cross) <= self._across_px
        ]
        return min(near, key=lambda s: abs(s.along - along), default=None)

    def update(self, points: Sequence[tuple[float, float]], now: float) -> list[bool]:
        """Which of this frame's rims, in order, to follow."""
        self._spots = [s for s in self._spots if now - s.last_seen <= self._forget_after_s]
        steps = self._steps(points)
        follow: list[bool] = []
        for i, (along, cross) in enumerate(points):
            others = [step for j, step in steps.items() if j != i]
            moving = bool(others) and statistics.median(others) > self._still_px
            spot = self._spot_at(along, cross)
            if spot is None:
                spot = _Spot(along, cross, now)
                self._spots.append(spot)
            spot.along, spot.cross, spot.last_seen = along, cross, now
            spot.moving_frames += moving
            follow.append(not spot.confirmed)
        self._previous = list(points)
        return follow
