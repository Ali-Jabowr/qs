"""Still rings are reflections (ADR-0029 decision 3): a ring that stays still in the picture while
the chain moves is not a cup.

`StillRings.update(points)` takes the rims of one frame as (along, cross) positions: along is the
coordinate the chain's travel changes, larger being further along, and cross the other one. It
says which of them to follow. The chain is moving when some rim sits further along than a rim of
the previous frame in the same row by `still_px` or more. A rim is still when it sits within
`still_px` of such a rim. Then:

- a still rim first seen while the chain moves is a reflection, and is never followed, nor is any
  rim at that place after it (so it stays out through a stop of the chain);
- a rim not yet known to move is held back for `HOLD_FRAMES` frames, which is how a ring present
  before the first cup passes is told from a cup standing in a stop: the cup is followed once it
  has stood that long with the chain still, or as soon as it moves.

A cup that moves is followed one frame after it is first seen, as it has to be seen twice to be
known to move. Once followed, a rim stays followed while it stands, so a cup is never dropped for
standing still. A cup that stops exactly where a reflection was found is taken for it. All of
this is unmeasured on real footage and the thresholds are the config's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

HOLD_FRAMES = 3  # frames a rim must stand, the chain still, before it is taken for a cup


@dataclass(frozen=True, slots=True)
class _Rim:
    along: float
    cross: float
    count: int  # consecutive frames it has been still, this one included
    followed: bool


class StillRings:
    def __init__(self, still_px: float) -> None:
        self._still_px = still_px
        self._previous: list[_Rim] = []
        self._reflections: list[tuple[float, float]] = []

    def _match(self, along: float, cross: float) -> _Rim | None:
        """The previous frame's rim in the same row that is nearest behind, or at, `along`."""
        behind = [
            r for r in self._previous
            if abs(r.cross - cross) <= self._still_px and r.along <= along + self._still_px
        ]
        return min(behind, key=lambda r: abs(r.along - along), default=None)

    def _reflected(self, along: float, cross: float) -> bool:
        return any(
            abs(a - along) <= self._still_px and abs(c - cross) <= self._still_px
            for a, c in self._reflections
        )

    def update(self, points: Sequence[tuple[float, float]]) -> list[bool]:
        """Which of this frame's rims, in order, to follow."""
        matches = [self._match(*point) for point in points]
        moving = any(m is not None and a - m.along >= self._still_px
                     for (a, _), m in zip(points, matches, strict=True))
        current: list[_Rim] = []
        follow: list[bool] = []
        for (along, cross), match in zip(points, matches, strict=True):
            followed = match is not None and match.followed
            count = 1
            if self._reflected(along, cross):
                keep = False
            elif match is not None and along - match.along >= self._still_px:
                followed = keep = True  # it moves
            elif match is None:
                keep = False  # new: not yet known to move or stand
            else:
                count = match.count + 1
                if followed:
                    keep = True
                elif moving:
                    self._reflections.append((along, cross))
                    keep = False
                else:
                    followed = keep = count >= HOLD_FRAMES
            current.append(_Rim(along, cross, count, followed))
            follow.append(keep)
        self._previous = current
        return follow
