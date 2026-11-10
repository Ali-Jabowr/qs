"""The cup trigger: a `TriggerSource` that sees cups in preview frames (ADR-0028).

`CupTracker` follows cups by position with no images. The chain moves one way, so a cup found in
a frame is matched to the nearest followed cup behind it, or ahead of it by less than the
tolerance, which only absorbs a wobble in the fitted rim. A cup with no match is a new cup. Each
followed cup fires once: when first seen wholly in the inspection zone or, having never been, when
it has been gone from view for `LOST_AFTER_S`. A rim missed for a moment, to glare say, is
therefore the same cup when it comes back.

A rim hidden for longer than `LOST_AFTER_S` is forgotten, so on a stopped chain the same cup can
come back as a new cup and be judged twice. This is kept on purpose: remembering it longer would
let the next cup on a moving chain be taken for it and get no record at all, and a duplicate is
visible (`qs score-line` counts it among the records paired with no cup) where a lost cup is not.

`CupTrigger` feeds the tracker with `find_rims` on each new preview. A preview is only ever looked
at for finding: the frame judged is whatever `capture(event)` returns afterwards, one frame on, and
the inspector looks for the cup there itself. A cup first wholly in the zone is near the zone's
near edge, so it is still in the zone a frame later provided the zone is at least a rim and two
frames of travel long.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from qs.config import LineConfig
from qs.contracts import FrameSource, TriggerEvent
from qs.errors import ConfigError
from qs.inspector.cup import find_rims
from qs.still_rings import StillRings

LOST_AFTER_S = 0.25  # unmeasured on real footage; cups pass about every 1–1.5 s, by estimate
# Per direction: the rim centre's coordinate that travel changes, and the sign that makes "further
# along" the larger number.
_TRAVEL = {"right": (0, 1), "left": (0, -1), "down": (1, 1), "up": (1, -1)}


@dataclass(frozen=True, slots=True)
class SeenCup:
    """One cup in one frame: how far along the direction of travel, and if wholly in the zone."""

    along: float
    in_zone: bool


@dataclass(eq=False, slots=True)
class _Track:
    along: float
    last_seen: float
    fired: bool = False


class CupTracker:
    """`fires`, `update` and `finish` say how many events to fire; an event names no cup."""

    def __init__(self, tolerance_px: float) -> None:
        self._tolerance_px = tolerance_px
        self._tracks: list[_Track] = []

    def _nearest(self, along: float, free: list[_Track]) -> _Track | None:
        behind = [t for t in free if t.along <= along + self._tolerance_px]
        return min(behind, key=lambda t: abs(t.along - along), default=None)

    def update(self, seen: Sequence[SeenCup], now: float) -> int:
        """One frame's cups: how many events, of either kind, to fire."""
        return sum(self.fires(seen, now))

    def fires(self, seen: Sequence[SeenCup], now: float) -> tuple[int, int]:
        """One frame's cups: events for cups first wholly in the zone, and for cups gone from
        view that never were. The furthest along are matched first, so each takes its own track."""
        in_zone = lost = 0
        free = list(self._tracks)
        for cup in sorted(seen, key=lambda c: c.along, reverse=True):
            track = self._nearest(cup.along, free)
            if track is None:
                track = _Track(cup.along, now)
                self._tracks.append(track)
            else:
                free.remove(track)
                track.along = cup.along
                track.last_seen = now
            if cup.in_zone and not track.fired:
                track.fired = True
                in_zone += 1
        for track in free:
            if now - track.last_seen > LOST_AFTER_S:
                self._tracks.remove(track)
                lost += not track.fired
        return in_zone, lost

    def finish(self) -> int:
        """No more frames: every followed cup is gone, and those not yet fired fire now, as cups
        that never were wholly in the zone."""
        fires = sum(not track.fired for track in self._tracks)
        self._tracks.clear()
        return fires


class CupTrigger:
    """`TriggerSource` over a `FrameSource`'s previews. `seq` from 1, gapless; `t0` the clock at
    the `poll()` that returns the event. It looks at each preview array once, so the source must
    return the same array until it has a new frame. `ended` says that no frame will ever follow."""

    def __init__(
        self,
        line: LineConfig,
        source: FrameSource,
        *,
        source_id: str,
        clock: Callable[[], float],
        wall: Callable[[], datetime],
        ended: Callable[[], bool],
    ) -> None:
        if line.direction is None:
            raise ConfigError("inspector.line.direction: a cup trigger needs this key")
        self._axis, self._sign = _TRAVEL[line.direction]
        self._line = line
        self._source = source
        self._source_id = source_id
        self._clock = clock
        self._wall = wall
        self._ended = ended
        # Cups cannot overlap, so a position less than half the smallest rim behind a followed
        # cup can only be that same cup, whatever the camera's resolution.
        self._tracker = CupTracker(line.rim_axis_px[0] / 2)
        # With still_ring_px set, a ring that stays still while the chain moves is left out.
        self._still = None if line.still_ring_px is None else StillRings(
            line.still_ring_px,
            line.rim_axis_px[0] / 2,
            (
                line.still_ring_px + line.max_speed_px_s / line.fps
                if line.max_speed_px_s and line.fps else float("inf")
            ),
            LOST_AFTER_S,
        )
        self._looked_at: np.ndarray | None = None
        self._pending = 0  # events not yet returned, of both kinds
        self._pending_lost = 0  # of those, for cups never wholly in the zone; returned last
        self._seq = 0
        self._finished = False

    def _seen(self, pixels: np.ndarray) -> list[SeenCup]:
        rims = find_rims(pixels, self._line)
        if self._still is not None:
            follow = self._still.update(
                [(self._sign * r.ellipse[0][self._axis], r.ellipse[0][1 - self._axis])
                 for r in rims],
                self._clock(),
            )
            rims = tuple(r for r, keep in zip(rims, follow, strict=True) if keep)
        return [SeenCup(self._sign * rim.ellipse[0][self._axis], rim.in_zone) for rim in rims]

    def poll(self) -> TriggerEvent | None:
        """Never blocks. A frame that makes two cups fire gives one event on each of two polls."""
        if not self._finished:
            pixels = self._source.preview()
            if pixels is None and self._ended():
                self._add(0, self._tracker.finish())
                self._finished = True
            elif pixels is not None and pixels is not self._looked_at:
                self._looked_at = pixels
                self._add(*self._tracker.fires(self._seen(pixels), self._clock()))
        if self._pending == 0:
            return None
        never_in_zone = self._pending == self._pending_lost
        self._pending -= 1
        self._pending_lost -= never_in_zone
        self._seq += 1
        return TriggerEvent(
            self._seq, self._clock(), self._wall(), self._source_id, never_in_zone=never_in_zone
        )

    def _add(self, in_zone: int, lost: int) -> None:
        self._pending += in_zone + lost
        self._pending_lost += lost

    @property
    def exhausted(self) -> bool:
        return self._finished and self._pending == 0

    def close(self) -> None:
        """No more events, including any not yet returned."""
        self._finished = True
        self._pending = 0
        self._pending_lost = 0
