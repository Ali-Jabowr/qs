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
from qs.inspector.cup import Rim, find_rims

LOST_AFTER_S = 0.25  # unmeasured on real footage; cups pass about every 1–1.5 s, by estimate
# Per direction: the rim centre's coordinate that travel changes, and the sign that makes "further
# along" the larger number.
_COLD_FRAMES_TO_END = 2  # frames in a row below the threshold that end a warm stretch
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
        self.crossed = 0  # cups that crossed `line` in the last `fires`

    def _nearest(self, along: float, free: list[_Track]) -> _Track | None:
        behind = [t for t in free if t.along <= along + self._tolerance_px]
        return min(behind, key=lambda t: abs(t.along - along), default=None)

    def update(self, seen: Sequence[SeenCup], now: float) -> int:
        """One frame's cups: how many events, of either kind, to fire."""
        return sum(self.fires(seen, now))

    def fires(
        self, seen: Sequence[SeenCup], now: float, line: float | None = None
    ) -> tuple[int, int]:
        """One frame's cups: events for cups first wholly in the zone, and for cups gone from
        view that never were. The furthest along are matched first, so each takes its own track.
        With `line`, a position along the travel, `crossed` then holds how many followed cups went
        from below it to at or above it in this frame; a cup first seen is not one of them."""
        in_zone = lost = 0
        self.crossed = 0
        free = list(self._tracks)
        for cup in sorted(seen, key=lambda c: c.along, reverse=True):
            track = self._nearest(cup.along, free)
            if track is None:
                track = _Track(cup.along, now)
                self._tracks.append(track)
            else:
                free.remove(track)
                self.crossed += line is not None and track.along < line <= cup.along
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


def _slit_warmth(pixels: np.ndarray, slit: tuple[int, int, int, int]) -> float:
    """The mean of red minus blue (BGR frames) over the whole slit, as `inspector.line.warmth` takes
    it over a cup's inner half; the slit's crop is all that is converted."""
    x, y, w, h = slit
    crop = pixels[y : y + h, x : x + w].astype(np.int16)
    return float((crop[..., 2] - crop[..., 0]).mean())


class UntrackedWatch:
    """The self-check of ADR-0029 §7: warmth in a thin slit across the chain, beside the finder.
    A stretch starts when the slit's warmth is `warmth_min` or more and ends after two frames in a
    row below it, so a reading that flickers around the threshold is one stretch. `update` takes a
    frame's warmth and whether a tracked rim centre crossed the slit's centre line in it, and says
    if an untracked stretch has just ended: one that no crossing during it credited. A crossing
    while the slit is cold credits nothing. A stretch still warm at the last frame is not ended."""

    def __init__(self, warmth_min: float) -> None:
        self._warmth_min = warmth_min
        self._active = False
        self._credited = False
        self._cold_run = 0

    def update(self, warmth: float, crossed: bool) -> bool:
        warm = warmth >= self._warmth_min
        if not self._active:
            if warm:
                self._active = True
                self._credited = crossed
                self._cold_run = 0
            return False
        self._credited |= crossed
        self._cold_run = 0 if warm else self._cold_run + 1
        if self._cold_run < _COLD_FRAMES_TO_END:
            return False
        self._active = False
        return not self._credited


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
        # The self-check (ADR-0029 §7): a slit across the travel, and the position along the travel
        # of its centre line, which a tracked cup's rim centre must cross to credit a warm stretch.
        self._slit = line.untracked_window
        self._watch: UntrackedWatch | None = None
        self._centre_line: float | None = None
        if self._slit is not None and line.warmth_min is not None:
            x, y, w, h = self._slit
            self._centre_line = self._sign * ((x + w / 2) if self._axis == 0 else (y + h / 2))
            self._watch = UntrackedWatch(line.warmth_min)
        self._looked_at: np.ndarray | None = None
        self._pending = 0  # events not yet returned, of both kinds
        self._pending_lost = 0  # of those, for cups never wholly in the zone
        self._pending_untracked = 0  # of those, for untracked passages; returned last of all
        self._seq = 0
        self._finished = False

    def _seen(self, rims: Sequence[Rim]) -> list[SeenCup]:
        return [SeenCup(self._sign * rim.ellipse[0][self._axis], rim.in_zone) for rim in rims]

    def _untracked(self, pixels: np.ndarray) -> bool:
        """The slit's warmth in this frame and the cups that crossed its centre line, said to the
        watch; whether an untracked stretch has just ended."""
        if self._slit is None or self._watch is None:
            return False
        return self._watch.update(_slit_warmth(pixels, self._slit), self._tracker.crossed > 0)

    def poll(self) -> TriggerEvent | None:
        """Never blocks. A frame that makes two cups fire gives one event on each of two polls."""
        if not self._finished:
            pixels = self._source.preview()
            if pixels is None and self._ended():
                self._add(0, self._tracker.finish())
                self._finished = True
            elif pixels is not None and pixels is not self._looked_at:
                self._looked_at = pixels
                rims = find_rims(pixels, self._line)
                self._add(*self._tracker.fires(self._seen(rims), self._clock(), self._centre_line))
                if self._untracked(pixels):
                    self._pending += 1
                    self._pending_untracked += 1
        if self._pending == 0:
            return None
        untracked = self._pending == self._pending_untracked
        never_in_zone = (
            not untracked and self._pending - self._pending_untracked == self._pending_lost
        )
        self._pending -= 1
        self._pending_lost -= never_in_zone
        self._pending_untracked -= untracked
        self._seq += 1
        return TriggerEvent(
            self._seq,
            self._clock(),
            self._wall(),
            self._source_id,
            never_in_zone=never_in_zone,
            untracked=untracked,
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
        self._pending_untracked = 0
