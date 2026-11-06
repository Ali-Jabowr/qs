"""Line replay: saved frames played back as a camera on the line would show them, so that the
whole line station can run and be scored with no camera.

The files are one recording at a fixed `fps`, in name order: file *k* is what the camera shows from
`k / fps` after `open()`, on the clock supplied. `preview()` returns the frame showing now and
consumes nothing. `capture(event)` returns the first frame later than `event.t0` plus one exposure
(C4), read afresh from its file, so a preview's array never reaches the inspector (C9). Each file
is read and checked by `ReplaySource`, so a frame here means what it means in the bench's replay.

A camera waits for the frame it must return; replay has no reason to, so `capture()` returns at
once with `grabbed_at` set to the frame's time in the recording. A recording that ends before that
frame is a `CaptureError`: the unit becomes NO_VERDICT(`capture`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qs.config import StationConfig
from qs.contracts import Frame, Health, TriggerEvent
from qs.cup_trigger import CupTrigger
from qs.errors import CaptureError
from qs.sources.replay import Clock, ReplayError, ReplaySource, WallClock, utc_now

_log = logging.getLogger(__name__)

# `ReplaySource` chooses a file by the event's `seq` and uses nothing else of it.
_READ_FILE = TriggerEvent(0, 0.0, datetime(1970, 1, 1, tzinfo=UTC), "line-replay")


class LineReplaySource:
    """`FrameSource` over one recording, paced by the clock."""

    def __init__(
        self, config: StationConfig, files: Sequence[Path], *, fps: float, clock: Clock
    ) -> None:
        self._files = ReplaySource(config, files, clock=clock)
        self._count = len(files)
        self._fps = fps
        self._exposure_s = config.camera.exposure_us / 1e6
        self._clock = clock
        self._start: float | None = None  # the recording plays on through a reconnect
        self._shown: tuple[int, np.ndarray | None] | None = None

    def _read(self, index: int) -> Frame:
        return self._files.capture(replace(_READ_FILE, seq=index + 1))

    def _index_now(self) -> int | None:
        if self._start is None:
            return None
        return int((self._clock() - self._start) * self._fps)

    @property
    def finished(self) -> bool:
        """True once the clock has passed the last frame."""
        index = self._index_now()
        return index is not None and index >= self._count

    def open(self) -> None:
        self._files.open()
        if self._start is None:
            self._start = self._clock()

    def capture(self, event: TriggerEvent) -> Frame:
        if self._start is None:
            raise CaptureError("capture() called before open()")
        index = int((event.t0 + self._exposure_s - self._start) * self._fps) + 1
        if index >= self._count:
            raise CaptureError(f"the recording ended before a frame after seq {event.seq}")
        frame = self._read(index)
        return replace(frame, event=event, grabbed_at=self._start + index / self._fps)

    def preview(self) -> np.ndarray | None:
        """The frame showing now, the same array until the next one is due. None before `open()`,
        past the last frame, or for a frame that cannot be read, which is logged once."""
        index = self._index_now()
        if index is None or index >= self._count:
            return None
        if self._shown is None or self._shown[0] != index:
            try:
                self._shown = (index, self._read(index).pixels)
            except CaptureError as exc:
                _log.warning("frame %d cannot be previewed: %s", index, exc)
                self._shown = (index, None)
        return self._shown[1]

    def health(self) -> Health:
        return self._files.health()

    def close(self) -> None:
        self._files.close()


def line_replay_pair(
    config: StationConfig,
    files: Sequence[Path],
    name: str,
    *,
    clock: Clock,
    wall: WallClock = utc_now,
) -> tuple[CupTrigger, LineReplaySource]:
    """The cup trigger reading the line replay's previews; the trigger ends when the files do."""
    line = config.inspector.line
    if line is None or line.fps is None:
        raise ReplayError("line replay: inspector.line needs fps")
    source = LineReplaySource(config, files, fps=line.fps, clock=clock)
    trigger = CupTrigger(
        line, source, source_id=f"replay:{name}", clock=clock, wall=wall,
        ended=lambda: source.finished,
    )
    return trigger, source
