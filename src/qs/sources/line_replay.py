"""Line replay: saved frames played back as a camera on the line would show them, so that the
whole line station can run and be scored with no camera.

The files are one recording at a fixed `fps`, in name order: file *k* is what the camera shows from
`k / fps` after `open()`, on the clock supplied. `preview()` returns the frame showing now and
consumes nothing. `capture(event)` returns the first frame later than `event.t0` plus one exposure
(C4), read afresh from its file, so a preview's array never reaches the inspector (C9). Each file
is read and checked by `ReplaySource`, so a frame here means what it means in the bench's replay.
A recording saved as a video plays the same way (`VideoReplaySource`): frame *k* of the video is
file *k*, and each frame gets the same checks as a file's pixels.

A camera waits for the frame it must return; replay has no reason to, so `capture()` returns at
once with `grabbed_at` set to the frame's time in the recording. A recording that ends before that
frame is a `CaptureError`: the unit becomes NO_VERDICT(`capture`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from qs.config import StationConfig
from qs.contracts import CameraReadback, Frame, Health, TriggerEvent
from qs.cup_trigger import CupTrigger
from qs.errors import CaptureError
from qs.labels import FrameReadError, VideoFrames
from qs.sources.replay import (
    READ_BY_INDEX,
    Clock,
    ReplayError,
    ReplaySource,
    WallClock,
    check_pixels,
    utc_now,
)
from qs.sources.webcam import read_reference

_log = logging.getLogger(__name__)


class RecordedFrames(Protocol):
    """A recording's frames by index from 0, each read and checked afresh as a camera's would be."""

    count: int

    def open(self) -> None: ...

    def read(self, index: int) -> Frame: ...

    def health(self) -> Health: ...

    def close(self) -> None: ...


class _Paced:
    """`FrameSource` over one recording's frames, paced by the clock."""

    def __init__(
        self, config: StationConfig, frames: RecordedFrames, *, fps: float, clock: Clock
    ) -> None:
        self._frames = frames
        self._count = frames.count
        self._fps = fps
        self._exposure_s = config.camera.exposure_us / 1e6
        self._clock = clock
        self._start: float | None = None  # the recording plays on through a reconnect
        self._shown: tuple[int, np.ndarray | None] | None = None

    def _index_now(self) -> int | None:
        if self._start is None:
            return None
        return int((self._clock() - self._start) * self._fps)

    @property
    def fps(self) -> float:
        """The rate the recording is played at."""
        return self._fps

    @property
    def finished(self) -> bool:
        """True once the clock has passed the last frame."""
        index = self._index_now()
        return index is not None and index >= self._count

    def open(self) -> None:
        self._frames.open()
        if self._start is None:
            self._start = self._clock()

    def frame_index(self, event: TriggerEvent) -> int:
        """The index of the frame `capture(event)` returns: the first after one exposure; it may
        be past the last frame. Only after `open()`."""
        return int((event.t0 + self._exposure_s - self._started()) * self._fps) + 1

    def _started(self) -> float:
        if self._start is None:
            raise CaptureError("capture() called before open()")
        return self._start

    def capture(self, event: TriggerEvent) -> Frame:
        index = self.frame_index(event)
        if index >= self._count:
            raise CaptureError(f"the recording ended before a frame after seq {event.seq}")
        frame = self._frames.read(index)
        return replace(frame, event=event, grabbed_at=self._started() + index / self._fps)

    def preview(self) -> np.ndarray | None:
        """The frame showing now, the same array until the next one is due. None before `open()`,
        past the last frame, or for a frame that cannot be read, which is logged once."""
        index = self._index_now()
        if index is None or index >= self._count:
            return None
        if self._shown is None or self._shown[0] != index:
            try:
                self._shown = (index, self._frames.read(index).pixels)
            except CaptureError as exc:
                _log.warning("frame %d cannot be previewed: %s", index, exc)
                self._shown = (index, None)
        return self._shown[1]

    def health(self) -> Health:
        return self._frames.health()

    def close(self) -> None:
        self._frames.close()


class LineReplaySource(_Paced):
    """`FrameSource` over one recording saved as image files, paced by the clock."""

    def __init__(
        self, config: StationConfig, files: Sequence[Path], *, fps: float, clock: Clock
    ) -> None:
        super().__init__(config, ReplaySource(config, files, clock=clock), fps=fps, clock=clock)


class _Video:
    """`RecordedFrames` over one video file, each frame checked as replay checks a file's."""

    def __init__(self, config: StationConfig, path: Path) -> None:
        self._config = config
        try:
            self._video = VideoFrames(path)
        except FrameReadError as exc:
            raise ReplayError(f"line replay: {exc}") from exc
        self.count = len(self._video.ids)

    def open(self) -> None:
        return None  # the video was opened to count its frames

    def read(self, index: int) -> Frame:
        name = self._video.ids[index]
        try:
            pixels = check_pixels(name, self._video.read(index), self._config)
        except FrameReadError as exc:
            raise CaptureError(str(exc)) from exc
        return Frame(
            event=READ_BY_INDEX,
            pixels=pixels,
            grabbed_at=0.0,
            camera=CameraReadback(None, None, None, None, verified=False),
            reference_patch=read_reference(pixels, self._config),
            capture_ms=0.0,
        )

    def health(self) -> Health:
        return Health(True, "replaying")

    def close(self) -> None:
        self._video.close()


class VideoReplaySource(_Paced):
    """`FrameSource` over one recording saved as a video, paced by the clock. The video's
    frames are read at the config's `inspector.line.fps`, whatever rate the file states."""

    def __init__(self, config: StationConfig, path: Path, *, fps: float, clock: Clock) -> None:
        super().__init__(config, _Video(config, path), fps=fps, clock=clock)


def _cup_pair(
    config: StationConfig,
    source: Callable[[float], _Paced],
    name: str,
    clock: Clock,
    wall: WallClock,
) -> tuple[CupTrigger, _Paced]:
    line = config.inspector.line
    if line is None or line.fps is None:
        raise ReplayError("line replay: inspector.line needs fps")
    paced = source(line.fps)
    trigger = CupTrigger(
        line, paced, source_id=f"replay:{name}", clock=clock, wall=wall,
        ended=lambda: paced.finished,
    )
    return trigger, paced


def video_replay_pair(
    config: StationConfig, path: Path, *, clock: Clock, wall: WallClock = utc_now
) -> tuple[CupTrigger, _Paced]:
    """The cup trigger reading a video's previews; the trigger ends when the video does."""
    return _cup_pair(
        config, lambda fps: VideoReplaySource(config, path, fps=fps, clock=clock),
        path.name, clock, wall,
    )


def line_replay_pair(
    config: StationConfig,
    files: Sequence[Path],
    name: str,
    *,
    clock: Clock,
    wall: WallClock = utc_now,
) -> tuple[CupTrigger, _Paced]:
    """The cup trigger reading the line replay's previews; the trigger ends when the files do."""
    return _cup_pair(
        config, lambda fps: LineReplaySource(config, files, fps=fps, clock=clock),
        name, clock, wall,
    )
