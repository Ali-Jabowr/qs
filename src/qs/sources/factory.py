"""The `FrameSource` for a station config: one `match` on `camera.type` (docs/11 §8)."""

from __future__ import annotations

from qs.config import StationConfig
from qs.contracts import FrameSource
from qs.sources.webcam import WebcamSource


class SourceUnavailable(Exception):
    """The config names a camera type with no source built.

    Not a `QSError`: those are the Pipeline's NO_VERDICT reasons; this stops a tool at start.
    """


def frame_source(config: StationConfig) -> FrameSource:
    """The source the config names; only the webcam exists (ADR-0010)."""
    match config.camera.type:
        case "webcam":
            return WebcamSource(config)
        case other:
            raise SourceUnavailable(
                f"camera.type {other!r} has no source — only webcam is built; the phone "
                "source comes only if the lock test fails (ADR-0010)"
            )
