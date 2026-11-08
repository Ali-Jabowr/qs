"""The time split of one camera position's good line frames into train, calibrate and test (#108).

Neighbouring frames of a video show the same cup, so a random split would put one cup on both
sides and hide false alarms. Each video is cut by time instead: the span of the frame ids
given for it, from the first to the last, gives the first 70% to training, the next 15% to
calibration and the last 15% to test, and the frames within `gap_s` after each cut are
dropped, so no cup crossing the view sits in two parts. The cuts depend on which ids are
given: `qs train` gives a video's `unit` frames and fits on `train`, so the false-alarm report
(#110) must give the same `unit` frames to read the other two parts at the same cuts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from qs.labels import FrameKind, Labels


class LineSplitError(ValueError):
    """A frame id the split cannot place."""


@dataclass(frozen=True, slots=True)
class LineSplit:
    """Frame ids by part, each ordered by video name, then frame index."""

    train: tuple[str, ...]
    calibrate: tuple[str, ...]
    test: tuple[str, ...]
    dropped: tuple[str, ...]


TRAIN_PERCENT = 70
CALIBRATE_END_PERCENT = 85


def video_frame(frame_id: str) -> tuple[str, int]:
    """A `<video>#<index>` frame id as its video file name and frame index."""
    video, sep, index = frame_id.rpartition("#")
    if not sep or not video or not index.isdigit():
        raise LineSplitError(f"{frame_id!r} is not a video frame (<video file name>#<index>)")
    return video, int(index)


def split_units(labels: Labels, *, fps: float, gap_s: float) -> LineSplit:
    """The split of the frames `labels` marks `unit`: the one `qs train` fits on and `qs
    line-alarms` reads back, so both always cut at the same frames."""
    good = [frame_id for frame_id, frame in labels.frames.items() if frame.label == FrameKind.UNIT]
    return split_frames(good, fps=fps, gap_s=gap_s)


def split_frames(frame_ids: Iterable[str], *, fps: float, gap_s: float) -> LineSplit:
    """Split `<video>#<index>` frame ids by time within each video, as the module says."""
    by_video: dict[str, list[int]] = {}
    for frame_id in frame_ids:
        video, index = video_frame(frame_id)
        by_video.setdefault(video, []).append(index)
    gap = round(gap_s * fps)
    parts: dict[str, list[str]] = {"train": [], "calibrate": [], "test": [], "dropped": []}
    for video in sorted(by_video):
        indices = sorted(by_video[video])
        first, span = indices[0], indices[-1] - indices[0] + 1
        train_end = first + span * TRAIN_PERCENT // 100
        calibrate_end = first + span * CALIBRATE_END_PERCENT // 100
        for index in indices:
            if index < train_end:
                part = "train"
            elif train_end + gap <= index < calibrate_end:
                part = "calibrate"
            elif index >= calibrate_end + gap:
                part = "test"
            else:
                part = "dropped"
            parts[part].append(f"{video}#{index}")
    return LineSplit(**{name: tuple(ids) for name, ids in parts.items()})
