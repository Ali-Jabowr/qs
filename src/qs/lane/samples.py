"""Which frames of a labelled video train which class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qs.labels import Cup, CupState

CLASSES = ("full", "empty", "none")

_CLASS_OF = {
    CupState.FULL: "full",
    CupState.FULL_WHITE: "full",
    CupState.FULL_PINK: "full",
    CupState.EMPTY: "empty",
    CupState.GAP: "none",
}


@dataclass(frozen=True, slots=True)
class Sample:
    """One training frame: its `video`, `frame`, class `cls`, and the `order` of its pass."""

    video: str
    frame: int
    cls: str
    order: int


def _claim(
    video: str,
    claimed: dict[int, tuple[str, str]],
    out: list[Sample],
    frame: int,
    cls: str,
    order: int,
    who: str,
) -> None:
    """Append one sample; a frame already claimed by another class raises, naming both claims."""
    if frame in claimed and claimed[frame][0] != cls:
        raise ValueError(
            f"{video}: {claimed[frame][1]} ({claimed[frame][0]}) and {who} ({cls}) "
            f"both claim frame {frame} with different classes"
        )
    claimed[frame] = (cls, who)
    out.append(Sample(video, frame, cls, order))


def samples(
    cups: Sequence[Cup],
    frame_counts: Mapping[str, int],
    *,
    half: int = 2,
    between: tuple[int, int] = (16, 48),
    fractions: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> list[Sample]:
    """The training frames of `cups`: each pass's own frames, and the space between two passes
    whose interval says the chain kept moving. Frames outside `frame_counts[video]` are dropped."""
    if half < 0:
        raise ValueError(f"half must be >= 0, not {half}")
    if any(not 0 <= f <= 1 for f in fractions):
        raise ValueError(f"fractions must all be within 0..1, not {fractions}")
    if between[0] > between[1]:
        raise ValueError(f"between must be (shortest, longest), not {between}")
    by_video: dict[str, list[Cup]] = {}
    for cup in cups:
        if cup.video not in frame_counts:
            raise ValueError(f"{cup.video}: pass {cup.order} has no entry in frame_counts")
        by_video.setdefault(cup.video, []).append(cup)
    out: list[Sample] = []
    for video, passes in by_video.items():
        count = frame_counts[video]
        claimed: dict[int, tuple[str, str]] = {}  # frame -> (class, who claimed it)

        passes = sorted(passes, key=lambda c: c.order)
        for i, cup in enumerate(passes):
            if i > 0:
                earlier = passes[i - 1]
                gap = cup.frame - earlier.frame
                if between[0] <= gap <= between[1]:
                    who = f"between pass {earlier.order} and pass {cup.order}"
                    for f in fractions:
                        frame = earlier.frame + round(gap * f)
                        _claim(video, claimed, out, frame, "none", cup.order, who)
            cls = _CLASS_OF[cup.state]
            for n in range(cup.frame - half, cup.frame + half + 1):
                if 0 <= n < count:
                    _claim(video, claimed, out, n, cls, cup.order, f"pass {cup.order}")
    return out
