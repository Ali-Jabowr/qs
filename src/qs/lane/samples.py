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
    by_video: dict[str, list[Cup]] = {}
    for cup in cups:
        by_video.setdefault(cup.video, []).append(cup)
    out: list[Sample] = []
    for video, passes in by_video.items():
        count = frame_counts[video]
        passes = sorted(passes, key=lambda c: c.order)
        for i, cup in enumerate(passes):
            if i > 0:
                earlier = passes[i - 1].frame
                if between[0] <= cup.frame - earlier <= between[1]:
                    gap = cup.frame - earlier
                    out += [
                        Sample(video, earlier + round(gap * f), "none", cup.order)
                        for f in fractions
                    ]
            cls = _CLASS_OF[cup.state]
            out += [
                Sample(video, n, cls, cup.order)
                for n in range(cup.frame - half, cup.frame + half + 1)
                if 0 <= n < count
            ]
    return out
