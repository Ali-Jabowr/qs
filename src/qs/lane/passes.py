"""Per-frame probabilities to passes, verdicts and pairing.

This is the pre-registered rule: every number in the final table goes through it, so it is
pure functions over plain values, with no model and no video.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from qs.contracts import Label
from qs.labels import Cup

# A frame is a cup frame when P(none) is strictly below this.
_NONE_LIMIT = 0.5


@dataclass(frozen=True, slots=True)
class Run:
    """A stretch of cup frames: first and last frame index, the cup frames in it, and the median
    P(empty) over them; `score` is None when there are too few frames to judge."""

    first: int
    last: int
    frames: int
    score: float | None


def runs_from_probs(
    probs: Sequence[Sequence[float]], *, min_frames: int = 3, max_gap: int = 1
) -> list[Run]:
    """The runs in `probs` (one `(P(full), P(empty), P(none))` row per frame, frame 0 first).

    Up to `max_gap` non-cup frames in a row inside a run are bridged; more split it. A run with
    fewer than `min_frames` cup frames is kept but has no score.
    """
    if min_frames < 1:
        raise ValueError(f"min_frames must be at least 1, got {min_frames}")
    if max_gap < 0:
        raise ValueError(f"max_gap must not be negative, got {max_gap}")
    runs: list[Run] = []
    first = last = -1
    empties: list[float] = []

    def close() -> None:
        if empties:
            score = statistics.median(empties) if len(empties) >= min_frames else None
            runs.append(Run(first, last, len(empties), score))

    for index, row in enumerate(probs):
        if len(row) != 3:
            raise ValueError(f"frame {index}: expected 3 probabilities, got {len(row)}")
        if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in row):
            raise ValueError(f"frame {index}: probabilities must be finite, in 0..1: {tuple(row)}")
        if row[2] >= _NONE_LIMIT:
            continue
        if empties and index - last - 1 > max_gap:
            close()
            empties = []
        if not empties:
            first = index
        last = index
        empties.append(float(row[1]))
    close()
    return runs


def verdict(run: Run, threshold: float) -> Label:
    """DEFECT when the score is above `threshold` (not at it), PASS otherwise; NO_VERDICT when the
    run has no score."""
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    if run.score is None:
        return Label.NO_VERDICT
    if not math.isfinite(run.score):
        raise ValueError(f"run {run.first}-{run.last}: score must be finite, got {run.score}")
    return Label.DEFECT if run.score > threshold else Label.PASS


def pair_runs(
    cups: Sequence[Cup], runs: Sequence[Run], window: int
) -> tuple[list[tuple[Cup, Run | None]], int]:
    """Each labelled pass, by `(frame, order)`, with an unused run, or None; `runs` are taken in
    frame order. The run that holds the pass's frame wins; failing that, the nearest run within
    `window` frames of it (distance to `[first, last]`), the earlier one on a tie. Also the number
    of runs left unpaired that overlap the span from the first pass to the last."""
    if len({cup.video for cup in cups}) > 1:
        raise ValueError("cups must all come from one video, as the runs do")
    ordered = sorted(cups, key=lambda cup: (cup.frame, cup.order))
    by_frame = sorted(runs, key=lambda run: run.first)
    used: set[int] = set()
    pairs: list[tuple[Cup, Run | None]] = []
    for cup in ordered:
        free = [i for i in range(len(by_frame)) if i not in used]
        holding = [i for i in free if by_frame[i].first <= cup.frame <= by_frame[i].last]
        near = [(_distance(by_frame[i], cup.frame), i) for i in free]
        within = [pair for pair in near if pair[0] <= window]
        match = holding[0] if holding else min(within)[1] if within else None
        if match is not None:
            used.add(match)
        pairs.append((cup, None if match is None else by_frame[match]))
    if not ordered:
        return pairs, 0
    low, high = ordered[0].frame, ordered[-1].frame
    unpaired = sum(
        1
        for i, run in enumerate(by_frame)
        if i not in used and run.last >= low and run.first <= high
    )
    return pairs, unpaired


def _distance(run: Run, frame: int) -> int:
    """Frames from `frame` to the nearest end of `run`; 0 inside it."""
    return max(run.first - frame, frame - run.last, 0)
