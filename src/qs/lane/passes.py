"""Per-frame probabilities to passes, verdicts and pairing.

This is the pre-registered rule: every number in the final table goes through it, so it is
pure functions over plain values, with no model and no video.
"""

from __future__ import annotations

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
    if run.score is None:
        return Label.NO_VERDICT
    return Label.DEFECT if run.score > threshold else Label.PASS


def pair_runs(
    cups: Sequence[Cup], runs: Sequence[Run], window: int
) -> tuple[list[tuple[Cup, Run | None]], int]:
    """Each labelled pass, in frame order, with the first unused run lying within `window` frames
    of it (`first - window <= frame <= last + window`), or None. Also the number of runs left
    unpaired that start between the first and the last pass, frames included."""
    ordered = sorted(cups, key=lambda cup: cup.frame)
    used: set[int] = set()
    pairs: list[tuple[Cup, Run | None]] = []
    for cup in ordered:
        match = next(
            (
                i
                for i, run in enumerate(runs)
                if i not in used and run.first - window <= cup.frame <= run.last + window
            ),
            None,
        )
        if match is not None:
            used.add(match)
        pairs.append((cup, None if match is None else runs[match]))
    if not ordered:
        return pairs, 0
    low, high = ordered[0].frame, ordered[-1].frame
    unpaired = sum(1 for i, run in enumerate(runs) if i not in used and low <= run.first <= high)
    return pairs, unpaired
