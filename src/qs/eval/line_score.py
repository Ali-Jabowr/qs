"""Scoring line footage against hand-counted cups (#105).

Each labelled cup is paired with the station's record nearest to it within a window of frames,
taken in order: a record is used once, and a cup never seen takes none, so it cannot shift the
cups after it onto the wrong records. Then the pairs are counted per kind of cup. Every rate
carries the project's one-sided 95% bound (`qs.eval.stats`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass
from statistics import median

from qs.contracts import Label
from qs.eval.stats import lower_bound, upper_bound
from qs.labels import Cup, CupState


@dataclass(frozen=True, slots=True)
class Judged:
    """One record of the line station: the video frame it judged, and its verdict."""

    frame: int
    label: Label


@dataclass(frozen=True, slots=True)
class Tally:
    total: int = 0
    defect: int = 0
    passed: int = 0
    no_verdict: int = 0
    unseen: int = 0


@dataclass(frozen=True, slots=True)
class CupScore:
    """Empty and full cups; white and pink are full cups whose flavour was labelled. `unpaired`
    counts records inside the labelled span that no cup was paired with. `gap` is the chain
    positions with no cup, never part of `full`."""

    empty: Tally
    full: Tally
    white: Tally
    pink: Tally
    unpaired: int
    offsets: tuple[int, ...]
    gap: Tally = Tally()


def _tally(pairs: Sequence[tuple[Cup, Judged | None]]) -> Tally:
    labels = [None if judged is None else judged.label for _, judged in pairs]
    return Tally(
        total=len(labels),
        defect=labels.count(Label.DEFECT),
        passed=labels.count(Label.PASS),
        no_verdict=labels.count(Label.NO_VERDICT),
        unseen=labels.count(None),
    )


def pair_cups(
    cups: Sequence[Cup], judged: Sequence[Judged], window: int
) -> list[tuple[Cup, Judged | None]]:
    """Each cup, in order, with the nearest record within `window` frames of it that comes after
    the previous cup's record; None when there is none. A record is used at most once."""
    records = sorted(judged, key=lambda j: j.frame)
    pairs: list[tuple[Cup, Judged | None]] = []
    after = 0  # index of the first record a later cup may still take
    for this in sorted(cups, key=lambda c: (c.frame, c.order)):
        near = [
            i for i in range(after, len(records))
            if abs(records[i].frame - this.frame) <= window
        ]
        best = min(near, key=lambda i: abs(records[i].frame - this.frame), default=None)
        pairs.append((this, None if best is None else records[best]))
        if best is not None:
            after = best + 1
    return pairs


def score_cups(cups: Sequence[Cup], judged: Sequence[Judged], window: int) -> CupScore:
    """The counts, per kind of cup, of how the station judged the cups it was paired with."""
    pairs = pair_cups(cups, judged, window)
    first = min(c.frame for c in cups) - window
    last = max(c.frame for c in cups) + window
    paired = {id(j) for _, j in pairs if j is not None}
    return CupScore(
        empty=_tally([p for p in pairs if p[0].state == CupState.EMPTY]),
        full=_tally([p for p in pairs if p[0].state not in (CupState.EMPTY, CupState.GAP)]),
        white=_tally([p for p in pairs if p[0].state == CupState.FULL_WHITE]),
        pink=_tally([p for p in pairs if p[0].state == CupState.FULL_PINK]),
        gap=_tally([p for p in pairs if p[0].state == CupState.GAP]),
        unpaired=sum(id(j) not in paired and first <= j.frame <= last for j in judged),
        offsets=tuple(j.frame - c.frame for c, j in pairs if j is not None),
    )


def combined(scores: Sequence[CupScore]) -> CupScore:
    """The scores of several videos as one: every count summed."""

    def total(tallies: Sequence[Tally]) -> Tally:
        return Tally(*(sum(column) for column in zip(*(astuple(t) for t in tallies), strict=True)))

    return CupScore(
        empty=total([s.empty for s in scores]),
        full=total([s.full for s in scores]),
        white=total([s.white for s in scores]),
        pink=total([s.pink for s in scores]),
        gap=total([s.gap for s in scores]),
        unpaired=sum(s.unpaired for s in scores),
        offsets=tuple(o for s in scores for o in s.offsets),
    )


def _rate(what: str, k: int, n: int, *, good: bool) -> str:
    """`k of n` with its one-sided 95% bound: the lower bound on an outcome wanted, the upper
    bound on one not wanted."""
    if n == 0:
        return f"- {what}: none labelled"
    bound = f"lower bound {lower_bound(n, k) * 100:.1f}%" if good else (
        f"upper bound {upper_bound(n, k) * 100:.1f}%")
    return f"- {what}: {k} of {n} = {k / n * 100:.1f}%, {bound}"


def _offset_lines(offsets: Sequence[int], window: int) -> list[str]:
    """The median record-minus-label frame, warned about when over half the window: a pair can
    then fall to the wrong side of it, and a cup never seen take the next cup's record."""
    if not offsets:
        return ["- record frame minus labelled frame: no pairs"]
    middle = median(offsets)
    lines = [f"- record frame minus labelled frame: median {middle:.1f} over {len(offsets)} pairs"]
    if abs(middle) > window / 2:
        lines.append(f"- WARNING: the median offset is over half the window ({window} frames): "
                     "a cup never seen may have taken the next cup's record")
    return lines


def position_lines(score: CupScore, window: int) -> list[str]:
    """One camera position's numbers."""
    empty, full = score.empty, score.full
    gap_lines = [] if score.gap.total == 0 else [
        _rate("gaps given a record", score.gap.total - score.gap.unseen, score.gap.total,
              good=False)]
    return [
        _rate("empty cups caught", empty.defect, empty.total, good=True),
        _rate("false БРАК on full cups", full.defect, full.total, good=False),
        _rate("empty cups never seen", empty.unseen, empty.total, good=False),
        _rate("full cups never seen", full.unseen, full.total, good=False),
        _rate("empty cups seen but not judged", empty.no_verdict, empty.total, good=False),
        _rate("full cups seen but not judged", full.no_verdict, full.total, good=False),
        _rate("false БРАК on white cups", score.white.defect, score.white.total, good=False),
        _rate("false БРАК on pink cups", score.pink.defect, score.pink.total, good=False),
        *gap_lines,
        f"- records paired with no cup: {score.unpaired}",
        *_offset_lines(score.offsets, window),
    ]


def hand_line(labelled: int, verdicts: Sequence[Label]) -> str:
    """How many frames were labelled as showing a hand, and the verdicts of those judged."""
    return (f"- hand frames labelled: {labelled}; judged: {len(verdicts)} "
            f"(БРАК {verdicts.count(Label.DEFECT)}, ГОДЕН {verdicts.count(Label.PASS)}, "
            f"НЕТ РЕШЕНИЯ {verdicts.count(Label.NO_VERDICT)})")


def header_lines(config_hash: str, window: int) -> list[str]:
    """The stamp, what the camera cannot see, and what the numbers were made with."""
    return [
        "# INDICATIVE — line footage scored against hand-counted cups",
        "",
        "These numbers are **INDICATIVE**: one visit's footage, counted by hand; not Gate A and "
        "not a client claim (ADR-0007).",
        "The camera sees the top of each cup only, from one position: nothing under the моти, "
        "inside it or beside the cup is judged, and a cup it never saw is not caught.",
        "",
        f"- config_hash: `{config_hash}`",
        f"- a record is paired with a cup at most {window} frames from it",
    ]
