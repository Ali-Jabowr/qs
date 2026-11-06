"""Scoring line footage against hand-counted cups (#105): the pairing of labelled cups with the
station's records, and the counts read from it. Every expected count is worked by hand from the
frames written in the test."""

from __future__ import annotations

from qs.contracts import Label
from qs.eval.line_score import Judged, Tally, score_cups
from qs.labels import Cup

WINDOW = 15


def cup(order: int, state: str, frame: int) -> Cup:
    return Cup("line.mp4", order, state, frame)


def test_a_cup_never_seen_does_not_shift_the_cups_after_it() -> None:
    # Cup 1 never gives a record. Paired by order, cup 1 would take cup 2's ГОДЕН and cup 2
    # cup 3's БРАК: an empty cup passed and a false БРАК, where there is neither.
    cups = [cup(1, "empty", 10), cup(2, "full", 50), cup(3, "empty", 90)]
    judged = [Judged(52, Label.PASS), Judged(91, Label.DEFECT)]

    score = score_cups(cups, judged, WINDOW)

    assert score.empty == Tally(total=2, defect=1, passed=0, no_verdict=0, unseen=1)
    assert score.full == Tally(total=1, defect=0, passed=1, no_verdict=0, unseen=0)


def test_a_record_paired_with_no_cup_is_counted_only_inside_the_labelled_span() -> None:
    # The span is frames 100–200, widened by the window: 85–215. The record at 150 is no cup's;
    # those at 40 and 260 are outside the span, where cups were never counted.
    cups = [cup(1, "full", 100), cup(2, "full", 200)]
    judged = [
        Judged(40, Label.PASS),
        Judged(101, Label.PASS),
        Judged(150, Label.DEFECT),
        Judged(199, Label.PASS),
        Judged(260, Label.PASS),
    ]

    score = score_cups(cups, judged, WINDOW)

    assert score.unpaired == 1
    assert score.full == Tally(total=2, defect=0, passed=2, no_verdict=0, unseen=0)


def test_white_and_pink_cups_are_also_counted_apart() -> None:
    # A full cup with no flavour counts only among the full cups.
    cups = [cup(1, "full_white", 10), cup(2, "full_pink", 50), cup(3, "full", 90),
            cup(4, "full_pink", 130)]
    judged = [Judged(11, Label.DEFECT), Judged(51, Label.PASS), Judged(91, Label.NO_VERDICT)]

    score = score_cups(cups, judged, WINDOW)

    assert score.white == Tally(total=1, defect=1, passed=0, no_verdict=0, unseen=0)
    assert score.pink == Tally(total=2, defect=0, passed=1, no_verdict=0, unseen=1)
    assert score.full == Tally(total=4, defect=1, passed=1, no_verdict=1, unseen=1)
