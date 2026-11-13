"""The pre-registered rule: frame probabilities to runs, verdicts and pairs with labelled passes."""

from __future__ import annotations

import numpy as np
import pytest

from qs.contracts import Label
from qs.labels import Cup, CupState
from qs.lane.passes import Run, pair_runs, runs_from_probs, verdict

F = (0.9, 0.05, 0.05)
E = (0.05, 0.9, 0.05)
N = (0.05, 0.05, 0.9)
EDGE = (0.25, 0.25, 0.5)  # P(none) exactly 0.5: not a cup frame
JUST_CUP = (0.26, 0.25, 0.49)
LOW = (0.5, 0.25, 0.25)  # P(empty) 0.25, exact in binary so the mean below is exact
HIGH = (0.0, 0.75, 0.25)


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        ([], []),
        ([N, N, N], []),
        ([N, F, F, N, N], [Run(1, 2, 2, None)]),  # a run of 2 has no score
        ([N, E, E, E, N], [Run(1, 3, 3, 0.9)]),
        ([N, F, F, F, N], [Run(1, 3, 3, 0.05)]),
        ([F, F, N, F, F, N, N], [Run(0, 4, 4, 0.05)]),  # one stray frame is bridged
        ([F, F, F, N, N, E, E, E], [Run(0, 2, 3, 0.05), Run(5, 7, 3, 0.9)]),  # two split
        ([F, E, E], [Run(0, 2, 3, 0.9)]),  # the median, not the mean
        ([F, F, E], [Run(0, 2, 3, 0.05)]),
        ([F, F, N, N, F], [Run(0, 1, 2, None), Run(4, 4, 1, None)]),
        ([F, N, F], [Run(0, 2, 2, None)]),  # bridged, but only 2 cup frames
        ([F, F, N, F, F, N, N, N, F, F, F], [Run(0, 4, 4, 0.05), Run(8, 10, 3, 0.05)]),
        ([N, F, F, F, N, N, N, N], [Run(1, 3, 3, 0.05)]),  # trailing gap is not in the run
        ([EDGE, EDGE, EDGE], []),
        ([JUST_CUP, JUST_CUP, JUST_CUP], [Run(0, 2, 3, 0.25)]),
        ([LOW, HIGH, LOW, HIGH], [Run(0, 3, 4, 0.5)]),  # even count: mean of the middle two
    ],
)
def test_runs_from_probs(probs, expected):
    assert runs_from_probs(probs) == expected


def test_a_cup_standing_at_a_stop_is_one_run():
    assert runs_from_probs([F] * 1800) == [Run(0, 1799, 1800, 0.05)]


def test_runs_from_probs_accepts_a_numpy_array():
    assert runs_from_probs(np.array([N, E, E, E, N])) == [Run(1, 3, 3, 0.9)]


@pytest.mark.parametrize(
    ("probs", "kwargs", "expected"),
    [
        ([N, F, F, N], {"min_frames": 2}, [Run(1, 2, 2, 0.05)]),
        ([N, F, F, F, N], {"min_frames": 4}, [Run(1, 3, 3, None)]),
        ([F, F, N, F, F], {"max_gap": 0}, [Run(0, 1, 2, None), Run(3, 4, 2, None)]),
        ([F, F, N, N, F, F], {"max_gap": 2}, [Run(0, 5, 4, 0.05)]),
    ],
)
def test_runs_from_probs_parameters(probs, kwargs, expected):
    assert runs_from_probs(probs, **kwargs) == expected


@pytest.mark.parametrize("row", [(0.5, 0.5), (0.2, 0.2, 0.2, 0.4), ()])
def test_a_row_without_three_probabilities_is_an_error(row):
    with pytest.raises(ValueError, match="frame 1"):
        runs_from_probs([F, row])


@pytest.mark.parametrize(
    ("run", "threshold", "expected"),
    [
        (Run(1, 3, 3, 0.9), 0.5, Label.DEFECT),
        (Run(1, 3, 3, 0.5), 0.5, Label.PASS),  # above, not at
        (Run(1, 3, 3, 0.51), 0.5, Label.DEFECT),
        (Run(1, 3, 3, 0.05), 0.5, Label.PASS),
        (Run(1, 3, 3, 0.0), 0.0, Label.PASS),
        (Run(1, 2, 2, None), 0.5, Label.NO_VERDICT),
        (Run(1, 2, 2, None), 0.0, Label.NO_VERDICT),
    ],
)
def test_verdict(run, threshold, expected):
    assert verdict(run, threshold) == expected


def cup(order: int, frame: int) -> Cup:
    return Cup(video="a.mp4", order=order, state=CupState.FULL, frame=frame)


def run(first: int, last: int, score: float = 0.05) -> Run:
    return Run(first, last, last - first + 1, score)


@pytest.mark.parametrize(
    ("cups", "runs", "window", "expected_pairs", "expected_unpaired"),
    [
        (
            [cup(1, 100), cup(2, 900)],
            [run(90, 110), run(200, 1700, 0.9)],
            15,
            [(cup(1, 100), run(90, 110)), (cup(2, 900), run(200, 1700, 0.9))],
            0,
        ),
        (
            [cup(1, 100), cup(2, 500)],
            [run(90, 110), run(300, 320, 0.9)],
            15,
            [(cup(1, 100), run(90, 110)), (cup(2, 500), None)],
            1,
        ),
        (
            [cup(1, 100), cup(2, 104)],
            [run(90, 110)],
            15,
            [(cup(1, 100), run(90, 110)), (cup(2, 104), None)],
            0,
        ),
        # the window is inclusive on both sides
        ([cup(1, 75)], [run(90, 110)], 15, [(cup(1, 75), run(90, 110))], 0),
        ([cup(1, 125)], [run(90, 110)], 15, [(cup(1, 125), run(90, 110))], 0),
        ([cup(1, 74)], [run(90, 110)], 15, [(cup(1, 74), None)], 0),
        ([cup(1, 126)], [run(90, 110)], 15, [(cup(1, 126), None)], 0),
        # cups given out of frame order are paired and returned in frame order
        (
            [cup(2, 900), cup(1, 100)],
            [run(90, 110), run(890, 910, 0.9)],
            15,
            [(cup(1, 100), run(90, 110)), (cup(2, 900), run(890, 910, 0.9))],
            0,
        ),
        # an unpaired run that starts before the first pass or after the last is not counted
        (
            [cup(1, 100), cup(2, 200)],
            [run(10, 20), run(95, 105), run(195, 205), run(900, 910)],
            15,
            [(cup(1, 100), run(95, 105)), (cup(2, 200), run(195, 205))],
            0,
        ),
        # one starting exactly on the first or last pass frame is counted
        (
            [cup(1, 100), cup(2, 300)],
            [run(100, 102), run(200, 202), run(300, 302)],
            0,
            [(cup(1, 100), run(100, 102)), (cup(2, 300), run(300, 302))],
            1,
        ),
        # one run covering two passes serves the first only
        (
            [cup(1, 100), cup(2, 200)],
            [run(50, 250)],
            15,
            [(cup(1, 100), run(50, 250)), (cup(2, 200), None)],
            0,
        ),
        # a run within reach of two passes goes to the earlier; the later takes the next run
        (
            [cup(1, 100), cup(2, 110)],
            [run(95, 105), run(108, 115, 0.9)],
            15,
            [(cup(1, 100), run(95, 105)), (cup(2, 110), run(108, 115, 0.9))],
            0,
        ),
        ([cup(1, 100)], [], 15, [(cup(1, 100), None)], 0),
        # no passes: nothing to pair and no span to count unpaired runs in
        ([], [run(90, 110)], 15, [], 0),
        ([], [], 15, [], 0),
        (
            [cup(1, 100), cup(2, 900)],
            [run(90, 110), run(300, 320, 0.9), run(500, 520, 0.9), run(890, 910)],
            15,
            [(cup(1, 100), run(90, 110)), (cup(2, 900), run(890, 910))],
            2,
        ),
    ],
)
def test_pair_runs(cups, runs, window, expected_pairs, expected_unpaired):
    assert pair_runs(cups, runs, window) == (expected_pairs, expected_unpaired)
