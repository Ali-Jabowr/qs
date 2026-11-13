"""The comparison table: literal tallies in, the table and its bounds out (#160).

Bounds are written as 4-decimal fractions; they were worked out with scipy's beta quantiles,
not with `qs.eval.stats`.
"""

from __future__ import annotations

import pytest

from qs.eval.line_score import CupScore, Tally
from qs.lane.table import MethodResult, render_table


def _score(empty: Tally = Tally(), full: Tally = Tally(), gap: Tally = Tally()) -> CupScore:
    return CupScore(empty=empty, full=full, white=Tally(), pink=Tally(), unpaired=0,
                    offsets=(), gap=gap)


def _table(score: CupScore, name: str = "C", gap_scored: bool = True) -> str:
    return render_table([MethodResult(name, score, gap_scored)])


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        # empty: 150 passes, 148 БРАК, 1 ГОДЕН, 1 no verdict
        (_score(empty=Tally(150, 148, 1, 0, 1)), "| 148 of 150 | 0.9586 |"),
        # full: 1 000, 4 БРАК, 990 ГОДЕН, 4 НЕТ РЕШЕНИЯ, 2 no verdict
        (_score(full=Tally(1000, 4, 990, 4, 2)), "| 4 of 1000 | 0.0091 |"),
        (_score(full=Tally(1000, 4, 990, 4, 2)), "| 6 of 1000 | 0.0118 |"),
        # gaps: 100, no verdict on any
        (_score(gap=Tally(100, 0, 0, 0, 100)), "| 0 of 100 | 0.0295 |"),
        # a gap that got a verdict counts, whichever verdict it was
        (_score(gap=Tally(2, 0, 1, 0, 1)), "| 1 of 2 | 0.9747 |"),
        # empty 3 passes, 1 caught
        (_score(empty=Tally(3, 1, 1, 0, 1)), "| 1 of 3 | 0.0170 |"),
    ],
)
def test_cells_show_counts_and_bounds(score: CupScore, expected: str) -> None:
    assert expected in _table(score)


@pytest.mark.parametrize(
    ("score", "expected", "never"),
    [
        (_score(full=Tally(10, 0, 10, 0, 0), gap=Tally(5, 0, 0, 0, 5)), "no empty pass",
         "0 of 0"),
        (_score(empty=Tally(10, 10, 0, 0, 0), gap=Tally(5, 0, 0, 0, 5)), "no full pass",
         "0 of 0"),
        (_score(empty=Tally(10, 10, 0, 0, 0), full=Tally(10, 0, 10, 0, 0)), "no gap",
         "0 of 0"),
    ],
)
def test_a_kind_with_no_pass_reads_as_such_never_as_a_rate(
    score: CupScore, expected: str, never: str
) -> None:
    table = _table(score)
    assert expected in table
    assert never not in table


@pytest.mark.parametrize(
    ("score", "row"),
    [
        (_score(full=Tally(1000, 4, 990, 4, 2)), "| full | 990 | 4 | 4 | 2 |"),
        (_score(empty=Tally(150, 148, 1, 0, 1)), "| empty | 1 | 148 | 0 | 1 |"),
        (_score(gap=Tally(100, 0, 0, 0, 100)), "| gap | 0 | 0 | 0 | 100 |"),
    ],
)
def test_the_three_by_four_table_has_columns_pass_defect_no_decision_no_verdict(
    score: CupScore, row: str
) -> None:
    table = _table(score)
    assert "| | ГОДЕН | БРАК | НЕТ РЕШЕНИЯ | no verdict |" in table
    assert row in table


def test_a_method_not_run_on_gaps_says_so() -> None:
    table = _table(_score(empty=Tally(10, 10, 0, 0, 0)), name="B PatchCore (judge only)",
                   gap_scored=False)
    assert "B PatchCore (judge only)" in table
    assert "gaps not scored" in table


@pytest.mark.parametrize(
    ("run", "first_line"),
    [
        (None, "| method | empty caught | lower bound | full flagged | upper bound | "
               "full no answer | upper bound | gaps with a verdict | upper bound |"),
        (2, "RESCORED (run 2)"),
        (3, "RESCORED (run 3)"),
    ],
)
def test_a_rescoring_is_marked_on_the_first_line(run: int | None, first_line: str) -> None:
    table = render_table([MethodResult("C", _score(empty=Tally(1, 1, 0, 0, 0)))], run)
    assert table.splitlines()[0] == first_line


def test_one_row_per_method_in_the_order_given() -> None:
    table = render_table([
        MethodResult("A", _score(empty=Tally(1, 1, 0, 0, 0))),
        MethodResult("C", _score(empty=Tally(1, 0, 1, 0, 0))),
    ])
    assert table.index("| A |") < table.index("| C |")
