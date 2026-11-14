"""The comparison table of `qs compare`: tallies in, Markdown text out (#160).

One row per method with the counts and their one-sided 95% bounds (`qs.eval.stats`): the lower
bound on a share wanted (empty cups caught), the upper bound on one not wanted. Under it, per
method, the 3 x 4 counts of what each kind of labelled pass was judged. A kind with no pass
reads as such, never as a rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qs.eval.line_score import CupScore, Tally
from qs.eval.stats import lower_bound, upper_bound

_COLUMNS = ("method", "empty caught", "lower bound", "full flagged", "upper bound",
            "full no answer", "upper bound", "gaps with a verdict", "upper bound")


@dataclass(frozen=True, slots=True)
class MethodResult:
    """One method's counts. `gap_scored` is False for a method that is not run on gaps."""

    name: str
    score: CupScore
    gap_scored: bool = True


def _cells(n: int, k: int, none: str, *, good: bool) -> list[str]:
    """`k of n` and its bound; `none` and a dash when there is no pass to count."""
    if n == 0:
        return [none, "-"]
    bound = lower_bound(n, k) if good else upper_bound(n, k)
    return [f"{k} of {n}", f"{bound:.4f}"]


def _row(result: MethodResult) -> str:
    empty, full, gap = result.score.empty, result.score.full, result.score.gap
    cells = [result.name]
    cells += _cells(empty.total, empty.defect, "no empty pass", good=True)
    cells += _cells(full.total, full.defect, "no full pass", good=False)
    cells += _cells(full.total, full.no_verdict + full.unseen, "no full pass", good=False)
    if result.gap_scored:
        cells += _cells(gap.total, gap.total - gap.unseen, "no gap", good=False)
    else:
        cells += ["gaps not scored", "-"]
    return _line(cells)


def _line(cells: Sequence[str]) -> str:
    return ("| " + " | ".join(cells) + " |").replace("|  |", "| |")


def _counts(label: str, tally: Tally | None) -> str:
    if tally is None:
        return _line([label, "-", "-", "-", "-"])
    return _line([label, str(tally.passed), str(tally.defect), str(tally.no_verdict),
                  str(tally.unseen)])


def _confusion(result: MethodResult) -> list[str]:
    score = result.score
    return [
        f"### {result.name}: what each labelled pass was judged",
        "",
        _line(["", "ГОДЕН", "БРАК", "НЕТ РЕШЕНИЯ", "no verdict"]),
        _line(["---"] * 5),
        _counts("full", score.full),
        _counts("empty", score.empty),
        _counts("gap", score.gap if result.gap_scored else None),
    ]


def render_table(results: Sequence[MethodResult], rescored_run: int | None = None) -> str:
    """The table text. With `rescored_run`, a first line `RESCORED (run N)` stands before it."""
    lines = [f"RESCORED (run {rescored_run})", ""] if rescored_run is not None else []
    lines += [_line(_COLUMNS), _line(["---"] * len(_COLUMNS))]
    lines += [_row(result) for result in results]
    for result in results:
        lines += ["", *_confusion(result)]
    return "\n".join(lines)
