"""How far the station's records sit from the frames the cups were labelled at (#105). The window
pairs a record with a cup only if it is near; when records sit well off the labelled frames as a
rule, a never-seen cup can take the next cup's record, so the median offset is reported and a
large one warned about. Every expected value is worked by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.contracts import Label
from qs.eval.line_score import Judged, score_cups
from qs.labels import Cup
from test_score_line import CUPS, _labels, _run


def test_each_pair_gives_its_record_frame_minus_its_label_frame() -> None:
    cups = [Cup("line.mp4", 1, "full", 10), Cup("line.mp4", 2, "empty", 50),
            Cup("line.mp4", 3, "full", 90)]
    judged = [Judged(13, Label.PASS), Judged(48, Label.DEFECT)]

    score = score_cups(cups, judged, 15)

    assert score.offsets == (3, -2)


def test_the_median_offset_is_printed_per_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Records at 6, 19, 40 and 45 against labels at 7, 18, 39 and 46: offsets -1, 1, 1, -1.
    labels = _labels(tmp_path, "top.json", CUPS)

    _run(tmp_path, f"top={labels}")

    out = capsys.readouterr().out
    assert "- record frame minus labelled frame: median 0.0 over 4 pairs" in out
    assert "WARNING" not in out


def test_a_median_offset_over_half_the_window_is_warned_about(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Records at 6 and 19 against labels at 9 and 22: offsets -3 and -3, over half of 5.
    labels = _labels(tmp_path, "top.json", [("empty", 9), ("full_white", 22)])

    _run(tmp_path, f"top={labels}", window=5)

    out = capsys.readouterr().out
    assert "- record frame minus labelled frame: median -3.0 over 2 pairs" in out
    assert ("- WARNING: the median offset is over half the window (5 frames): a cup never "
            "seen may have taken the next cup's record") in out
