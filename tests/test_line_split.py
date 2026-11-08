"""The time split of a camera position's good line frames (#108).

Worked by hand for `a.avi`, frames 0, 10, …, 90 and 65, at 10 fps with a 0.5 s gap: the labelled
span is 0–90, 91 frames, so training ends before 0 + ⌊0.70 × 91⌋ = 63 and calibration before
0 + ⌊0.85 × 91⌋ = 77; the gap is 5 frames, so calibration starts at 68 and test at 82. Frame 65
falls in the first gap and is dropped. `b.avi` has frames 5 and 6: its span is 2 frames, training
ends before 6, and the held-out blocks start past its last frame.
"""

from __future__ import annotations

import pytest

from qs.eval.line_split import LineSplit, LineSplitError, split_frames

A = ["a.avi#0", "a.avi#10", "a.avi#20", "a.avi#30", "a.avi#40", "a.avi#50", "a.avi#60",
     "a.avi#65", "a.avi#70", "a.avi#90"]


def test_each_video_is_split_by_time_with_a_gap_before_each_held_out_block() -> None:
    assert split_frames([*A, "b.avi#6", "b.avi#5"], fps=10.0, gap_s=0.5) == LineSplit(
        train=("a.avi#0", "a.avi#10", "a.avi#20", "a.avi#30", "a.avi#40", "a.avi#50",
               "a.avi#60", "b.avi#5"),
        calibrate=("a.avi#70",),
        test=("a.avi#90",),
        dropped=("a.avi#65", "b.avi#6"),
    )


def test_a_frame_index_orders_by_number_not_by_text() -> None:
    split = split_frames(["c.avi#100", "c.avi#9", "c.avi#20"], fps=10.0, gap_s=0.0)
    assert split.train == ("c.avi#9", "c.avi#20")


@pytest.mark.parametrize("frame_id", ["photo_001.png", "a.avi#", "a.avi#x", "#3"])
def test_a_frame_id_that_is_not_a_video_frame_is_refused(frame_id: str) -> None:
    with pytest.raises(LineSplitError, match="not a video frame"):
        split_frames([frame_id], fps=10.0, gap_s=0.5)
