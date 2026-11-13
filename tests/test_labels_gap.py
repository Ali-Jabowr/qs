"""A pass can be a gap (#153): a chain position with no cup, labelled by `g`, never counted as a
full cup by the scoring, and the lane window drawn on the frame shown while labelling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.cmd.label import parse_window
from qs.contracts import Label
from qs.eval.line_score import CupScore, Judged, Tally, score_cups
from qs.labels import Cup, CupSession, VideoFrames, load_labels

WINDOW = 15


def draw_video(path: Path, count: int) -> Path:
    """A black video, `count` frames of 320 x 120."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (320, 120))
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.zeros((120, 320, 3), np.uint8))
    writer.release()
    return path


def cup_session(root: Path, window: tuple[int, int, int, int] | None = None) -> CupSession:
    path = draw_video(root / "a.avi", 60)
    return CupSession(load_labels(root / "labels.json", "a"), VideoFrames(path), window)


def test_key_g_records_a_gap_that_is_on_disk_and_loads_back_equal(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    for _ in range(40):
        session.press("Right")
    session.press("g")

    document = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert document["cups"] == [{"video": "a.avi", "order": 1, "state": "gap", "frame": 40}]
    assert load_labels(tmp_path / "labels.json", "a").cups == (Cup("a.avi", 1, "gap", 40),)


def test_a_labels_file_written_before_gaps_existed_loads_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({
            "version": 1,
            "source": "a",
            "frames": {},
            "cups": [
                {"video": "a.avi", "order": 1, "state": "full", "frame": 10},
                {"video": "a.avi", "order": 2, "state": "empty", "frame": 42},
            ],
        }),
        encoding="utf-8",
    )

    assert load_labels(path, "a").cups == (
        Cup("a.avi", 1, "full", 10), Cup("a.avi", 2, "empty", 42))


def test_the_counts_on_screen_name_the_gaps(tmp_path: Path) -> None:
    session = cup_session(tmp_path)
    session.press("f")
    session.press("Right")
    session.press("g")

    assert session.summary().startswith("2 cups in a.avi (1 full, 0 empty, 1 gap)")


def test_a_gap_is_scored_on_its_own_and_not_as_a_full_cup() -> None:
    cups = [Cup("a.mkv", 1, "full", 100), Cup("a.mkv", 2, "gap", 132),
            Cup("a.mkv", 3, "empty", 164)]
    judged = [Judged(100, Label.PASS), Judged(132, Label.DEFECT), Judged(164, Label.DEFECT)]

    score = score_cups(cups, judged, WINDOW)

    assert score.full.total == 1
    assert score.empty.total == 1
    assert score.empty.defect == 1
    assert score.gap.total == 1
    assert score.gap.defect == 1


def test_a_gap_with_no_record_is_unseen() -> None:
    cups = [Cup("a.mkv", 1, "full", 100), Cup("a.mkv", 2, "gap", 132),
            Cup("a.mkv", 3, "empty", 164)]
    judged = [Judged(100, Label.PASS), Judged(164, Label.DEFECT)]

    assert score_cups(cups, judged, WINDOW).gap.unseen == 1


def test_a_score_built_with_the_six_old_fields_has_an_all_zero_gap() -> None:
    tally = Tally(total=1)

    score = CupScore(empty=tally, full=tally, white=tally, pink=tally, unpaired=0, offsets=())

    assert score.gap == Tally(total=0, defect=0, passed=0, no_verdict=0, unseen=0)


def test_window_is_parsed_to_four_numbers() -> None:
    assert parse_window("100,60,80,90") == (100, 60, 80, 90)


def test_window_with_three_numbers_is_refused_in_one_line() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="X,Y,W,H") as refusal:
        parse_window("100,60,80")
    assert "\n" not in str(refusal.value)


def test_the_window_is_drawn_on_a_copy_and_the_video_frame_is_unchanged(tmp_path: Path) -> None:
    session = cup_session(tmp_path, window=(100, 60, 80, 40))
    original = VideoFrames(tmp_path / "a.avi").read(0)

    shown = session.image()

    assert shown.shape == original.shape
    assert shown[60, 140].any()  # on the top edge of the rectangle
    assert not shown[80, 140].any()  # inside it
    assert not shown[5, 5].any()  # outside it
    assert np.array_equal(session.frames.read(0), original)
    assert not original.any()


def test_without_a_window_the_frame_is_shown_as_read(tmp_path: Path) -> None:
    session = cup_session(tmp_path)

    assert not session.image().any()
