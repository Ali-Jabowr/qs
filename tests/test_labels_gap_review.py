"""Gaps and the lane window, second round (#153 review): combined scores, the printed gap line,
the frame the session keeps, the window's bounds, and `main` passing `--window` on."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.cmd.label import main, parse_window
from qs.eval.line_score import CupScore, Tally, combined, position_lines
from qs.labels import CupSession, VideoFrames, _Session, load_labels
from qs.screen import KeyPress

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "station-A-bench.example.yml"
NONE = Tally()


def score(gap: Tally | None = None) -> CupScore:
    base = {
        "empty": Tally(total=2, defect=2), "full": Tally(total=3, passed=3),
        "white": NONE, "pink": NONE, "unpaired": 0, "offsets": (1, 3),
    }
    return CupScore(**base) if gap is None else CupScore(**base, gap=gap)  # type: ignore[arg-type]


def draw_video(path: Path, count: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (320, 120))
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.zeros((120, 320, 3), np.uint8))
    writer.release()
    return path


def test_combined_sums_the_gaps_of_every_video() -> None:
    first = score(Tally(total=2, defect=1, unseen=1))
    second = score(Tally(total=3, unseen=3))

    assert combined([first, second]).gap == Tally(
        total=5, defect=1, passed=0, no_verdict=0, unseen=4)


def test_the_gap_line_is_printed_when_a_gap_is_labelled() -> None:
    lines = position_lines(score(Tally(total=4, defect=1, unseen=3)), 15)

    assert lines == [
        "- empty cups caught: 2 of 2 = 100.0%, lower bound 22.4%",
        "- false БРАК on full cups: 0 of 3 = 0.0%, upper bound 63.2%",
        "- empty cups never seen: 0 of 2 = 0.0%, upper bound 77.6%",
        "- full cups never seen: 0 of 3 = 0.0%, upper bound 63.2%",
        "- empty cups seen but not judged: 0 of 2 = 0.0%, upper bound 77.6%",
        "- full cups seen but not judged: 0 of 3 = 0.0%, upper bound 63.2%",
        "- false БРАК on white cups: none labelled",
        "- false БРАК on pink cups: none labelled",
        "- gaps given a record: 1 of 4 = 25.0%, upper bound 75.1%",
        "- records paired with no cup: 0",
        "- record frame minus labelled frame: median 2.0 over 2 pairs",
    ]


def test_without_a_gap_the_lines_are_the_ones_printed_before_gaps_existed() -> None:
    assert position_lines(score(), 15) == [
        "- empty cups caught: 2 of 2 = 100.0%, lower bound 22.4%",
        "- false БРАК on full cups: 0 of 3 = 0.0%, upper bound 63.2%",
        "- empty cups never seen: 0 of 2 = 0.0%, upper bound 77.6%",
        "- full cups never seen: 0 of 3 = 0.0%, upper bound 63.2%",
        "- empty cups seen but not judged: 0 of 2 = 0.0%, upper bound 77.6%",
        "- full cups seen but not judged: 0 of 3 = 0.0%, upper bound 63.2%",
        "- false БРАК on white cups: none labelled",
        "- false БРАК on pink cups: none labelled",
        "- records paired with no cup: 0",
        "- record frame minus labelled frame: median 2.0 over 2 pairs",
    ]


def test_drawing_the_window_leaves_the_frame_the_session_keeps_black(tmp_path: Path) -> None:
    video = VideoFrames(draw_video(tmp_path / "a.avi", 3))
    session = CupSession(load_labels(tmp_path / "labels.json", "a"), video, (100, 60, 80, 40))

    shown = session.image()

    assert shown.any()
    assert not _Session.image(session).any()


@pytest.mark.parametrize(
    "text",
    ["100,60,0,90", "100,60,80,0", "100,60,-80,90", "100,60,80,-1", "-1,60,80,90", "100,-5,80,90"],
)
def test_a_window_with_no_size_or_a_corner_off_the_picture_is_refused_in_one_line(
    text: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="X,Y,W,H") as refusal:
        parse_window(text)
    assert "\n" not in str(refusal.value)


class FakeScreen:
    """Records the image of every tick; plays no keys, then closes."""

    def __init__(self) -> None:
        self.images: list[np.ndarray | None] = []
        self.closed = False
        self._script: deque[list[KeyPress]] = deque()

    def tick(self, image: np.ndarray | None, lines: list[str]) -> None:
        self.images.append(image)

    def keys(self) -> list[KeyPress]:
        self.closed = True
        return []

    def close(self) -> None:
        self.closed = True


def run_label(source: Path, mode: str, window: tuple[int, int, int, int] | None,
              screen: FakeScreen, root: Path) -> int:
    values = argparse.Namespace(
        config=str(EXAMPLE), mode=mode, source=str(source), every=None,
        labels=str(root / "labels.json"), window=window,
    )
    return main(values, make_screen=lambda display, title: screen)


def test_main_shows_the_window_on_the_frame_in_cups_mode(tmp_path: Path) -> None:
    screen = FakeScreen()

    code = run_label(draw_video(tmp_path / "a.avi", 3), "cups", (100, 60, 80, 40), screen, tmp_path)

    assert code == 0
    shown = screen.images[0]
    assert shown is not None
    assert shown[60, 140].any()
    assert not shown[5, 5].any()


def test_main_refuses_a_window_in_frames_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_label(draw_video(tmp_path / "a.avi", 3), "frames", (1, 2, 3, 4),
                     FakeScreen(), tmp_path)

    assert code == 2
    assert capsys.readouterr().out == "window: --window only applies to --mode cups\n"
