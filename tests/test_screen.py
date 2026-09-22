from pathlib import Path

import numpy as np
import pytest

from qs.config import load_config
from qs.screen import KeyPress, Screen, compose

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "DejaVuSans.ttf"
EXAMPLE = ROOT / "config" / "station-A-bench.example.yml"


def test_compose_returns_the_requested_size() -> None:
    assert compose(np.zeros((48, 64, 3), np.uint8), [], FONT, (320, 200)).size == (320, 200)
    assert compose(None, [], FONT).size == (1280, 800)


def test_a_white_image_is_white_at_the_centre_of_the_image_area() -> None:
    canvas = compose(np.full((50, 100, 3), 255, np.uint8), [], FONT)
    assert canvas.getpixel((640, 360)) == (255, 255, 255)
    assert canvas.getpixel((640, 790)) == (0, 0, 0)


def test_the_image_is_converted_from_bgr() -> None:
    blue_bgr = np.zeros((50, 100, 3), np.uint8)
    blue_bgr[:, :, 0] = 255
    assert compose(blue_bgr, [], FONT).getpixel((640, 360)) == (0, 0, 255)


def test_no_image_still_renders_its_placeholder() -> None:
    canvas = np.asarray(compose(None, [], FONT))
    assert canvas[:720].max() > 0
    assert canvas[720:].max() == 0


def test_lines_are_drawn_in_the_text_panel() -> None:
    blank = np.asarray(compose(np.full((50, 100, 3), 255, np.uint8), [], FONT))
    texted = np.asarray(compose(np.full((50, 100, 3), 255, np.uint8), ["one", "two"], FONT))
    assert blank[720:].max() == 0
    assert texted[720:].max() > 0


def test_only_the_first_three_lines_are_drawn() -> None:
    three = np.asarray(compose(None, ["a", "b", "c"], FONT))
    four = np.asarray(compose(None, ["a", "b", "c", "d"], FONT))
    assert np.array_equal(three, four)


@pytest.mark.gui
def test_a_real_screen_shows_frames_and_queues_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    screen = Screen(load_config(EXAMPLE).config.display, "QS test")
    try:
        screen.tick(np.full((48, 64, 3), 200, np.uint8), ["hello"])
        assert screen._root is not None
        screen._root.event_generate("<KeyPress>", keysym="space")
        screen.tick(None, [])
        keys = screen.keys()
        assert len(keys) == 1
        assert isinstance(keys[0], KeyPress)
        assert keys[0].keysym == "space"
        assert screen.keys() == []
        assert screen.closed is False
    finally:
        screen.close()
        screen.close()
    assert screen.closed is True
    screen.tick(None, [])  # after close: does nothing
