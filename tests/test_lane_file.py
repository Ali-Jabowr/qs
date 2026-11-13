"""The lane file and the crop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qs.errors import ConfigError
from qs.lane.lanefile import Lane, crop, load_lane

GOOD = "version: 1\nwindow: [500, 240, 276, 230]\nfps: 30.0\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lane.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_reads_window_and_fps(tmp_path: Path) -> None:
    assert load_lane(write(tmp_path, GOOD)) == Lane((500, 240, 276, 230), 30.0)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("version: 1\nfps: 30.0\n", "window"),
        ("version: 1\nwindow: [500, 240, 276, 230]\n", "fps"),
        ("window: [500, 240, 276, 230]\nfps: 30.0\n", "version"),
        (GOOD + "extra: 1\n", "extra"),
        ("version: 1\nwindow: [500, 240, 0, 230]\nfps: 30.0\n", "window"),
        ("version: 1\nwindow: [500, 240, 276, 230]\nfps: .inf\n", "fps"),
        ("version: 1\nwindow: [500, 240, 276, 230]\nfps: .nan\n", "fps"),
    ],
)
def test_load_refuses_a_bad_file_naming_the_key(tmp_path: Path, text: str, key: str) -> None:
    with pytest.raises(ConfigError, match=key):
        load_lane(write(tmp_path, text))


def frame() -> np.ndarray:
    return np.arange(720 * 1280 * 3, dtype=np.uint32).reshape(720, 1280, 3).astype(np.uint8)


def test_crop_is_that_rectangle() -> None:
    pixels = frame()
    out = crop(pixels, Lane((500, 240, 276, 230), 30.0))
    assert out.shape == (230, 276, 3)
    assert np.array_equal(out, pixels[240:470, 500:776])


def test_crop_is_a_copy() -> None:
    pixels = frame()
    before = pixels.copy()
    out = crop(pixels, Lane((500, 240, 276, 230), 30.0))
    out[:] = 0
    assert np.array_equal(pixels, before)


def test_crop_outside_the_frame_names_window_and_frame_size() -> None:
    with pytest.raises(ConfigError, match=r"\(1200, 240, 276, 230\).*1280x720"):
        crop(frame(), Lane((1200, 240, 276, 230), 30.0))
