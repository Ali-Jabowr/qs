"""The line's warmth keys (#132, ADR-0029 decision 4), required of every line station since
`blob_share_min` went (#135)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config

RIM_140: dict[str, Any] = {"rim_size_px": [50, 140]}  # the zone rule takes the major axis, 140


def test_the_warmth_keys_are_read(tmp_path: Path) -> None:
    line = load_config(warmth_config(tmp_path)).config.inspector.line

    assert line is not None
    assert (line.warmth_min, line.burst, line.warmth_spread_max, line.max_speed_px_s) == (
        5.0, 3, 40.0, 320.0,
    )
    assert line.burst_min_frames == 3


def test_the_minimum_frames_can_be_set(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, burst=4, burst_min_frames=3, zone=[20, 50, 300, 180])

    line = load_config(path).config.inspector.line

    assert line is not None
    assert (line.burst, line.burst_min_frames) == (4, 3)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"burst": None}, "inspector.line.burst"),
        ({"warmth_spread_max": None}, "inspector.line.warmth_spread_max"),
        ({"max_speed_px_s": None}, "inspector.line.max_speed_px_s"),
        ({"direction": None}, "inspector.line.direction"),
        ({"warmth_min": None}, "inspector.line.warmth_min"),
        ({"burst": 2}, "inspector.line.burst_min_frames"),
        ({"burst": 0}, "inspector.line.burst"),
        ({"warmth_spread_max": 0.0}, "inspector.line.warmth_spread_max"),
        ({"max_speed_px_s": 0.0}, "inspector.line.max_speed_px_s"),
        ({"zone": [20, 50, 259, 180], **RIM_140}, "inspector.line.zone"),
        ({"burst": 4, **RIM_140}, "inspector.line.zone"),
    ],
)
def test_a_warmth_config_that_cannot_work_is_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path = warmth_config(tmp_path, **overrides)

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_a_zone_exactly_a_rim_plus_a_burst_of_travel_long_is_accepted(tmp_path: Path) -> None:
    load_config(warmth_config(tmp_path, zone=[20, 50, 260, 180], **RIM_140))


@pytest.mark.parametrize("camera", ["webcam", "phone"])
def test_a_warmth_station_on_a_live_camera_is_refused(tmp_path: Path, camera: str) -> None:
    path = warmth_config(tmp_path, camera=camera)

    with pytest.raises(ConfigError, match=re.escape(
        "inspector.line.warmth_min: needs camera.type replay: multi-frame capture is only "
        "available from recorded video for now"
    )):
        load_config(path)


def test_a_warmth_station_on_recorded_video_is_accepted(tmp_path: Path) -> None:
    load_config(warmth_config(tmp_path, camera="replay"))
