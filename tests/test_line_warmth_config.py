"""The line's warmth keys sit beside `blob_share_min` and change nothing for a config without
them (#132, ADR-0029 decision 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config

NO_WARMTH: dict[str, Any] = {
    "warmth_min": None, "burst": None, "warmth_spread_max": None, "max_speed_px_s": None,
}


def test_the_warmth_keys_are_read(tmp_path: Path) -> None:
    line = load_config(warmth_config(tmp_path)).config.inspector.line

    assert line is not None
    assert (line.warmth_min, line.burst, line.warmth_spread_max, line.max_speed_px_s) == (
        5.0, 3, 40.0, 320.0,
    )
    assert line.burst_min_frames == 3


def test_the_minimum_frames_can_be_set(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, burst=4, burst_min_frames=2, zone=[20, 50, 300, 180])

    line = load_config(path).config.inspector.line

    assert line is not None
    assert (line.burst, line.burst_min_frames) == (4, 2)


def test_a_config_without_the_warmth_threshold_keeps_blob_share(tmp_path: Path) -> None:
    line = load_config(warmth_config(tmp_path, **NO_WARMTH)).config.inspector.line

    assert line is not None
    assert (line.warmth_min, line.burst, line.warmth_spread_max, line.max_speed_px_s) == (
        None, None, None, None,
    )
    assert line.blob_share_min == 0.3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"burst": None}, "inspector.line.burst"),
        ({"warmth_spread_max": None}, "inspector.line.warmth_spread_max"),
        ({"max_speed_px_s": None}, "inspector.line.max_speed_px_s"),
        ({"direction": None}, "inspector.line.direction"),
        ({"warmth_min": None}, "inspector.line.burst"),
        ({"burst": 2}, "inspector.line.burst_min_frames"),
        ({"burst": 0}, "inspector.line.burst"),
        ({"warmth_spread_max": 0.0}, "inspector.line.warmth_spread_max"),
        ({"max_speed_px_s": 0.0}, "inspector.line.max_speed_px_s"),
        ({"zone": [20, 50, 259, 180]}, "inspector.line.zone"),
        ({"burst": 4}, "inspector.line.zone"),
    ],
)
def test_a_warmth_config_that_cannot_work_is_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path = warmth_config(tmp_path, **overrides)

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_a_zone_exactly_a_rim_plus_a_burst_of_travel_long_is_accepted(tmp_path: Path) -> None:
    load_config(warmth_config(tmp_path, zone=[20, 50, 260, 180]))
