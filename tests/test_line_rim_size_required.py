"""`rim_size_px` is the only rim size of a line station (#135, ADR-0029 decision 3): the wide
`rim_axis_px` range is gone, and a config that still sets it is told to use `rim_size_px`."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config


def test_a_config_that_still_sets_rim_axis_px_is_told_to_use_rim_size_px(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, rim_axis_px=[50, 140])

    with pytest.raises(
        ConfigError,
        match=r"inspector\.line\.rim_axis_px: .*replaced by inspector\.line\.rim_size_px",
    ):
        load_config(path)


def test_a_line_config_without_rim_size_px_is_refused(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, rim_size_px=None)

    with pytest.raises(ConfigError, match=r"missing key\(s\): inspector\.line\.rim_size_px"):
        load_config(path)


def test_a_rim_too_small_to_measure_inside_is_refused(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, rim_size_px=[15, 100])

    with pytest.raises(ConfigError, match=r"inspector\.line\.rim_size_px: minor 15 is below 16"):
        load_config(path)
