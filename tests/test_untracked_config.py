"""`inspector.line.untracked_window`: the fixed window of the self-check (#134, ADR-0029 §7).

A box [x, y, w, h] in the frame. It needs `warmth_min`, whose threshold it reuses, and a station
without it is read as before.
"""

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


def test_the_window_is_read(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, untracked_window=[130, 80, 80, 80])
    line = load_config(path).config.inspector.line

    assert line is not None
    assert line.untracked_window == (130, 80, 80, 80)


def test_a_config_without_the_window_has_none(tmp_path: Path) -> None:
    line = load_config(warmth_config(tmp_path)).config.inspector.line

    assert line is not None
    assert line.untracked_window is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"untracked_window": [130, 80, 80]}, "inspector.line.untracked_window"),
        ({"untracked_window": [130, 80, 0, 80]}, "inspector.line.untracked_window"),
        ({"untracked_window": [300, 80, 80, 80]}, "inspector.line.untracked_window"),
        ({"untracked_window": [130, 80, 80, 80], **NO_WARMTH}, "inspector.line.untracked_window"),
    ],
    ids=["three numbers", "no width", "outside the frame", "no warmth_min"],
)
def test_a_window_that_cannot_work_is_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path = warmth_config(tmp_path, **overrides)

    with pytest.raises(ConfigError, match=message):
        load_config(path)
