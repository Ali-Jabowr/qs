"""A line cup is judged on at least three burst frames (#145): ADR-0031 takes the median of
every burst frame, so `burst_min_frames` below 3 is refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config


@pytest.mark.parametrize("minimum", [1, 2])
def test_a_burst_minimum_under_three_frames_is_refused(tmp_path: Path, minimum: int) -> None:
    path = warmth_config(tmp_path, burst_min_frames=minimum)

    with pytest.raises(
        ConfigError, match=rf"^inspector\.line\.burst_min_frames: {minimum} is below 3"
    ):
        load_config(path)

