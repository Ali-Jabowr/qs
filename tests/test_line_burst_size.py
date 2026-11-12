"""A line cup is judged on at least three burst frames (#145): ADR-0031 takes the median of the
burst, so a `burst` of 1 or 2 is refused by its own key, not blamed on `burst_min_frames`."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config


@pytest.mark.parametrize("burst", [1, 2])
def test_a_burst_under_three_frames_is_refused_by_its_own_key(tmp_path: Path, burst: int) -> None:
    path = warmth_config(tmp_path, burst=burst)

    with pytest.raises(ConfigError, match=rf"^inspector\.line\.burst: {burst} is below 3"):
        load_config(path)
