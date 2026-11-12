"""Every line station sets `lens_mask_px` (#143). The mount always has a lens hole in its diffuser
(ADR-0029 decision 2), so a station file without the mask is refused rather than measuring warmth
over the hole's reflection."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config


def test_a_line_station_without_a_lens_mask_is_refused(tmp_path: Path) -> None:
    path = warmth_config(tmp_path, lens_mask_px=None)

    with pytest.raises(ConfigError, match=r"^missing key\(s\): inspector\.line\.lens_mask_px$"):
        load_config(path)
