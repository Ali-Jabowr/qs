"""The measured-rim and lens-mask keys of `inspector.line` (#133, ADR-0029 decisions 2-3).
`rim_size_px` is required (#135); a config without `lens_mask_px` masks nothing."""

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


def test_the_strict_keys_are_read(tmp_path: Path) -> None:
    path = warmth_config(
        tmp_path, rim_size_px=[70, 100], lens_mask_px=[160, 120, 20]
    )

    line = load_config(path).config.inspector.line

    assert line is not None
    assert (line.rim_size_px, line.lens_mask_px) == (
        (70, 100), (160, 120, 20),
    )


def test_a_config_without_the_lens_mask_has_none(tmp_path: Path) -> None:
    line = load_config(warmth_config(tmp_path)).config.inspector.line

    assert line is not None
    assert line.lens_mask_px is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rim_size_px": [0, 100]}, "inspector.line.rim_size_px"),
        ({"rim_size_px": [100, 70]}, "inspector.line.rim_size_px"),
        ({"rim_size_px": 100}, "inspector.line.rim_size_px"),
        ({"lens_mask_px": [160, 120]}, "inspector.line.lens_mask_px"),
        ({"lens_mask_px": [160, 120, 0]}, "inspector.line.lens_mask_px"),
        ({"lens_mask_px": [400, 120, 20]}, "inspector.line.lens_mask_px"),
        ({"lens_mask_px": [160, 120, 20], **NO_WARMTH}, "inspector.line.lens_mask_px"),
    ],
)
def test_a_strict_key_that_cannot_work_is_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path = warmth_config(tmp_path, **overrides)

    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rim_size_px": 100},
         r"inspector.line.rim_size_px: expected two integers \[minor, major\], got 100"),
        ({"lens_mask_px": [300, 120, 20]},
         r"inspector.line.lens_mask_px: \[280, 100, 41, 41\] is outside the 320x240 frame"),
        ({"lens_mask_px": [319, 120, 1]},
         r"inspector.line.lens_mask_px: \[318, 119, 3, 3\] is outside the 320x240 frame"),
    ],
)
def test_the_refusal_says_what_was_expected_and_counts_the_circle_s_last_pixel(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(warmth_config(tmp_path, **overrides))
