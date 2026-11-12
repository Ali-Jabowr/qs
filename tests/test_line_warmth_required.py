"""Every line station judges an empty cup by median warmth over a burst (#135, ADR-0029 decisions
4-5): `blob_share` is gone, so the warmth keys are required, and a config that still sets
`blob_share_min` is told which keys replaced it."""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.config import load_config
from qs.errors import ConfigError
from warmth_fixtures import warmth_config


def test_a_config_that_still_sets_blob_share_min_is_told_to_use_the_warmth_keys(
    tmp_path: Path,
) -> None:
    path = warmth_config(tmp_path, blob_share_min=0.3)

    with pytest.raises(
        ConfigError,
        match=(
            r"^inspector\.line\.blob_share_min: no longer read; replaced by "
            r"inspector\.line\.warmth_min, inspector\.line\.burst, "
            r"inspector\.line\.warmth_spread_max, inspector\.line\.max_speed_px_s$"
        ),
    ):
        load_config(path)


@pytest.mark.parametrize(
    "key", ["warmth_min", "burst", "warmth_spread_max", "max_speed_px_s", "direction", "fps"]
)
def test_a_line_config_without_a_warmth_key_is_refused(tmp_path: Path, key: str) -> None:
    path = warmth_config(tmp_path, **{key: None})

    with pytest.raises(ConfigError, match=rf"^missing key\(s\): inspector\.line\.{key}$"):
        load_config(path)
