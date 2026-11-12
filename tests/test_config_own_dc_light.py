"""A line station that brings its own constant-current DC light may set an exposure that is not a
multiple of 10 ms at 50 Hz mains (ADR-0030): the light has no PWM, so mains flicker never
reaches the picture. `lighting.own_dc_light: true` is the declaration. Without it the flicker rule
stands, and a bench refuses the declaration. Seen through `load_config`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from qs.config import load_config
from qs.errors import ConfigError
from station_fixtures import config_mapping

FLICKER = (
    "camera.exposure_us: 1000 is not a multiple of 10000 "
    "(10 ms), required with 50 Hz mains to avoid flicker banding"
)
BENCH_REFUSAL = (
    "lighting.own_dc_light: only a line station may declare its own DC light — "
    "the bench keeps the flicker rule"
)


def _edited(path: Path, *, exposure_us: int, own_dc_light: object) -> Path:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["camera"]["power_line_hz"] = 50
    data["camera"]["exposure_us"] = exposure_us
    if own_dc_light is not None:
        data["lighting"]["own_dc_light"] = own_dc_light
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _config_file(root: Path) -> Path:
    data = config_mapping(root, inspector_type="line")
    data["trigger"]["type"] = "cup"
    data["inspector"]["checks"] = ["empty_cup"]
    data["inspector"]["line"] = {
        "zone": [20, 30, 240, 180],
        "rim_size_px": [105, 105],
        "blob_share_min": 0.3,
        "direction": "right",
        "fps": 8.0,
    }
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _bench_file(root: Path) -> Path:
    path = root / "bench.yml"
    path.write_text(yaml.safe_dump(config_mapping(root), allow_unicode=True), encoding="utf-8")
    return path


def test_a_line_station_with_its_own_dc_light_loads_a_1_ms_exposure(tmp_path: Path) -> None:
    path = _edited(_config_file(tmp_path), exposure_us=1000, own_dc_light=True)
    loaded = load_config(path)
    assert loaded.config.camera.exposure_us == 1000
    assert loaded.config.lighting.own_dc_light is True


@pytest.mark.parametrize("declared", [None, False])
def test_a_line_station_without_the_declaration_keeps_the_flicker_rule(
    tmp_path: Path, declared: object
) -> None:
    path = _edited(_config_file(tmp_path), exposure_us=1000, own_dc_light=declared)
    with pytest.raises(ConfigError, match="^" + re.escape(FLICKER)):
        load_config(path)


def test_a_bench_station_may_not_declare_its_own_dc_light(tmp_path: Path) -> None:
    path = _edited(_bench_file(tmp_path), exposure_us=10000, own_dc_light=True)
    with pytest.raises(ConfigError, match="^" + re.escape(BENCH_REFUSAL)):
        load_config(path)


def test_the_declaration_must_be_true_or_false(tmp_path: Path) -> None:
    path = _edited(_config_file(tmp_path), exposure_us=10000, own_dc_light="yes")
    with pytest.raises(ConfigError, match=r"^lighting\.own_dc_light: expected a boolean, got str"):
        load_config(path)
