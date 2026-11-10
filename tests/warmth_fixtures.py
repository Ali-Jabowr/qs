"""A line station config that judges by median warmth (#132), drawn at 320x240 and 8 fps.

A cup crosses at 40 px a frame, so `speed_px_s` is 320. Three frames of travel plus the widest
rim (140) are 260 px, and the zone is 280 wide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from station_fixtures import config_mapping

FPS = 8.0
WARMTH_LINE: dict[str, Any] = {
    "zone": [20, 50, 280, 180],
    "rim_axis_px": [50, 140],
    "blob_share_min": 0.3,
    "direction": "right",
    "fps": FPS,
    "warmth_min": 5.0,
    "burst": 3,
    "warmth_spread_max": 40.0,
    "max_speed_px_s": 320.0,
}


def warmth_config(root: Path, **line: Any) -> Path:
    """A cup-triggered line config checking `empty_cup` only; `line` overrides keys of the
    `inspector.line` block, and a value of None removes the key."""
    data = config_mapping(root, inspector_type="line")
    data["trigger"]["type"] = "cup"
    data["inspector"]["checks"] = ["empty_cup"]
    block = {**WARMTH_LINE, **line}
    data["inspector"]["line"] = {key: value for key, value in block.items() if value is not None}
    path = root / "line.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path
